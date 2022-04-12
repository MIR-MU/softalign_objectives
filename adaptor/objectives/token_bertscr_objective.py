import itertools
from typing import Tuple, Iterator, Union, Dict, Optional, List

import torch
import torch.nn.functional as F
from adaptor.adapter import Adapter
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer, BertTokenizer

from adaptor.evaluators.generative import BLEU, BERTScore
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.objectives.seq2seq_soft import MinimumRiskTraining
from adaptor.objectives.seq_bertscr_objectives import BERTScoreObjectiveBase


class TokenBertScoreObjective(BERTScoreObjectiveBase):

    def __init__(self, *args, num_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        padded_shape = self.compatible_head_model.config.vocab_size - num_samples
        # self.label_pad = torch.zeros(padded_shape, dtype=torch.float, requires_grad=True)

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 3,
                      ignored_label: int = -100) -> torch.Tensor:
        batch = {k: v.to(self.device) for k, v in next(self.own_iterator).items()}
        input_batch = {k: v for k, v in batch.items() if k not in ("oid", "labels")}

        loss_inst = torch.nn.CrossEntropyLoss()

        outputs = self.compatible_head_model(**input_batch)
        topk_logits, topk_indices = outputs.logits.topk(10, dim=-1)

        loss = torch.tensor(0., requires_grad=True, device=self.device)

        targets_per_sample = torch.empty(0, lm_logit_outputs.shape[-1], device=self.device)
        for ref_ids, sample_pred_tokens, logits in zip(batch["labels"].tolist(), topk_indices.tolist(), outputs.logits):
            for pos in range(len(ref_ids)):
                # concat the previous tokens, a predicted token, and the succeeding tokens
                prev_ids = ref_ids[:pos]
                current_predicted_ids = sample_pred_tokens[pos]
                next_ids = ref_ids[pos + 1:]
                hyps_ids = [prev_ids + [pred_id] + next_ids for pred_id in current_predicted_ids]

                with torch.no_grad():
                    hyps = self.tokenizer.batch_decode(hyps_ids)
                    hyps_embedder_ids, hyps_embeddings = self._embeddings_for_text(hyps)

                    ref = self.tokenizer.decode(ref_ids)
                    _, refs_embeddings = self._embeddings_for_text([ref])
                    refs_embeddings = refs_embeddings[0]

                refs_embeddings.requires_grad_(True)
                hyps_embeddings.requires_grad_(True)

                pos_dists = []
                for hyp_ids, hyp_emb_ids, hyp_emb in zip(hyps_ids, hyps_embedder_ids["input_ids"], hyps_embeddings):
                    if hyp_ids[pos] in self.tokenizer.all_special_ids:
                        if pos < len(hyp_ids) - 1:
                            # we penalize all special tokens except the ones terminating the hyp (to avoid repetition)
                            # but we use mean dist as penalisation, not to break the following normalisation
                            special_symbol_dist = torch.hstack(pos_dists).mean().item()
                        else:
                            special_symbol_dist = 0.
                        pos_dists.append(torch.tensor([special_symbol_dist], requires_grad=True, device=self.device))
                        continue

                    own_id_pos, ref_id_pos, hyp_pos_dist = self._distances_for_hyp_ids(refs_embeddings,
                                                                                       hyp_ids,
                                                                                       hyp_emb_ids,
                                                                                       hyp_emb,
                                                                                       start_own_pos=pos,
                                                                                       end_own_pos=pos + 1)
                    pos_dists.append(hyp_pos_dist)

                pos_dists_t = torch.hstack(pos_dists)
                # scoring of hypotheses variations on a given position:
                # list(zip(self.tokenizer.batch_decode(current_predicted_ids), (pos_dists_t / pos_dists_t.max()).tolist()))
                pos_dists_scaled = pos_dists_t / pos_dists_t.max(-1).values
                pos_targets_adjusted = 1 - pos_dists_scaled
                assert (pos_dists_scaled <= 1).all() and (pos_dists_scaled >= 0).all(), \
                    "Some computed targets have strange range!"
                # done: adjusted targets are on different positions
                pos_targets = torch.zeros_like(lm_logit_outputs[0, 0], device=self.device)
                pos_targets[current_predicted_ids] = pos_targets_adjusted
                # done: distances after softmax do not vary - all are too small!
                # TODO: do targets even require_grad? Now they do.
                targets_per_sample = torch.vstack([targets_per_sample, pos_targets])

            sample_loss = loss_inst(logits, targets_per_sample)

            loss = loss + sample_loss
        # targets = torch.hstack([targets_per_sample, self.label_pad.expand(targets_per_sample.shape[0], -1)])
        # targets_batched = targets.resize_as(topk_logits)
        # done: check that argmax(targets_batched) mostly match the reference_ids
        return loss / len(outputs.logits)


class TokenBertScoreObjectiveOld(Sequence2Sequence):
    bertscore_model = "bert-base-multilingual-cased"
    multinomial: bool
    sample_trials: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "lang_model_name_or_path" in kwargs:
            from transformers import AutoModelWithLMHead
            self.lang_model = AutoModelWithLMHead.from_pretrained(kwargs["lang_model_name_or_path"])
        else:
            from bert_score import get_model
            from bert_score import model2layers

            self.lang_model = get_model(self.bertscore_model,
                                        num_layers=model2layers[self.bertscore_model],
                                        all_layers=False)
            self.lang_model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.multinomial = kwargs["multinomial"] if "multinomial" in kwargs else False

        # TODO: impact of the sample size?
        self.sample_trials = kwargs["sample_trials"] if "sample_trials" in kwargs else 16

    def _lowest_pairwise_dists(self, ref_tokens: torch.LongTensor, hyp_tokens: torch.LongTensor,
                               loss_fn: torch.nn.Module = torch.nn.MSELoss()) -> torch.FloatTensor:
        with torch.no_grad():
            ref_tokens_padded = torch.clone(ref_tokens)
            ref_tokens_padded[ref_tokens_padded < 0] = self.tokenizer.pad_token_id
            ref_embeddings = self.lang_model(input_ids=ref_tokens_padded).last_hidden_state
            # model omits one dimension for batch_size=1

        per_samples_embeddings = []
        # re-batch, in order to be able to sample more
        for batch_hyp in hyp_tokens:
            per_samples_embeddings.append(self.lang_model(input_ids=batch_hyp).last_hidden_state)

        per_samples_min_idxs = []
        for ref_embeddings_one, hyps_embeddings in zip(ref_embeddings, per_samples_embeddings):
            dists = torch.cdist(ref_embeddings_one, hyps_embeddings, p=1)
            pairwise_euclid_dists, idx = dists.min(-2)
            per_samples_min_idxs.append(idx)

        loss_agg = torch.tensor(0., requires_grad=True)
        for batch_i, batch_min_idx in enumerate(per_samples_min_idxs):
            min_dist_ref_embeddings = ref_embeddings[batch_i, batch_min_idx]
            hyp_embeddings = per_samples_embeddings[batch_i]
            loss_agg = loss_agg + loss_fn(hyp_embeddings, min_dist_ref_embeddings)

        return loss_agg

    def _compute_loss(self, lm_logit_outputs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """This loss does the following sequence of steps:
        1. infer contextualized embeddings (e) for each token of translation hypothesis (e_hyp) and reference (e_ref)
        2. for each hypothesis token, find the best-matching token from reference, based on their embedding distance.
           as t_i_ref = argmin(cos(e_hyp, e_t_ref) for t in ref_tokens)
        2. computes a loss as a sum of these minimal distances:
           L = sum(cos(e_j_hyp, e_i_ref) for e_j_hyp in e_hyp)
        """
        # hypotheses generation
        output_log_probs = F.log_softmax(lm_logit_outputs, dim=-1)
        top_k_output_ids = output_log_probs.argsort(descending=True)[..., :self.sample_trials]

        # we do not care about the actual tokens log-probs now
        # top_k_output_log_probs = output_log_probs.gather(-1, top_k_output_ids)

        # per-score-rank-sequences are passed for embeddings inference - a corruption on lower ranks affects rank
        # proportionally to its aggregated score
        per_rank_output_sequences = top_k_output_ids.transpose(-1, -2)

        # # weighting by token probs - we do not do that for now
        # top_k_output_dists = self._lowest_pairwise_dists(labels, per_rank_output_sequences)
        # # transpose distances back and weight them by a token-level confidence of the model
        # log_loss = top_k_output_log_probs + top_k_output_dists.log().transpose(-1, -2)
        # return log_loss.exp().mean()

        loss_agg = self._lowest_pairwise_dists(labels, per_rank_output_sequences)

        return loss_agg
