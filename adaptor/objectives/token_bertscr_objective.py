from itertools import count
import logging
from typing import Tuple, Iterator, Union, Dict, Optional, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from adaptor.adapter import Adapter
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer, BertTokenizer

from adaptor.evaluators.generative import BLEU, BERTScore
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.objectives.seq2seq_soft import MinimumRiskTraining
from adaptor.objectives.seq_bertscr_objectives import BERTScoreObjectiveBase


logger = logging.getLogger()


class TokenBertScoreObjective(BERTScoreObjectiveBase):

    def __init__(self, *args, num_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        padded_shape = self.compatible_head_model.config.vocab_size - num_samples
        # self.label_pad = torch.zeros(padded_shape, dtype=torch.float, requires_grad=True)

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      num_samples: int = 3,
                      ignored_label: int = -100) -> torch.Tensor:

        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels")}

        loss_inst = torch.nn.CrossEntropyLoss()

        # outputs = self.compatible_head_model(**input_batch)
        topk_logits, topk_indices = lm_logit_outputs.topk(self.num_samples, dim=-1)

        loss = torch.tensor(0., requires_grad=True, device=self.device)

        targets_per_sample = torch.empty(0, lm_logit_outputs.shape[-1], device=self.device)
        for ref_ids, sample_pred_tokens, logits in zip(inputs["labels"].tolist(), topk_indices.tolist(),
                                                       lm_logit_outputs):
            with torch.no_grad():
                ref = self.tokenizer.decode(ref_ids)
                _, refs_embeddings = self._embeddings_for_text([ref])
                refs_embeddings = refs_embeddings[0]

            refs_embeddings.requires_grad_(True)

            # TODO: maybe remove me:
            for pos in range(len(ref_ids)):
                # concat the previous tokens, a predicted token, and the succeeding tokens
                prev_ids = ref_ids[:pos]
                current_predicted_ids = sample_pred_tokens[pos]
                next_ids = ref_ids[pos + 1:]
                hyps_ids = [prev_ids + [pred_id] + next_ids for pred_id in current_predicted_ids]

                with torch.no_grad():
                    hyps = self.tokenizer.batch_decode(hyps_ids)
                    hyps_embedder_ids, hyps_embeddings = self._embeddings_for_text(hyps)

                hyps_embeddings.requires_grad_(True)

                pos_dists = []
                for hyp_ids, hyp_emb_ids, hyp_emb in zip(hyps_ids, hyps_embedder_ids["input_ids"], hyps_embeddings):
                    if hyp_ids[pos] in self.tokenizer.all_special_ids:
                        if pos < len(hyp_ids) - 1:
                            # we penalize all special tokens except the ones terminating the hyp (to avoid repetition)
                            # but we use mean dist as penalisation, not to break the later normalisation
                            special_symbol_dist = -1.
                        else:
                            special_symbol_dist = 0.
                        pos_dists.append(torch.tensor([special_symbol_dist], device=self.device))
                        continue
                    # TODO: don't do this per-position! Reducing complexity from |pos|ˆ2/2 to |pos|
                    own_id_pos, ref_id_pos, hyp_pos_dist = self._distances_for_hyp_ids(refs_embeddings,
                                                                                       hyp_ids,
                                                                                       hyp_emb_ids,
                                                                                       hyp_emb,
                                                                                       start_own_pos=pos,
                                                                                       end_own_pos=pos + 1)
                    if len(hyp_pos_dist) < 1:
                        logger.warning("Misalignment on pos %s inputs %s." % (pos, inputs))
                        logger.warning("This is usually due to a generation of a special token within the first %s "
                                       "positions. Expecting target=0. for this position.", self.num_samples)
                        hyp_pos_dist = torch.tensor(2., device=self.device)

                        # TODO: remove me:
                        # self._distances_for_hyp_ids(refs_embeddings,
                        #                             hyp_ids,
                        #                             hyp_emb_ids,
                        #                             hyp_emb,
                        #                             start_own_pos=pos,
                        #                             end_own_pos=pos + 1)

                    pos_dists.append(hyp_pos_dist)

                pos_dists_t = torch.hstack(pos_dists)

                if torch.any(pos_dists_t > 1):
                    # except the sanctioned item
                    # adjust so-assigned max distances by the other items max, to avoid making "trues" of everything
                    valid_dists = pos_dists_t[(0 <= pos_dists_t) & (pos_dists_t <= 1)]

                    if valid_dists.numel():
                        pos_dists_t[pos_dists_t > 1] = torch.tensor(valid_dists.max().item(), requires_grad=True)
                    else:
                        pos_dists_t[pos_dists_t > 1] = torch.tensor(1., requires_grad=True)

                if torch.any(pos_dists_t < 0):
                    # assign min in-batch value for special tokens in the middle of hypotheses,
                    # again, in order not to spoil normalisation
                    valid_dists = pos_dists_t[(0 <= pos_dists_t) & (pos_dists_t <= 1)]

                    if valid_dists.numel():
                        pos_dists_t[pos_dists_t < 0] = torch.tensor(valid_dists.min().item(), requires_grad=True)
                    else:
                        pos_dists_t[pos_dists_t < 0] = torch.tensor(0., requires_grad=True)
                if pos_dists_t.max() != 0:
                    pos_dists_scaled = pos_dists_t / pos_dists_t.max(-1).values
                else:
                    # otherwise, we have all zeros => presumably, all hypotheses are bad
                    pos_dists_scaled = pos_dists_t
                pos_targets_adjusted = 1 - pos_dists_scaled
                # scoring of hypotheses variations on a given position:
                # list(zip(self.tokenizer.batch_decode(current_predicted_ids), pos_targets_adjusted.tolist()))
                assert ((pos_dists_scaled <= 1).all() and (pos_dists_scaled >= 0).all()), \
                    "Some computed targets have strange range! \nDistances:%s\nTargets: %s" \
                    % (pos_dists, pos_dists_scaled)
                # done: adjusted targets are on different positions
                pos_targets = torch.zeros_like(lm_logit_outputs[0, 0], device=self.device)
                pos_targets[current_predicted_ids] = pos_targets_adjusted
                # done: distances after softmax do not vary - all are too small!
                # TODO: do targets even require_grad? Now they do.
                targets_per_sample = torch.vstack([targets_per_sample, pos_targets])

                # here ends no_grad()

            sample_loss = loss_inst(logits, targets_per_sample)
            loss = loss + sample_loss
        # targets = torch.hstack([targets_per_sample, self.label_pad.expand(targets_per_sample.shape[0], -1)])
        # targets_batched = targets.resize_as(topk_logits)
        # done: check that argmax(targets_batched) mostly match the reference_ids
        return loss / len(lm_logit_outputs)


class DeconTokenBertScoreObjective(BERTScoreObjectiveBase):

    def __init__(self, *args, emb_infer_batch_size: int = 32, emb_size: int = 768, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: here goes inference of decontextualized embeddings
        # 1. per-batch inference of embeddings for all references
        source_texts, ref_texts = self._per_split_iterators("train")
        ref_texts = list(ref_texts)
        # 2. per-batch append to a dict of Marian sentencepieces, recast to CPU
        # TODO: this could be further optimised - index is a single big matrix
        self.spiece_embeddings = torch.zeros(len(self.tokenizer.decoder.keys()), emb_size)
        spiece_counts = [0 for spiece in self.tokenizer.decoder.keys()]

        for batch_offset in tqdm(range(0, len(ref_texts), emb_infer_batch_size),
                                 desc="%s: Inferring embeddings for decontextualization" % self,
                                 total=len(ref_texts) // emb_infer_batch_size):
            batch_texts = ref_texts[batch_offset: batch_offset + emb_infer_batch_size]
            with torch.no_grad():
                ref_emb_inputs, ref_embs = self._embeddings_for_text(batch_texts)

            ref_own_inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation="longest_first", padding=True)

            for i, item_emb_ids, item_own_ids in zip(count(), ref_emb_inputs.input_ids, ref_own_inputs.input_ids):
                own_indices, embedder_indices = self._get_own_to_embedder_alignment(item_own_ids, item_emb_ids, 0)

                spiece_batch_ids = ref_own_inputs.input_ids[i, own_indices]  # type: ignore
                spiece_batch_embeddings = ref_embs[i, embedder_indices]

                # 3. averaging
                for spiece, spiece_emb in zip(spiece_batch_ids, spiece_batch_embeddings):
                    if not spiece_counts[spiece.item()]:
                        spiece_counts[spiece.item()] += 1
                        self.spiece_embeddings[spiece.item()] = spiece_emb
                    else:
                        spiece_counts[spiece.item()] += 1
                        # weighted average: we need the counter already updated here
                        new_emb_weight = 1 / spiece_counts[spiece.item()]
                        new_emb_weighted = spiece_emb * new_emb_weight

                        orig_emb_weight = 1 - new_emb_weight
                        orig_emb_weighted = self.spiece_embeddings[spiece.item()] * orig_emb_weight
                        # TODO: are the embeddings not too different?
                        # assert on first addition:
                        # torch.vstack([spiece_emb, self.spiece_embeddings[spiece.item()]]).mean(0) ==
                        # orig_emb_weighted + new_emb_weighted
                        self.spiece_embeddings[spiece.item()] = orig_emb_weighted + new_emb_weighted

        # TODO: do we want embeddings as leafs?
        self.spiece_embeddings.requires_grad_(True)
        self.spiece_counts = torch.tensor(spiece_counts, dtype=torch.int32)
        logger.warning("Indexation done. %s nonzero embeddings, averaged from %s embeddings"
              % (sum(bool(count) for count in spiece_counts), sum(spiece_counts)))

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        # 1. Construct a matrix: sample X position X top-k logit index (remember the index separately):
        # print(lm_logit_outputs.shape)
        # TODO: we could relatively easily penalize random matches to distant positions -- logits are ordered
        indexed_tokens = torch.where(self.spiece_counts > 0)[0]
        indexed_tokens_logits = lm_logit_outputs[..., indexed_tokens]
        indexed_tokens_embs = self.spiece_embeddings[indexed_tokens]
        with torch.no_grad():
            ref_emb_inputs, ref_embs = self._embeddings_for_text(self.tokenizer.batch_decode(labels))
        ref_embs.requires_grad_(True)

        # 2. Compute distances
        min_dists_to_reference, min_dist_positions = torch.cdist(indexed_tokens_embs, ref_embs).min(-1)

        min_dists_to_reference_normed = min_dists_to_reference / min_dists_to_reference.max(-1).values
        # TODO: min_dist_positions do not respect the ranking of the indexed tokens - maybe embeddings are wrong?
        # 3. targets = distances of k-first tokens, label smoothing for the remainder (see how)
        loss = torch.nn.CrossEntropyLoss()
        loss_val = loss(indexed_tokens_logits, (1 - min_dists_to_reference_normed.expand_as(indexed_tokens_logits)))
        return loss_val
