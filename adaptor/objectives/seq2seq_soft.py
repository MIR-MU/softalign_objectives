from typing import Tuple, Iterator

import torch
import torch.nn.functional as F

from adaptor.evaluators.generative import BLEU
from adaptor.objectives.seq2seq import Sequence2Sequence


class MinimumRiskTraining(Sequence2Sequence):
    # https://aclanthology.org/P16-1159.pdf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "smoothing_alpha" in kwargs:
            self.smoothing_alpha = kwargs["smoothing_alpha"]
        else:
            self.smoothing_alpha = 5e-3
        self.samples_queue = []

    def _get_inputs_iterator(self, split: str) -> Iterator:
        for sample in super()._get_inputs_iterator(split):
            # this objective needs to remember its inputs, to be able to conditionally generate
            self.samples_queue.append(sample)
            yield sample

    def _dynamic_beam_search(self, num_samples: int = 10):
        # non-regressive inference -> all candidates are retrieved
        # from the initial-inference logits (w/out decoder_ids)

        # see https://huggingface.co/blog/how-to-generate#top-k-sampling
        sample = {k: v for k, v in self.samples_queue.pop().items() if k not in ("oid", "labels")}
        outputs = self.compatible_head_model.generate(**sample,
                                                      # do_sample=True,
                                                      # top_k=3,
                                                      num_beams=5,
                                                      output_scores=True,
                                                      return_dict_in_generate=True,
                                                      num_return_sequences=num_samples)
        return outputs

    def _static_sample(self, lm_logit_outputs: torch.FloatTensor,
                       num_samples: int = 10) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        # TODO: beam search can be performed on statically-inferred logits, with lower accuracy but significantly faster
        pass

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 1) -> torch.FloatTensor:
        # github.com/pytorch/fairseq/blob/23adb0c110fdd5e9166b3939987c5d26df996ec3/fairseq/criterions/sequence_risk_criterion.py#L44
        # + https://aclanthology.org/P16-1159.pdf
        # Questions:
        # impact of sample size?
        # importance of beam search? Can be replaced with faster argmax decoding?

        outputs = self._dynamic_beam_search(num_samples)

        trained_eval_metric = self.evaluators["train"][0]

        # ignore-loss tokens (-100) are replaced with padding, and removed in batch_decode
        labels[labels < 0] = self.tokenizer.pad_token_id

        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        ref_evaluations = torch.tensor([trained_eval_metric.evaluate_str([ref], [hyp]) for ref, hyp in zip(refs, hyps)],
                                       requires_grad=True)

        # BLEU is scaled to <0, 100>, we want to keep it that way for reporting, but we scale it for the loss
        if isinstance(trained_eval_metric, BLEU):
            ref_evaluations = ref_evaluations / 100

        sequences_loss = torch.abs(outputs.sequences_scores - ref_evaluations.log())

        return sequences_loss.exp().mean()


class MinimumFlow(Sequence2Sequence):

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
