import itertools
from typing import Tuple, Iterator, Union, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import BatchEncoding

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
        device = self.compatible_head_model.device
        sample = {k: v.to(device) for k, v in self.samples_queue.pop().items()
                  if k not in ("oid", "labels", "decoder_input_ids")}
        outputs = self.compatible_head_model.generate(**sample,
                                                      # do_sample=True,
                                                      # top_k=5,
                                                      num_beams=1,
                                                      output_scores=True,
                                                      return_dict_in_generate=True,
                                                      num_return_sequences=1)
        return outputs

    def sample_one(self, inputs: Union[Dict[str, torch.LongTensor], BatchEncoding],
                   top_k_sampling: Optional[int] = 3,
                   greedy: Optional[bool] = False) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        device = self.compatible_head_model.device
        seq = []
        seq_probs = []
        with self.tokenizer.as_target_tokenizer():
            init_outputs = self.tokenizer("", return_tensors="pt").input_ids
            decoder_input = self.compatible_head_model.prepare_decoder_input_ids_from_labels(init_outputs).tolist()

        encoder_outputs = self.compatible_head_model.model.encoder(**inputs)
        past = None

        while True:
            decoder_input_t = torch.tensor(decoder_input).to(device)

            outputs = self.compatible_head_model(attention_mask=inputs["attention_mask"],
                                                 input_ids=None,
                                                 past_key_values=past,
                                                 decoder_input_ids=decoder_input_t,
                                                 encoder_outputs=encoder_outputs)
            tokens_probs = outputs.logits.softmax(-1)
            if greedy:
                # greedy search
                next_token_id = tokens_probs.argmax(-1)
                next_token_prob = tokens_probs[0, 0, next_token_id]
            else:
                if top_k_sampling is not None:
                    # top-k sampling: prob mass is distributed over top-k logits
                    top_k_args = tokens_probs.argsort(-1, descending=True)[:, :, :top_k_sampling]
                    tokens_probs = tokens_probs.gather(-1, top_k_args).softmax(-1)

                next_token_distr = torch.distributions.Categorical(tokens_probs)
                sampled_next_token_rank = next_token_distr.sample().item()
                if top_k_sampling is not None:
                    next_token_id = top_k_args[0, 0, sampled_next_token_rank].item()
                    next_token_prob = tokens_probs[0, 0, sampled_next_token_rank]
                else:
                    next_token_id = sampled_next_token_rank
                    next_token_prob = tokens_probs[0, 0, next_token_id]

            seq.append(next_token_id)
            seq_probs.append(next_token_prob)

            past = outputs.past_key_values

            if next_token_id == self.tokenizer.eos_token_id or len(seq) > 3 * len(inputs["input_ids"][0]):
                agg_prob = torch.hstack(seq_probs).mean()
                return torch.tensor(seq), agg_prob  # type: ignore

            decoder_input[0] = [next_token_id]

    def do_sample(self,
                  inputs: BatchEncoding,
                  num_samples: int) -> Iterator[Tuple[torch.LongTensor, torch.FloatTensor]]:
        """
        Samples a given set of translations using Monte Carlo.
        :return an iterator over tuples of [selected sequence, sequence score]
        """
        sampled = 0
        while sampled < num_samples:
            batch_attributes = inputs.keys()

            for input_tuple in zip(*(inputs[attr] for attr in batch_attributes)):
                input_tuple_batch = (i.unsqueeze(0) for i in input_tuple)
                yield self.sample_one(dict(zip(batch_attributes, input_tuple_batch)))

            sampled += 1

    def _static_sample(self, lm_logit_outputs: torch.FloatTensor,
                       num_samples: int = 10) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        # TODO: beam search can be performed on statically-inferred logits, with lower accuracy but significantly faster
        pass

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 100) -> torch.FloatTensor:
        # github.com/pytorch/fairseq/blob/23adb0c110fdd5e9166b3939987c5d26df996ec3/fairseq/criterions/sequence_risk_criterion.py#L44
        # + https://aclanthology.org/P16-1159.pdf
        # Questions:
        # impact of sample size?
        # importance of beam search? Can be replaced with faster argmax decoding?

        # outputs = self._dynamic_beam_search(num_samples)
        device = self.compatible_head_model.device
        sample = {k: v.to(device) for k, v in self.samples_queue.pop().items()
                  if k not in ("oid", "labels", "decoder_input_ids")}

        hypotheses, scores, evaluations = [], [], []
        refs = itertools.chain(*([ref] * num_samples for ref in self.tokenizer.batch_decode(labels,
                                                                                            skip_special_tokens=True)))
        trained_metric = self.evaluators["train"][0]

        for ref, (seq_ids, score) in zip(refs, self.do_sample(sample, num_samples)):
            hyp = self.tokenizer.decode(seq_ids)
            hypotheses.append(self.tokenizer.decode(seq_ids))
            scores.append(score)

            evaluations.append(trained_metric.evaluate_str([ref], [hyp]))
        scores = torch.vstack(scores)
        scores_scaled = scores.flatten() / scores.sum()

        # ignore-loss tokens (-100) are replaced with padding, and removed in batch_decode
        labels[labels < 0] = self.tokenizer.pad_token_id

        ref_evaluations = torch.tensor(evaluations).to(device)

        # BLEU is scaled to <0, 100>, we want to keep it that way for reporting, but we scale it for the loss
        if isinstance(trained_metric, BLEU):
            ref_evaluations = ref_evaluations / 100
        if not trained_metric.smaller_is_better:
            expected_risk = 1 - ref_evaluations
        else:
            expected_risk = ref_evaluations

        # loss = (scores_scaled * expected_risk).mean()
        loss = torch.nn.L1Loss()(scores_scaled, ref_evaluations.softmax(-1))
        # torch.nn.L1Loss()(scores_scaled * expected_risk.softmax(-1), torch.tensor(0))

        return loss


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
