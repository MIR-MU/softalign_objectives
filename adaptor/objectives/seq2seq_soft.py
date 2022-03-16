from typing import Tuple, Iterator, Union, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer, BertTokenizer

from adaptor.evaluators.generative import BLEU, BERTScore
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

    def sample_n(self,
                 inputs: Union[Dict[str, torch.LongTensor], BatchEncoding],
                 num_samples: int,
                 top_k_sampling: Optional[int] = 3,
                 greedy: Optional[bool] = False) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        device = self.compatible_head_model.device
        seq = torch.empty(num_samples, 0, dtype=torch.int64).to(device)
        seq_probs = torch.empty(num_samples, 0, dtype=torch.float32).to(device)

        assert all(v.shape[0] == 1 for v in inputs.values()), "We perform parallel sampling only for inputs of size 1."
        inputs_parallel = {k: v.expand(num_samples, -1) for k, v in inputs.items()}

        with self.tokenizer.as_target_tokenizer():
            init_outputs = self.tokenizer("", return_tensors="pt").input_ids.repeat(num_samples, 1)
            decoder_input_l = self.compatible_head_model.prepare_decoder_input_ids_from_labels(init_outputs).tolist()
            decoder_input = torch.tensor(decoder_input_l).to(device)

        encoder_outputs = self.compatible_head_model.model.encoder(**inputs)
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.expand(num_samples, -1, -1)
        past = None

        while True:
            outputs = self.compatible_head_model(attention_mask=inputs_parallel["attention_mask"],
                                                 input_ids=None,
                                                 past_key_values=past,
                                                 decoder_input_ids=decoder_input,
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
                    tokens_probs_topk = tokens_probs.gather(-1, top_k_args)
                    tokens_probs = tokens_probs_topk.softmax(-1)
                    # TODO debug: erroreous sample:
                    # inputs = {'input_ids': tensor([[ 4086,    36,    44,    34,    31,   251,     7,     8,
                    # 72,   846, 2374,     9,     0, 62508, 62508, 62508]]),
                    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
                    # decoder_inputs = tensor([[62508], [62508], [62508], [62508], [62508], [62508],
                    # [62508], [62508], [62508], [62508]], device='cuda:0')
                    # past = None

                next_token_distr = torch.distributions.Categorical(tokens_probs)
                sampled_next_token_rank = next_token_distr.sample()
                if top_k_sampling is not None:
                    next_token_id = top_k_args[0, 0, sampled_next_token_rank]
                    next_token_prob = tokens_probs[0, 0, sampled_next_token_rank]
                else:
                    next_token_id = sampled_next_token_rank
                    next_token_prob = tokens_probs[0, 0, next_token_id]

            seq = torch.hstack([seq, next_token_id])
            seq_probs = torch.hstack([seq_probs, next_token_prob])

            past = outputs.past_key_values

            # termination check -> when all the sequences generated <eos>, we prune to the first <eos>,
            # aggregate scores for policy gradients and return sequences with their scores
            seq_terminated = (seq == self.tokenizer.eos_token_id).any(dim=1)
            generated_length = len(seq[0])
            if seq_terminated.all() or generated_length > 3 * len(inputs["input_ids"][0]):
                # prune sequences to the earliest-eos-token
                earliest_eoses = (seq == self.tokenizer.eos_token_id).long().argmax(-1)
                agg_probs = []
                for seq_idx, eos_idx in enumerate(earliest_eoses):
                    seq[seq_idx, eos_idx + 1:] = self.tokenizer.pad_token_id
                    # eos_idx+1 => including eos token prob
                    pruned_seq_prob = seq_probs[seq_idx, :eos_idx + 1].log().sum()
                    agg_probs.append(pruned_seq_prob)

                return seq, torch.hstack(agg_probs)  # type: ignore

            decoder_input = next_token_id

    def do_sample(self,
                  inputs: BatchEncoding,
                  num_samples: int) -> Iterator[Tuple[torch.LongTensor, torch.FloatTensor]]:
        """
        Samples a given set of translations using Monte Carlo.
        :return an iterator over tuples of [selected sequence, sequence score]
        """
        batch_attributes = inputs.keys()

        for input_tuple in zip(*(inputs[attr] for attr in batch_attributes)):
            input_tuple_batch = (i.unsqueeze(0) for i in input_tuple)
            yield self.sample_n(dict(zip(batch_attributes, input_tuple_batch)), num_samples)

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 3) -> torch.FloatTensor:
        # github.com/pytorch/fairseq/blob/23adb0c110fdd5e9166b3939987c5d26df996ec3/fairseq/criterions/sequence_risk_criterion.py#L44
        # + https://aclanthology.org/P16-1159.pdf
        # Questions:
        # impact of sample size?
        # importance of beam search? Can be replaced with faster argmax decoding?

        # outputs = self._dynamic_beam_search(num_samples)
        device = self.compatible_head_model.device
        batch = {k: v.to(device) for k, v in self.samples_queue.pop().items()}
        input_batch = {k: v for k, v in batch.items() if k not in ("oid", "labels", "decoder_input_ids")}

        hypotheses, scores, evaluations = [], [], []
        trained_metric = self.evaluators["train"][0]

        for one_label, (seq_ids, score) in zip(batch["labels"], self.do_sample(input_batch, num_samples)):
            hyps = self.tokenizer.batch_decode(seq_ids, skip_special_tokens=True)

            # ignore-loss tokens (-100) are replaced with padding, and removed in decoding
            one_reference = self.tokenizer.decode([l if l > 0 else 0 for l in one_label], skip_special_tokens=True)
            hypotheses.append(hyps)
            scores.append(score)

            evaluations.extend([trained_metric.evaluate_str([one_reference], [hyp]) for hyp in hyps])

        scores = torch.vstack(scores)
        scores_scaled = scores.flatten()

        ref_evaluations = torch.tensor(evaluations).to(device)

        # BLEU is scaled to <0, 100>, we want to keep it that way for reporting, but we scale it for the loss
        if isinstance(trained_metric, BLEU):
            ref_evaluations = ref_evaluations / 100
        if not trained_metric.smaller_is_better:
            expected_risk = 1 - ref_evaluations
        else:
            expected_risk = ref_evaluations

        loss = (scores_scaled * expected_risk.log()).mean()
        # loss = torch.nn.L1Loss()(scores_scaled, ref_evaluations.softmax(-1))
        # torch.nn.L1Loss()(scores_scaled * expected_risk.softmax(-1), torch.tensor(0))
        if not torch.isfinite(loss) or not 0 <= loss <= 10:
            return torch.tensor(0., requires_grad=True).to(loss.device)
        return loss


class TokenBertScoreObjective(MinimumRiskTraining):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = BERTScore()

    def sample_n(self,
                 inputs: Union[Dict[str, torch.LongTensor], BatchEncoding],
                 num_samples: int,
                 top_k_sampling: Optional[int] = 3,
                 top_n_candidates_to_return: Optional[int] = 10) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # the same top-k sampling as in MRT, but we add a list of top-n candidates from each run

        # collectors
        device = self.compatible_head_model.device
        seq = torch.empty(num_samples, 0, dtype=torch.int64).to(device)
        seq_probs = torch.empty(num_samples, 0, dtype=torch.float32).to(device)
        top_n_best_candidates = []
        top_n_probs = []

        assert all(v.shape[0] == 1 for v in inputs.values()), "We perform parallel sampling only for inputs of size 1."
        inputs_parallel = {k: v.expand(num_samples, -1) for k, v in inputs.items()}

        with self.tokenizer.as_target_tokenizer():
            init_outputs = self.tokenizer("", return_tensors="pt").input_ids.repeat(num_samples, 1)
            decoder_input_l = self.compatible_head_model.prepare_decoder_input_ids_from_labels(init_outputs).tolist()
            decoder_input = torch.tensor(decoder_input_l).to(device)

        encoder_outputs = self.compatible_head_model.model.encoder(**inputs)
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.expand(num_samples, -1, -1)
        past = None

        while True:
            outputs = self.compatible_head_model(attention_mask=inputs_parallel["attention_mask"],
                                                 input_ids=None,
                                                 past_key_values=past,
                                                 decoder_input_ids=decoder_input,
                                                 encoder_outputs=encoder_outputs)
            tokens_probs = outputs.logits.softmax(-1)
            tokens_ids_sorted = tokens_probs.argsort(-1, descending=True)
            per_sample_topn_candidates = tokens_ids_sorted[:, :, :top_n_candidates_to_return]
            top_n_best_candidates.append(per_sample_topn_candidates)

            per_sample_topn_probs = tokens_probs.gather(-1, per_sample_topn_candidates)
            top_n_probs.append(per_sample_topn_probs)

            # top-k sampling: prob mass is distributed over top-k logits
            top_k_args = tokens_ids_sorted[:, :, :top_k_sampling]
            tokens_probs_topk = tokens_probs.gather(-1, top_k_args)
            tokens_probs = tokens_probs_topk.softmax(-1)

            next_token_distr = torch.distributions.Categorical(tokens_probs)
            sampled_next_token_rank = next_token_distr.sample()

            next_token_ids = top_k_args[0, 0, sampled_next_token_rank]
            next_token_probs = tokens_probs[0, 0, sampled_next_token_rank]

            seq = torch.hstack([seq, next_token_ids])
            seq_probs = torch.hstack([seq_probs, next_token_probs])

            past = outputs.past_key_values

            # termination check -> when all the sequences generated <eos>, we prune to the first <eos>,
            # aggregate scores for policy gradients and return sequences with their scores
            seq_terminated = (seq == self.tokenizer.eos_token_id).any(dim=1)
            generated_length = len(seq[0])
            if seq_terminated.all() or generated_length > 3 * len(inputs["input_ids"][0]):
                # prune sequences to the earliest-eos-token
                earliest_eoses = (seq == self.tokenizer.eos_token_id).long().argmax(-1)
                agg_probs = []
                for seq_idx, eos_idx in enumerate(earliest_eoses):
                    seq[seq_idx, eos_idx + 1:] = self.tokenizer.pad_token_id
                    # eos_idx+1 => including eos token prob
                    pruned_seq_prob = seq_probs[seq_idx, :eos_idx + 1].log().sum()
                    agg_probs.append(pruned_seq_prob)

                # return seq, torch.hstack(agg_probs), torch.hstack(top_n_best_candidates), torch.hstack(top_n_probs)
                # TODO: refactor
                return seq, torch.hstack(agg_probs), torch.stack(top_n_best_candidates, dim=-1).squeeze(1), \
                       torch.stack(top_n_probs, dim=-1).squeeze(1)

            decoder_input = next_token_ids

    def do_sample(self,
                  inputs: BatchEncoding,
                  num_samples: int) -> Iterator[Tuple[torch.LongTensor, torch.FloatTensor]]:
        """
        Samples a given set of translations using Monte Carlo.
        :return an iterator over tuples of [selected sequence, sequence score]
        """
        batch_attributes = inputs.keys()

        for input_tuple in zip(*(inputs[attr] for attr in batch_attributes)):
            input_tuple_batch = (i.unsqueeze(0) for i in input_tuple)
            yield self.sample_n(dict(zip(batch_attributes, input_tuple_batch)), num_samples)

    @staticmethod
    def _erase_bert_tokenizer_extras(text: str, symbols: Tuple[str] = (" ", "#", "[CLS]")) -> str:
        out_text = text
        for symbol in symbols:
            out_text = out_text.replace(symbol, "")
        return out_text

    def _offsets_for_text_tokenizer(self,
                                    hyp_ids: torch.LongTensor,
                                    tokenizer: PreTrainedTokenizer) -> List[int]:
        if isinstance(tokenizer, BertTokenizer):
            lengths = [len(self._erase_bert_tokenizer_extras(subword))
                       for subword in tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)]
        else:
            lengths = [len(subword) for subword in tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)]
        cum_lengths = [lengths[0]]
        for ith_length in lengths[1:]:
            cum_lengths.append(cum_lengths[-1] + ith_length)
        return cum_lengths

    def _best_aligned_embedding_pos(self, hyp_ids: torch.LongTensor,
                                    hyp_pos: int, embedder_ids: torch.LongTensor) -> int:

        def _offsets_intersection(offsets1: Tuple[int, int], offsets2: Tuple[int, int]) -> int:
            min_len = min(offsets1[1] - offsets1[0], offsets2[1] - offsets2[0])

            if min_len == 0:
                # spaces are tokenized to a zero length => for these cases, we want to get the overlaying subword emb.
                if offsets1[0] <= offsets2[0] <= offsets2[1] <= offsets1[1]:
                    # offsets1 overlays offsets2 -> returning nonzero allows to identify current position
                    return 1
                elif offsets2[0] <= offsets1[0] <= offsets1[1] <= offsets2[1]:
                    # offsets2 overlays offsets1
                    return 1
                # no overlay
                return 0

            if offsets1[0] <= offsets2[0]:
                # the first one starts earlier
                if offsets1[1] <= offsets2[0]:
                    # the first one ends before the second starts
                    return 0
                else:
                    # the first one ends after the second starts
                    return min(min_len, offsets1[1] - offsets2[0])
            else:
                # the second one starts earlier, otherwise analogical
                if offsets2[1] <= offsets1[0]:
                    return 0
                else:
                    return min(min_len, offsets2[1] - offsets1[0])

        own_offsets = self._offsets_for_text_tokenizer(hyp_ids, self.tokenizer)
        embedder_offsets = self._offsets_for_text_tokenizer(embedder_ids, self.scorer.scorer._tokenizer)
        # search both (sorted) offset lists around the requested hyp_pos, find candidates and pick the best-intersector
        searched_offset = own_offsets[hyp_pos - 1], own_offsets[hyp_pos]
        best_match_intersection = 0

        last_match = hyp_pos
        for embedder_offset_dist in range(len(embedder_offsets)):
            fwd_idx = hyp_pos + embedder_offset_dist
            # TODO: calibrate indexes on an example
            fwd_offset = embedder_offsets[fwd_idx - 1], embedder_offsets[fwd_idx]
            fwd_intersection = _offsets_intersection(searched_offset, fwd_offset)

            bwd_idx = hyp_pos - embedder_offset_dist
            bwd_offset = embedder_offsets[bwd_idx - 1], embedder_offsets[bwd_idx]
            bwd_intersection = _offsets_intersection(searched_offset, bwd_offset)

            # no better match can follow after any intersection has been already found
            if max(fwd_intersection, bwd_intersection) < best_match_intersection:
                # we could do some automatic assert like:
                # self.tokenizer.decode(hyp_ids[2]), self.scorer.scorer._tokenizer.decode(embedder_ids[3])
                embedded_segment = [self._erase_bert_tokenizer_extras(s)
                                    for s in self.scorer.scorer._tokenizer.batch_decode(embedder_ids)][last_match]
                if embedded_segment in self.scorer.scorer._tokenizer.all_special_tokens:
                    embedded_segment = ""
                generated_segment = self.tokenizer.decode(hyp_ids[hyp_pos])

                if embedded_segment and generated_segment \
                        and embedded_segment not in self.scorer.scorer._tokenizer.all_special_tokens:
                    # assert there is some intersection for all nonempty segments
                    # TODO: check an impact of embedding empty segments by the closest embeddings
                    assert set(embedded_segment).intersection(generated_segment)

                return last_match

            if fwd_intersection > best_match_intersection:
                last_match = fwd_idx
                best_match_intersection = fwd_intersection
            if bwd_intersection > best_match_intersection:
                last_match = bwd_idx
                best_match_intersection = bwd_intersection

    def _evaluate_position_in_hyps(self, hyps: torch.LongTensor, predicted_pos: int,
                                   ref_embedding: torch.FloatTensor) -> torch.FloatTensor:
        hyps_text = self.tokenizer.batch_decode(hyps, skip_special_tokens=True)

        embedder_inputs = self.scorer.scorer._tokenizer(hyps_text, return_tensors="pt",
                                                        truncation="longest_first", padding=True).to(self.device)
        embeddings = self.scorer.scorer._model(**embedder_inputs)[0]

        embedder_ids_all = embedder_inputs["input_ids"]
        matched_embeddings_pos = torch.tensor([self._best_aligned_embedding_pos(own_ids, predicted_pos, embedder_ids)
                                               for own_ids, embedder_ids in zip(hyps, embedder_ids_all)])

        matched_embeddings_idx = matched_embeddings_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 768).to(self.device)
        matched_embeddings = embeddings.gather(1, matched_embeddings_idx).squeeze(1)

        embeddings_dists = torch.cdist(matched_embeddings, ref_embedding)
        embeddings_best_dists = embeddings_dists.min(-1).values
        embeddings_dists_scaled = (embeddings_best_dists - embeddings_best_dists.min()) / \
                                  (embeddings_best_dists.max() - embeddings_best_dists.min())
        embeddings_similarities = 1 - embeddings_dists_scaled

        return embeddings_similarities

    def _fwd_for_hypothesis(self,
                            input_batch: Dict[str, torch.LongTensor],
                            hyp_ids: torch.LongTensor) -> torch.FloatTensor:
        past_tokens = self.compatible_head_model.prepare_decoder_input_ids_from_labels(hyp_ids)
        return self.compatible_head_model(**input_batch, decoder_input_ids=past_tokens)

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 3,
                      ignored_label: int = -100) -> torch.FloatTensor:
        batch = {k: v.to(self.device) for k, v in self.samples_queue.pop().items()}
        input_batch = {k: v for k, v in batch.items() if k not in ("oid", "labels", "decoder_input_ids")}

        losses = []

        loss_inst = torch.nn.CrossEntropyLoss(reduction="none")

        for ref_ids, (hyps_ids, hyp_scores, candidates, tokens_scores) in zip(batch["labels"],
                                                                              self.do_sample(input_batch, num_samples)):
            hyps_text = self.tokenizer.batch_decode(hyps_ids, skip_special_tokens=True)
            ref_text = self.tokenizer.decode([l if l > 0 else 0 for l in ref_ids], skip_special_tokens=True)

            all_hyps_score = torch.tensor([self.scorer.evaluate_str([ref_text], [hyp_text]) for hyp_text in hyps_text])

            ref_embedder_inputs = self.scorer.scorer._tokenizer(ref_text, return_tensors="pt").to(self.device)
            ref_embeddings = self.scorer.scorer._model(**ref_embedder_inputs)[0][0]

            # generate pseudo-labels that we differentiate against tokens_scores
            for pos in range(2, hyps_ids.shape[-1]):
                # a first max val of zero/one tensor of <eos> matches
                hyps_termination_pos = (hyps_ids == self.tokenizer.eos_token_id).long().argmax(-1)
                non_terminated_hyps_mask = pos <= hyps_termination_pos

                pseudolabels = []
                hyps_quality = all_hyps_score[non_terminated_hyps_mask]

                for hyp, hyp_candidates, in zip(hyps_ids[non_terminated_hyps_mask], candidates[non_terminated_hyps_mask]):
                    pos_permutations_hyps = []
                    for cand_id in hyp_candidates[:, pos]:
                        permuted_hyp = torch.cat([hyp[:pos], torch.tensor([cand_id]).to(self.device), hyp[pos + 1:]], 0)
                        pos_permutations_hyps.append(permuted_hyp)
                    pos_permutations_hyps = torch.vstack(pos_permutations_hyps)
                    # TODO: check the transpositions
                    # TODO: second input, this returns different dim than input
                    hyps_similarities = self._evaluate_position_in_hyps(pos_permutations_hyps, pos, ref_embeddings)

                    # pseudo labels are a probability distribution over top-n target tokens within the context of top-1
                    pseudolabels.append(hyps_similarities)

                hyp_token_scores = tokens_scores[non_terminated_hyps_mask, :, pos]

                hyp_loss = loss_inst(hyp_token_scores, torch.vstack(pseudolabels))
                # loss is weighted by an overall BERTScore of the generated hypothesis
                loss_weighted = hyp_loss * hyps_quality

                losses.append(loss_weighted.mean())

        return torch.hstack(losses).mean()


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
