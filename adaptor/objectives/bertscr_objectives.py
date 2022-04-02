import itertools
from typing import Tuple, Iterator, Union, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer, BertTokenizer

from adaptor.evaluators.generative import BLEU, BERTScore
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.objectives.seq2seq_soft import MinimumRiskTraining


class BERTScoreObjectiveBase(Sequence2Sequence):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = BERTScore()

        self.our_single_sample = None

        self.samples_queue = []

    def _get_inputs_iterator(self, split: str) -> Iterator:
        for sample in super()._get_inputs_iterator(split):
            # this objective needs to remember its inputs, to be able to conditionally generate
            self.samples_queue.append(sample)
            # if self.our_single_sample is None:
            #     self.our_single_sample = sample
            yield sample
            # else:
            #     yield self.our_single_sample

    def _get_next_sample(self) -> Union[Dict[str, torch.LongTensor], BatchEncoding]:
        return self.samples_queue.pop()
        # return self.our_single_sample

    def sample_n(self,
                 inputs: Union[Dict[str, torch.LongTensor], BatchEncoding],
                 num_samples: int,
                 top_k_sampling: Optional[int] = 3,
                 greedy: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                try:
                    next_token_distr = torch.distributions.Categorical(tokens_probs)
                    sampled_next_token_rank = next_token_distr.sample()
                except RuntimeError as e:
                    # print("Runtime error on inputs %s" % inputs)
                    print("Runtime error on inputs - OOM error?")
                    raise e
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
                    # eos_idx+1 => including eos token prob to the aggregation
                    pruned_seq_prob = seq_probs[seq_idx, :eos_idx + 1].log().sum()
                    agg_probs.append(pruned_seq_prob)

                return seq, seq_probs, torch.hstack(agg_probs)

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


class SeqBertScoreObjective(BERTScoreObjectiveBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distances_pads = {i: torch.zeros(self.tokenizer.model_max_length - i,
                                              requires_grad=True, device=self.device)
                               for i in range(self.tokenizer.model_max_length)}
        self.scores_pads = {i: torch.ones(self.tokenizer.model_max_length - i,
                                          requires_grad=True, device=self.device)
                            for i in range(self.tokenizer.model_max_length)}

        self.distances_memory = []
        self.scores_memory = []

    def _get_inputs_iterator(self, split: str) -> Iterator:
        if split == "train":
            for sample in super()._get_inputs_iterator(split):
                # this objective needs to remember its inputs, to be able to conditionally generate
                self.samples_queue.append(sample)
                # if self.our_single_sample is None:
                #     self.our_single_sample = sample
                yield sample
                # else:
                #     yield self.our_single_sample
        else:
            self._truncate()
            return super()._get_inputs_iterator(split)

    def _erase_bert_tokenizer_extras(self,
                                     text: str,
                                     tokenizer: PreTrainedTokenizer,
                                     symbols: List[str] = (" ", "#")) -> str:
        out_text = text
        for symbol in itertools.chain(symbols, tokenizer.all_special_tokens):
            out_text = out_text.replace(symbol, "")
        return out_text

    def _intervals_for_text_tokenizer(self,
                                      tokenizer_ids: torch.Tensor,
                                      tokenizer: PreTrainedTokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
        # TODO: eliminate this tokenization completely?
        # TODO: could we use a dict of distances instead?
        if isinstance(tokenizer, BertTokenizer):
            tokens = [self._erase_bert_tokenizer_extras(subword, tokenizer)
                      for subword in tokenizer.batch_decode(tokenizer_ids, skip_special_tokens=True)]
        else:
            tokens = [subword for subword in tokenizer.batch_decode(tokenizer_ids, skip_special_tokens=True)
                      if subword not in tokenizer.all_special_tokens]

        lengths = [len(token) for token in tokens]
        cum_lengths = [0, lengths[0]]  # TODO: IndexError: list index out of range (cpu experiment)
        for ith_length in lengths[1:]:
            cum_lengths.append(cum_lengths[-1] + ith_length)

        intervals = [(cum_lengths[i], cum_lengths[i + 1]) for i in range(len(cum_lengths) - 1)]
        return tokens, intervals

    @staticmethod
    def _offsets_intersection(x: Tuple[int, int], y: Tuple[int, int]) -> float:
        """
        Returns a size of the offset of the two intervals and their relative position: ()
        """
        min_len = min(x[1] - x[0], y[1] - y[0])

        if min_len == 0:
            # spaces are tokenized to a zero length => for these cases, we want to get the overlaying subword emb.
            if x[0] <= y[0] <= y[1] <= x[1]:
                # x overlays y -> returning nonzero allows to identify current position
                return 0.5
            elif y[0] <= x[0] <= x[1] <= y[1]:
                # y overlays x
                return 0.5
            # no overlay
            return 0

        if x[0] <= y[0]:
            # the first one starts earlier
            if x[1] <= y[0]:
                # the first one ends before the second starts
                return 0.
            else:
                # the first one ends after the second starts
                return min(min_len, x[1] - y[0])
        else:
            # the second one starts earlier, otherwise analogical
            if y[1] <= x[0]:
                return 0.
            else:
                return min(min_len, y[1] - x[0])

    def _get_own_to_embedder_alignment(self, own_ids: torch.Tensor, embedder_ids: torch.Tensor,
                                       start_pos: int, end_pos: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Gets the coordinates mapping own_ids to the coordinates of embedder_ids.
        """
        own_tokens, own_intervals = self._intervals_for_text_tokenizer(own_ids, self.tokenizer)
        emb_tokens, emb_intervals = self._intervals_for_text_tokenizer(embedder_ids, self.scorer.scorer._tokenizer)

        own_pointer = 0
        emb_pointer = start_pos

        end_pos = end_pos if end_pos is not None else len(own_tokens)

        own_indices = []
        embedder_indices = []

        # alignment (position, quality = intersection size)
        best_alignment = -1
        best_intersection = -1
        # iterate forwards, gathering the positions on the way
        while own_pointer < end_pos:
            own_interval = own_intervals[own_pointer]
            try:
                emb_interval = emb_intervals[emb_pointer]
            except IndexError:
                # exhausted embeddings intervals -> this is ok, we'll just skip the remaining own_ids
                break

            intersection = self._offsets_intersection(own_interval, emb_interval)
            if intersection > best_intersection:
                best_intersection = intersection
                best_alignment = emb_pointer

            # if emb interval ends after own, return the best already-found alignment of current own position
            if emb_interval[1] >= own_interval[1]:
                own_token_chars = set(own_tokens[own_pointer])
                embedded_token_chars = set(emb_tokens[best_alignment])
                if own_token_chars and not own_token_chars.intersection(embedded_token_chars):
                    # print("WARNING: Alignment of non-empty input id to embedding with no intersection with the embeded "
                    #       "token. Own tokens: %s, embedded tokens: %s"
                    #       % (own_tokens[own_pointer], emb_tokens[best_alignment]))
                    pass
                else:
                    own_indices.append(own_pointer)
                    embedder_indices.append(best_alignment)

                best_intersection = -1
                best_alignment = -1

                own_pointer += 1
            else:
                emb_pointer += 1

        return own_indices, embedder_indices

    @staticmethod
    def _embeddings_on_coordinates(embeddings: torch.Tensor, coordinates: List[int]) -> torch.Tensor:
        """
        Retrieves embeddings located on the corresponding coordinates.
        This might be more complex when processing in batches
        """
        # TODO: impact of the gathering device?
        return embeddings[coordinates]

    def _distances_for_hyp_ids(self,
                               ref_embeddings: torch.Tensor,
                               hyp_own_ids: torch.Tensor,
                               hyp_embedder_ids: torch.Tensor,
                               hyp_embeddings: torch.Tensor,
                               start_own_pos: Optional[int] = 0,
                               end_own_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes minimal hyp_embeddings distances of the hyp_embedder_ids aligned with hyp_own_ids to ref_embeddings
        Returns a tensor of distances of shape (end_own_pos - start_own_pos)
        """
        own_ids_pos, aligned_hyp_ids = list(self._get_own_to_embedder_alignment(hyp_own_ids, hyp_embedder_ids,
                                                                                start_own_pos, end_own_pos))
        # alignments check:
        # list(zip(self.tokenizer.batch_decode(hyp_own_ids[own_ids_pos]),
        #          self.scorer.scorer._tokenizer.batch_decode(hyp_embedder_ids[aligned_hyp_ids])))
        hyp_ids_embeddings = self._embeddings_on_coordinates(hyp_embeddings, aligned_hyp_ids)

        # here, also alignments to the reference can be accessed (.min(-1).indices)
        hyp_ids_distances = torch.cdist(hyp_ids_embeddings, ref_embeddings).softmax(-1).min(-1)

        # alignment to the reference check:
        return own_ids_pos, hyp_ids_distances.indices, hyp_ids_distances.values

    def _get_decodeable_hyps_mask(self, embedder_inputs: torch.LongTensor) -> torch.Tensor:
        encoding_contains_unks = (embedder_inputs == self.scorer.scorer._tokenizer.unk_token_id).any(axis=1)
        return ~encoding_contains_unks

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      num_samples: int = 20,
                      ignored_label: int = -100) -> torch.FloatTensor:
        print("Samples queue size: %s" % len(self.samples_queue))

        batch = {k: v.to(self.device) for k, v in self._get_next_sample().items()}
        input_batch = {k: v for k, v in batch.items() if k not in ("oid", "labels", "decoder_input_ids")}

        losses = []
        # batch_distances = []
        # batch_scores = []

        for ref_ids, (hyps_own_ids, hyps_token_scores, hyp_scores) in zip(batch["labels"],
                                                                          self.do_sample(input_batch, num_samples)):
            hyps_text = self.tokenizer.batch_decode(hyps_own_ids, skip_special_tokens=True)
            ref_text = self.tokenizer.decode([l if l > 0 else 0 for l in ref_ids], skip_special_tokens=True)

            # remove hypotheses that can not be encoded with the embedder

            # all_hyps_score = torch.tensor([self.scorer.evaluate_str([ref_text], [hyp_str]) for hyp_str in hyps_text])
            with torch.no_grad():
                ref_embedder_inputs = self.scorer.scorer._tokenizer(ref_text,
                                                                    return_tensors="pt",
                                                                    truncation="longest_first",
                                                                    padding=True).to(self.device)

                ref_embeddings = self.scorer.scorer._model(**ref_embedder_inputs)[0][0]

                hyps_embedder_inputs = self.scorer.scorer._tokenizer(hyps_text,
                                                                     return_tensors="pt",
                                                                     truncation="longest_first",
                                                                     padding=True).to(self.device)
                # TODO: try without the mask, if it fails, apply the mask
                # decodeable_hyps_mask = self._get_decodeable_hyps_mask(hyps_embedder_inputs).to(self.device)

                hyps_embeddings = self.scorer.scorer._model(**hyps_embedder_inputs)[0]

            ref_embeddings.requires_grad_(True)
            hyps_embeddings.requires_grad_(True)

            # TODO: this could be vectorised by matching one-hot per-character representations of wordpieces
            for hyp_own_ids, hyp_token_scores, hyp_embeder_ids, hyp_embeddings in zip(hyps_own_ids,
                                                                                      hyps_token_scores,
                                                                                      hyps_embedder_inputs.input_ids,
                                                                                      hyps_embeddings):
                own_indices, emb_indices, distances = self._distances_for_hyp_ids(ref_embeddings,
                                                                                  hyp_own_ids,
                                                                                  hyp_embeder_ids,
                                                                                  hyp_embeddings)
                # TODO once functional, add other covariates - weighting by confidence / by overall hyp_score
                # Multiply by the corresponding per-token probabilities, to construct the DCG to trained model
                # losses.append(distances * hyp_token_scores[own_indices])

                self.distances_memory.append(distances)
                scores = hyp_token_scores[own_indices]
                self.scores_memory.append(scores)

                distances_padded = torch.hstack([distances, self.distances_pads[distances.shape[0]]])

                scores_padded = torch.hstack([scores, self.scores_pads[scores.shape[0]]])

                # batch_distances.append(distances_padded)
                # batch_scores.append(scores_padded)

                losses.append(torch.nn.L1Loss()(distances_padded, 1 - scores_padded))

        # if not losses:
        #     print("Warning: empty set of valid hypotheses to compute loss on.")
        #     # return torch.tensor(0., dtype=torch.float, requires_grad=True)
        #     return torch.FloatTensor(0., dtype=torch.float, requires_grad=True)
        # else:
        #     return torch.hstack(losses).mean()

        return torch.hstack(losses).mean()

    def _truncate(self):
        for dist in self.distances_memory:
            del dist
        for score in self.scores_memory:
            del score


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
