"""
Implementations of the Sequence Alignment objectives, including the ablations.
"""

import itertools
import logging
from typing import Tuple, Union, Dict, Optional, List, Iterable

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, BertTokenizer

from adaptor.evaluators.generative import BERTScore
from adaptor.objectives.seq2seq import Sequence2Sequence

logger = logging.getLogger()


class AlignmentBase(Sequence2Sequence):
    """
    Base functionality of the Alignment-based objectives.
    Provides the following functionality:
    * hypotheses sampling including their differentiable per-token scores,
    * alignment of the wordpieces of two different tokenizers
    * implementation of decontextualization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # in all our experiments, we use BERTScore as the contextualized embeddings provider
        self.scorer = BERTScore()

    def sample_n(self,
                 inputs: Union[Dict[str, torch.LongTensor], BatchEncoding],
                 num_samples: int,
                 top_k_sampling: Optional[int] = 3,
                 greedy: Optional[bool] = False,
                 collect_logits: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor]:
        """
        Samples n hypotheses from the model, according to the model probability distribution.
        If top-k is given, sample only from the k-best-rated next-tokens.
        If collect_logits, return also logits for these tokens (with grad_fn).
        If greedy, return only the most-likely next-token(s).
        """
        device = self.compatible_head_model.device
        seq = torch.empty(num_samples, 0, dtype=torch.int64).to(device)
        seq_probs = torch.empty(num_samples, 0, dtype=torch.float32).to(device)
        logits_history = []

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
            try:
                outputs = self.compatible_head_model(attention_mask=inputs_parallel["attention_mask"],
                                                     input_ids=None,
                                                     past_key_values=past,
                                                     decoder_input_ids=decoder_input,
                                                     encoder_outputs=encoder_outputs)
                if collect_logits:
                    logits_history.append(outputs.logits)

            except RuntimeError as e:
                print("Runtime error on decoder_input: %s" % decoder_input)
                raise e
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
                    print("Runtime error on input: \n%s" % inputs)
                    raise e
                if top_k_sampling is not None:
                    next_token_id = top_k_args[0, 0, sampled_next_token_rank]
                    next_token_prob = tokens_probs[0, 0, sampled_next_token_rank]
                else:
                    next_token_id = sampled_next_token_rank
                    next_token_prob = tokens_probs[0, 0, next_token_id]

            assert ((0 <= next_token_id).all() and
            (next_token_id < self.compatible_head_model.base_model.encoder.embed_tokens.num_embeddings).all()).item(), \
                "Assertion failed: next_token_id: %s" % next_token_id

            seq = torch.hstack([seq, next_token_id])
            seq_probs = torch.hstack([seq_probs, next_token_prob])

            past = outputs.past_key_values

            # termination check -> when all the sequences generated <eos>, we prune to the first <eos>,
            # aggregate scores for policy gradients and return sequences with their scores
            seq_terminated = (seq == self.tokenizer.eos_token_id).any(dim=1)
            generated_length = len(seq[0])
            max_length_reached = generated_length >= self.compatible_head_model.config.max_length - 1
            if seq_terminated.all() or generated_length > 3 * len(inputs["input_ids"][0]) or max_length_reached:
                # prune sequences to the earliest-eos-token
                earliest_eoses = (seq == self.tokenizer.eos_token_id).long().argmax(-1)
                agg_probs = []
                for seq_idx, eos_idx in enumerate(earliest_eoses):
                    seq[seq_idx, eos_idx + 1:] = self.tokenizer.pad_token_id
                    # eos_idx+1 => including eos token prob to the aggregation
                    pruned_seq_prob = seq_probs[seq_idx, :eos_idx + 1].log().sum()
                    agg_probs.append(pruned_seq_prob)

                if collect_logits:
                    logits_history = torch.hstack(logits_history)

                return seq, seq_probs, torch.hstack(agg_probs), logits_history

            decoder_input = next_token_id

    def do_sample(self,
                  inputs: Union[Iterable[Dict[str, torch.Tensor]], BatchEncoding],
                  num_samples: int,
                  collect_logits: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor]:
        """
        Samples a given set of translations using Monte Carlo.
        :return an iterator over tuples of [selected sequence, sequence score]
        """
        batch_attributes = inputs.keys()

        for input_tuple in zip(*(inputs[attr] for attr in batch_attributes)):
            input_tuple_batch = (i.unsqueeze(0) for i in input_tuple)
            yield self.sample_n(dict(zip(batch_attributes, input_tuple_batch)),
                                num_samples,
                                collect_logits=collect_logits)

    def _erase_bert_tokenizer_extras(self,
                                     text: str,
                                     tokenizer: PreTrainedTokenizer,
                                     symbols: List[str] = (" ", "#")) -> str:
        """
        Removes the decoding specialities of BERT Tokenizer.
        """
        out_text = text
        for symbol in itertools.chain(symbols, tokenizer.all_special_tokens):
            out_text = out_text.replace(symbol, "")
        return out_text

    def _intervals_for_text_tokenizer(self,
                                      tokenizer_ids: torch.Tensor,
                                      tokenizer: PreTrainedTokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Computes intervals of token_ids of the given tokenizer. The intervals are used to find the optimal alignment
        of the result of two distinct tokenizers, in our case BART's sentencepiece and BERT's wordpiece.
        """
        if isinstance(tokenizer, BertTokenizer):
            tokens = [self._erase_bert_tokenizer_extras(subword, tokenizer)
                      for subword in tokenizer.batch_decode(tokenizer_ids, skip_special_tokens=True)]
        else:
            tokens = [subword if subword not in tokenizer.all_special_tokens else ""
                      for subword in tokenizer.batch_decode(tokenizer_ids, skip_special_tokens=True)]

        lengths = [len(token) for token in tokens]
        cum_lengths = [0, lengths[0]]
        for ith_length in lengths[1:]:
            cum_lengths.append(cum_lengths[-1] + ith_length)

        intervals = [(cum_lengths[i], cum_lengths[i + 1]) for i in range(len(cum_lengths) - 1)]
        return tokens, intervals

    @staticmethod
    def _offsets_intersection(x: Tuple[int, int], y: Tuple[int, int]) -> float:
        """
        Returns a size of the offset of the two intervals and their relative position.
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
        If start_pos or end_pos is given, this will only perform the mapping on such-selected interval.
        """
        own_tokens, own_intervals = self._intervals_for_text_tokenizer(own_ids, self.tokenizer)
        emb_tokens, emb_intervals = self._intervals_for_text_tokenizer(embedder_ids, self.scorer.scorer._tokenizer)

        own_pointer = start_pos
        emb_pointer = 0

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
                    print("WARNING: Alignment of non-empty input id to embedding with no intersection with the embeded "
                          "token. Own tokens: %s, embedded tokens: %s"
                          % (own_tokens[own_pointer], emb_tokens[best_alignment]))
                    # we skip the tokens that we can not align, but this show happen for a vast minority of alignments
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
        emb_distances = torch.cdist(hyp_ids_embeddings, ref_embeddings)
        # norm by the size of the reference -> this should make the distances of different references comparable
        hyp_ids_distances_normed = emb_distances / emb_distances.max()
        hyp_ids_distances_min = hyp_ids_distances_normed.min(-1)

        # alignment to the reference check:
        return own_ids_pos, hyp_ids_distances_min.indices, hyp_ids_distances_min.values

    def _embeddings_for_text(self, texts: List[str]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Computes BERTScore embeddings for a list of texts.
        """
        embedder_inputs = self.scorer.scorer._tokenizer(texts,
                                                        return_tensors="pt",
                                                        truncation="longest_first",
                                                        padding=True).to(self.device)

        embeddings = self.scorer.scorer._model(**embedder_inputs)[0]
        return embedder_inputs, embeddings

    def decon_spiece_embeddings_from_texts(self,
                                           texts: List[str],
                                           infer_batch_size: int = 32,
                                           emb_size: int = 768) -> Tuple[List[int], torch.Tensor]:
        """
        Collects decontextualized embeddings for a given list of texts.
        To allow optimisation, the embeddings should have predefined `emb_size` dimensionality.
        `infer_batch_size` can be scaled according to the available GPU infrastructure.
        """

        spiece_embeddings = torch.zeros(len(self.tokenizer.decoder.keys()), emb_size, device=self.device)
        spiece_counts = [0 for spiece in self.tokenizer.decoder.keys()]

        from tqdm import tqdm
        for batch_offset in tqdm(range(0, len(texts), infer_batch_size),
                                 desc="%s: Inferring embeddings for decontextualization" % self,
                                 total=len(texts) // infer_batch_size):
            batch_texts = texts[batch_offset: batch_offset + infer_batch_size]
            with torch.no_grad():
                ref_emb_inputs, ref_embs = self._embeddings_for_text(batch_texts)

            ref_own_inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation="longest_first", padding=True)

            for i, (item_emb_ids, item_own_ids) in enumerate(zip(ref_emb_inputs.input_ids, ref_own_inputs.input_ids)):
                own_indices, embedder_indices = self._get_own_to_embedder_alignment(item_own_ids, item_emb_ids, 0)

                spiece_batch_ids = ref_own_inputs.input_ids[i, own_indices]  # type: ignore
                spiece_batch_embeddings = ref_embs[i, embedder_indices]

                # 3. averaging
                for spiece, spiece_emb in zip(spiece_batch_ids, spiece_batch_embeddings):
                    if not spiece_counts[spiece.item()]:
                        spiece_counts[spiece.item()] += 1
                        spiece_embeddings[spiece.item()] = spiece_emb
                    else:
                        spiece_counts[spiece.item()] += 1
                        # weighted average: we need the counter already updated here
                        new_emb_weight = 1 / spiece_counts[spiece.item()]
                        new_emb_weighted = spiece_emb * new_emb_weight

                        orig_emb_weight = 1 - new_emb_weight
                        orig_emb_weighted = spiece_embeddings[spiece.item()] * orig_emb_weight

                        # sanity check: assert the result of the first aggregation here:
                        # torch.vstack([spiece_emb, spiece_embeddings[spiece.item()]]).mean(0) ==
                        # orig_emb_weighted + new_emb_weighted
                        spiece_embeddings[spiece.item()] = orig_emb_weighted + new_emb_weighted

        return spiece_counts, spiece_embeddings


class SeqAlignObjective(AlignmentBase):
    """
    SeqAlign objective samples the top-k tokens of selected `num_samples` hypotheses and updates the model
    conditioned by the prefixes of self-generated hypotheses.
    The model is updated according to L1Loss, measuring the distance of model scores to the alignment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # memory allocation to optimize performance:
        self.distances_pads = {i: torch.zeros(self.tokenizer.model_max_length - i,
                                              requires_grad=True, device=self.device)
                               for i in range(self.tokenizer.model_max_length)}
        self.scores_pads = {i: torch.ones(self.tokenizer.model_max_length - i,
                                          requires_grad=True, device=self.device)
                            for i in range(self.tokenizer.model_max_length)}

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      num_samples: int = 10,
                      ignored_label: int = -100) -> torch.Tensor:

        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels", "decoder_input_ids")}

        losses = []

        try:
            # sample hypotheses
            translations_sampler = self.do_sample(input_batch, num_samples)

            for ref_ids, (hyps_own_ids, hyps_token_scores, hyp_scores, _) in zip(inputs["labels"], translations_sampler):
                # encode both reference and hypotheses
                hyps_text = self.tokenizer.batch_decode(hyps_own_ids, skip_special_tokens=True)
                ref_text = self.tokenizer.decode([l if l > 0 else 0 for l in ref_ids], skip_special_tokens=True)

                # infer grounding embeddings of both reference and hypotheses
                with torch.no_grad():
                    _, ref_embeddings = self._embeddings_for_text([ref_text])
                    ref_embeddings = ref_embeddings[0]
                    hyps_embedder_inputs, hyps_embeddings = self._embeddings_for_text(hyps_text)

                ref_embeddings.requires_grad_(True)
                hyps_embeddings.requires_grad_(True)

                # per-top-k token iteration
                # this could be further optimized by matching one-hot per-character representations of wordpieces
                for hyp_own_ids, hyp_token_scores, hyp_embeder_ids, hyp_embeddings in zip(hyps_own_ids,
                                                                                          hyps_token_scores,
                                                                                          hyps_embedder_inputs.input_ids,
                                                                                          hyps_embeddings):
                    # compute alignment: indices and quality
                    own_indices, emb_indices, distances = self._distances_for_hyp_ids(ref_embeddings,
                                                                                      hyp_own_ids,
                                                                                      hyp_embeder_ids,
                                                                                      hyp_embeddings)
                    # subsample corresponding tokens
                    scores = hyp_token_scores[own_indices]

                    # pad to the same (=max) length
                    distances_padded = torch.hstack([distances, self.distances_pads[distances.shape[0]]])
                    scores_padded = torch.hstack([scores, self.scores_pads[scores.shape[0]]])

                    # compute loss value
                    losses.append(torch.nn.L1Loss()(distances_padded, 1 - scores_padded))
        except RuntimeError as e:
            # if something fails, we log it and continue, omitting it from the aggregated loss
            logger.error("%s: Skipping input and returning zero loss" % e)

            return torch.tensor(0., requires_grad=True, dtype=torch.float)

        # we aggregate the losses and yield it for backward and update
        return torch.hstack(losses).mean()


# following objectives are the ablation analyses if SeqAlignObjective.


class SeqAlignCEDecObjective(AlignmentBase):
    """
    Sequential objective based on CrossEntropy loss.
    The alignments are computed from decontextualized embeddings.
    Thanks to the explicit embedding assignment to every token of the trained model vocabulary,
    we do not have to perform the matching "on-the-fly", making the objective faster and _compute_loss simpler.
    """

    def __init__(self, *args, emb_infer_batch_size: int = 32, emb_size: int = 768, num_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

        source_texts, ref_texts = self._per_split_iterators("train")
        ref_texts = list(ref_texts)

        # inference of the decontextualized embeddings
        spiece_counts, self.spiece_embeddings = self.decon_spiece_embeddings_from_texts(ref_texts,
                                                                                        emb_infer_batch_size,
                                                                                        emb_size)
        self.spiece_embeddings.requires_grad_(True)
        # counts of each wordpiece is used to find embeddings which are not in the vocab
        self.spiece_counts = torch.tensor(spiece_counts, dtype=torch.int32, device=self.device)
        logger.warning("Indexation done. %s nonzero embeddings, averaged from %s embeddings"
              % (sum(bool(count) for count in spiece_counts), sum(spiece_counts)))

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      ignored_label: int = -100) -> torch.Tensor:
        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels", "decoder_input_ids")}

        # compute reference embeddings - per-token hypotheses embeddings are pre-computed
        with torch.no_grad():
            ref_emb_inputs, ref_embs = self._embeddings_for_text(self.tokenizer.batch_decode(labels))
        ref_embs.requires_grad_(True)

        # mask to select only over the known embeddings, i.e. the ones that we have the embeddings for
        indexed_tokens = torch.where(self.spiece_counts > 0)[0]

        losses = []
        loss_inst = torch.nn.CrossEntropyLoss()

        # sample hypotheses
        translations_sampler = self.do_sample(input_batch, self.num_samples, collect_logits=True)

        for per_sample_ref_embs, (hyps_own_ids, token_scores, hyps_scores, hyps_logits) in zip(ref_embs,
                                                                                               translations_sampler):
            indexed_hyps_logits = hyps_logits[..., indexed_tokens]
            indexed_tokens_embs = self.spiece_embeddings[indexed_tokens]
            min_dists_to_ref, min_dist_positions = torch.cdist(indexed_tokens_embs, per_sample_ref_embs).min(-1)
            min_dists_to_ref_normed = min_dists_to_ref / min_dists_to_ref.max(-1).values

            loss_val = loss_inst(indexed_hyps_logits, (1 - min_dists_to_ref_normed.expand_as(indexed_hyps_logits)))
            losses.append(loss_val)

        return torch.vstack(losses).mean()


class SeqAlignDecObjective(SeqAlignCEDecObjective):
    """
    Decontextualized SeqAlign objective.
    In contrary to the previous SeqAlignCEDecObjective, it merely exchanges CrossEntropyLoss to L1Loss.
    This delimits the impact of the change of loss type on the eventual performance.

    Thanks to the explicit embedding assignment to every token of the trained model vocabulary,
    we do not have to perform the matching "on-the-fly", making the objective faster and _compute_loss simpler.
    """

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      ignored_label: int = -100) -> torch.Tensor:
        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels", "decoder_input_ids")}

        # compute reference embeddings - per-token hypotheses embeddings are pre-computed
        with torch.no_grad():
            ref_emb_inputs, ref_embs = self._embeddings_for_text(self.tokenizer.batch_decode(labels))
        ref_embs.requires_grad_(True)

        # mask to select only over the known embeddings, i.e. the ones that we have the embeddings for
        indexed_tokens = torch.where(self.spiece_counts > 0)[0]

        losses = []
        loss_inst = torch.nn.L1Loss()

        # sample hypotheses
        translations_sampler = self.do_sample(input_batch, self.num_samples, collect_logits=True)

        for per_sample_ref_embs, (hyps_own_ids, token_scores, hyps_scores, hyps_logits) in zip(ref_embs,
                                                                                               translations_sampler):
            indexed_hyps_logits = hyps_logits[..., indexed_tokens]
            indexed_tokens_embs = self.spiece_embeddings[indexed_tokens]
            min_dists_to_ref, min_dist_positions = torch.cdist(indexed_tokens_embs, per_sample_ref_embs).min(-1)
            min_dists_to_ref_normed = min_dists_to_ref / min_dists_to_ref.max(-1).values

            loss_val = loss_inst(indexed_hyps_logits.exp(),
                                 (1 - min_dists_to_ref_normed.expand_as(indexed_hyps_logits)))
            losses.append(loss_val)

        return torch.vstack(losses).mean()


class SeqBertScoreRandom(AlignmentBase):
    """
    Sequential objective assigning random targets to each sampled hypothesis.
    This objective is an ablation to contextualized embeddings grounding, but can also be used to benchmark
    the time spent on sampling hypotheses. See the paper for both qualitative and speed benchmark results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scorer = None

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      num_samples: int = 5,
                      ignored_label: int = -100) -> torch.Tensor:
        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels", "decoder_input_ids")}

        losses = []

        try:
            translations_sampler = self.do_sample(input_batch, num_samples)

            for ref_ids, (hyps_own_ids, hyps_token_scores, hyp_scores, _) in zip(inputs["labels"], translations_sampler):
                for hyp_token_scores in hyps_token_scores:
                    scores = hyp_token_scores

                    losses.append(torch.nn.L1Loss()(torch.rand_like(scores), 1 - scores))
        except RuntimeError as e:
            logger.error("%s. Skipping input and returning zero loss." % e)

            return torch.tensor(0., requires_grad=True, dtype=torch.float)

        return torch.hstack(losses).mean()
