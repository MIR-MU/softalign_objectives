"""
Implementation of the TokenAlign objective, using the decontextualization implemented in AlignmentBase
"""

import logging
from typing import Union, Dict, Optional

import torch

from transformers import BatchEncoding

from adaptor.new_objectives.seq_bertscr_objectives import AlignmentBase


logger = logging.getLogger()


class TokenAlignObjective(AlignmentBase):
    """
    Token alignment objective (TokenAlign) grounding the candidate target tokens in their alignment to the reference,
    based on the quality of the best alignment of each next token to the full reference.
    """

    def __init__(self, *args, emb_infer_batch_size: int = 32, emb_size: int = 768, **kwargs):
        super().__init__(*args, **kwargs)

        source_texts, ref_texts = self._per_split_iterators("train")

        # inference of decontextualized embeddings
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
                      labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        # 1. Construct a matrix: sample X position X top-k logit index (remember the index separately)

        # mask to select only over the known embeddings, i.e. the ones that we have the embeddings for
        indexed_tokens = torch.where(self.spiece_counts > 0)[0]
        # collect predicted scores (with grad_fn) of the indexed tokens and their embeddings
        indexed_tokens_logits = lm_logit_outputs[..., indexed_tokens]
        indexed_tokens_embs = self.spiece_embeddings[indexed_tokens]
        with torch.no_grad():
            ref_emb_inputs, ref_embs = self._embeddings_for_text(self.tokenizer.batch_decode(labels))
        ref_embs.requires_grad_(True)

        # 2. Compute distances
        min_dists_to_reference, min_dist_positions = torch.cdist(indexed_tokens_embs, ref_embs).min(-1)

        # normalize
        min_dists_to_reference_normed = min_dists_to_reference / min_dists_to_reference.max(-1).values

        # 3. construct targets as the distances of all possible target tokens
        loss = torch.nn.CrossEntropyLoss()
        loss_val = loss(indexed_tokens_logits, (1 - min_dists_to_reference_normed.expand_as(indexed_tokens_logits)))

        return loss_val
