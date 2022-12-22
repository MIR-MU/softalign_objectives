from typing import Optional, Dict, Union

import torch
from transformers import BatchEncoding

from ..objectives.objective_base import SupervisedObjective
from ..objectives.seq2seq import Sequence2SequenceMixin


class SmoothedSequence2Sequence(Sequence2SequenceMixin, SupervisedObjective):

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      ignore_index: int = -100) -> torch.FloatTensor:
        """
        Seq2seq loss supporting label-smoothed targets.
        :param logit_outputs: Raw outputs of language modeling head model
        :param labels: Token ids of expected outputs.
        :return: Single value of the loss, with grad_fn.
        """
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations at some cases
        loss_fct = torch.nn.CrossEntropyLoss()

        logits_f = lm_logit_outputs.flatten(end_dim=1)
        labels_f = labels.flatten(end_dim=1)

        if lm_logit_outputs.shape == labels.shape:
            # for non-discrete targets, torch loss will not ignore the ignore_index targets,
            # so we exclude them manually
            ignore_idx = (labels_f == ignore_index).all(-1)
            logits_f = logits_f[~ignore_idx]
            labels_f = labels_f[~ignore_idx]

        lm_loss = loss_fct(logits_f, labels_f)

        return lm_loss
