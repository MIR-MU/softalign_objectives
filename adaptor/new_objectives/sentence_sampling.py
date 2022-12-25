from typing import Optional, Union, Dict

import torch
from transformers import BatchEncoding

from .seq_bertscr_objectives import AlignmentBase
from ..evaluators.generative import GenerativeEvaluator, BLEU


class ScheduledSampling(AlignmentBase):

    def __init__(self, *args, num_samples: int = 10, evaluator: GenerativeEvaluator = BLEU(), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.evaluator = evaluator

    def _compute_loss(self,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      lm_logit_outputs: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      ignored_label: int = -100) -> torch.Tensor:
        input_batch = {k: v for k, v in inputs.items() if k not in ("oid", "labels", "decoder_input_ids")}

        loss_inst = torch.nn.CrossEntropyLoss()
        loss_vals = []

        # sample hypotheses
        translations_sampler = self.do_sample(input_batch, self.num_samples, collect_logits=True)

        for hyps_ids, tokens_scores, hyps_scores, hyps_logits, hyps_labels in zip(translations_sampler, labels):
            # collect the evaluations for comparison
            best_eval_score = 1 if self.evaluator.smaller_is_better else 0
            best_hyp_ids = torch.empty()
            best_hyp_logits = torch.empty()

            for hyp_ids, hyp_logits, hyp_labels in zip(hyps_ids, hyps_logits, hyps_labels):

                hyp_evaluation = self.evaluator.evaluate_str([self.tokenizer.decode(hyp_ids)],
                                                             [self.tokenizer.decode(hyp_labels)])

                # remember the best (oracle) sentence and use it as a target
                if hyp_evaluation > best_eval_score:
                    best_hyp_ids = hyps_ids
                    best_hyp_logits = hyps_logits  # TODO: check that best_hyp_logits remain same between iterations

            best_sample_loss = loss_inst(best_hyp_logits, best_hyp_ids)
            loss_vals.append(best_sample_loss)

        return torch.vstack(loss_vals).mean()
