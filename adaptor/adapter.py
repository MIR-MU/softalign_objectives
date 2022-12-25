import gc
import logging
import os
from typing import List, Dict, Tuple, Union, Optional

import torch
from transformers import Trainer, BatchEncoding
from transformers import WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model

from .lang_module import LangModule
from .schedules import Schedule
from .utils import AdaptationArguments

logger = logging.getLogger()


class Adapter(Trainer):
    """
    Adapter instance is a lightweigt wrapper of HuggingFace Trainer.
    1. It performs mapping of IterableDatasets constructed in Schedule, to Trainer(*dataset)
    2. For user convenience, it re-evaluates arguments sanity for (multi-)objective adaptation.
    3. It propagates computation of loss to schedule, which distributes them to corresponding Objectives.
    4. It extends training logs (created in events `on_log` and `on_evaluate`) with objective-specific logs.
    5. It extends model persistence on checkpoints and after the training to a separate model for each Objective.
    """

    permitted_args = ["args", "tokenizer", "callbacks", "optimizers"]
    eval_metrics_prefix = "eval"

    def __init__(self, lang_module: LangModule, schedule: Schedule, args: AdaptationArguments, **kwargs):
        """
        Initialises Adapter, used in the same way as HuggingFace Trainer, refer to its documentation for more features.
        :param lang_module: Wrapper of multi-head model with registered heads for each objective of `schedule`.
        :param schedule: Adaptor's Schedule. Determines ordering of applying training Objectives and other.
        :param args: Positional arguments to be passed to HF Trainer.
        :param kwargs: Keyword arguments to be checked and passed to HF Trainer.
        """
        unexpected_args = [k for k in kwargs.keys() if k not in self.permitted_args]
        if unexpected_args:
            raise ValueError("Adapter(**kwargs) got these unexpected kwargs: %s" % unexpected_args)

        self.schedule = schedule

        orig_callbacks = [] if "callbacks" not in kwargs else kwargs.pop("callbacks")

        super().__init__(model=lang_module,
                         args=args,
                         train_dataset=self.schedule.iterable_dataset(split="train"),
                         eval_dataset=self.schedule.iterable_dataset(split="eval"),
                         data_collator=self.flattened_collator,
                         compute_metrics=None,  # would require a static prediction format among objectives
                         callbacks=orig_callbacks + [schedule.should_stop_check_callback()],
                         **kwargs)

    @staticmethod
    def flattened_collator(features: List[BatchEncoding]) -> BatchEncoding:
        """
        Objectives take care of their own data collation, so this collator just flattens the outputs of batch_size=1.

        :return: loss and a placeholder of unused outputs, for compatibility
        """
        assert len(features) == 1, "Sorry, for multi-GPU training, we only support DistributedDataParallel for now."

        return features[0]

    # noinspection PyTypeChecker
    @staticmethod
    def _do_smooth_labels(outputs: torch.FloatTensor, labels: torch.LongTensor, alpha: float) -> torch.FloatTensor:
        """
        Apply label smoothing on the given labels, assuming the shape of outputs.

        :params outputs: trained model outputs
        :param labels: given training targets to be smoothed, in a one-hot form
        :alpha propotion of target to be redistributed among false targets.

        :return smoothed version of given one-hot labels
        """
        smoothed_labels = torch.full_like(outputs, alpha / outputs.shape[-1])  # fill in negatives

        nonignored_labels_idx = labels.clone()
        nonignored_labels_idx[nonignored_labels_idx < 0] = 0
        ignored_labels_idx = labels < 0

        smoothed_labels = smoothed_labels.scatter(dim=-1, index=nonignored_labels_idx.unsqueeze(-1), value=1 - alpha)

        ignored_labels = labels[ignored_labels_idx].unsqueeze(-1).expand(-1, smoothed_labels.shape[-1])
        smoothed_labels[ignored_labels_idx] = ignored_labels.float()

        # randomized check
        assert nonignored_labels_idx[0][2] == smoothed_labels.argmax(-1)[0][2]

        return smoothed_labels

    def compute_loss(self,
                     model: LangModule,
                     inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, None]]:
        labels = inputs["labels"] if "labels" in inputs else inputs["label"]

        outputs = model(**inputs)
        if self.args.label_smoothing_factor > 0:
            labels = self._do_smooth_labels(outputs, labels, self.args.label_smoothing_factor)
            # raise NotImplementedError()  # objective-dependent label smoothing is custom
        loss = self.schedule.compute_loss(inputs, outputs, labels)

        mock_outputs = torch.tensor([-1, -1])  # compatibility with Trainer.prediction_step return types
        return (loss, mock_outputs) if return_outputs else loss

    def log(self, logs: List[Dict[str, float]]) -> None:
        gc.collect(generation=0)
        self._objects_log()

        is_eval_log = any(self.eval_metrics_prefix in log_key for log_key in logs)
        extended_logs = self.schedule.objectives_log(split="eval" if is_eval_log else "train")
        return super().log({**logs, **extended_logs})

    @staticmethod
    def _del_tensors_of_xshape(xshape: int = 1):
        import gc
        deleted = 0
        # manual garbage collection of the previous, unallocated inputs
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and len(obj.size()) == 2 and obj.size()[0] == xshape and obj.size()[1] <= 512:
                # print("Number of references: %s" % len(gc.get_referrers(obj)))
                del obj
                deleted += 1
        logger.warning("GPU: Deleted %s objects" % deleted)

    @staticmethod
    def _count_objects() -> Dict[Tuple[int], int]:
        import gc
        objects_counter = dict()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    k = tuple(obj.size())

                    if k in objects_counter:
                        objects_counter[k] += 1
                    else:
                        objects_counter[k] = 1
            except:
                pass
        return objects_counter

    @staticmethod
    def _count_objects_diff(smaller_count, bigger_count) -> Dict[str, int]:
        counts = {k: v - smaller_count[k] if k in smaller_count else v for k, v in bigger_count.items()}
        return {k: v for k, v in counts.items() if v != 0}

    @staticmethod
    def _objects_log() -> None:
        objects_counter = Adapter._count_objects()

        logger.warning("Number of torch tensors: \n%s" % sum(objects_counter.values()))

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        logger.warning("Evaluating...")

        out = super(Adapter, self).evaluate(*args, **kwargs)
        if "metric_key_prefix" in kwargs:
            self.eval_metrics_prefix = kwargs["metric_key_prefix"]

        # refresh exhausted evaluation iteration for possible next evaluation
        self.eval_dataset = self.schedule.iterable_dataset("eval")

        logger.warning("Allocated memory: %s" % torch.cuda.memory_allocated(0))

        return out

    def save_model(self, output_dir: Optional[str] = None) -> None:
        # HF native reload compatibility
        objectives_counter = {str(obj): 0 for obj in self.schedule.objectives["train"].values()}

        for objective_id in self.schedule.objectives["train"].keys():
            module = self.model.trainable_models[str(objective_id)]
            objective = self.schedule.objectives["train"][int(objective_id)]
            output_module_path = os.path.join(output_dir, str(objective))

            # if the objective of this id was already persisted, we'll index the configs of the next ones
            if objectives_counter[str(objective)] != 0:
                output_module_path += "_{}".format(objectives_counter[str(objective)])
                objectives_counter[str(objective)] += 1

            # we persist a shared tokenizer and training args either way
            self.model.tokenizer.save_pretrained(output_module_path)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

            if hasattr(module, "save_pretrained") or hasattr(unwrap_model(module), "save_pretrained"):
                # if the head module has "save_pretrained" method, it will be called for persistence
                module.save_pretrained(output_module_path, use_diff=True)
            else:
                # otherwise, we persist only a raw pytorch module
                torch.save(module.state_dict(), os.path.join(output_module_path, WEIGHTS_NAME))

            logger.info(f"Model of objective {str(objective)} saved in {output_module_path}")
