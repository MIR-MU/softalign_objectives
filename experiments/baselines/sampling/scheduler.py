import random
from typing import Iterator, Iterable

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule


class LinearDecay(Schedule):

    label = "linear_decay"

    passed_steps: int = 0

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in a linear decay schedule.
        See Bengio et al 2015: https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf
        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the chosen to objectives
        """
        objectives = list(self.objectives[split].values())
        assert len(objectives) == 2, "For LinearDecaySchedule, you need to pass exactly two objectives"

        while True:
            training_progress = self.passed_steps / self.args.max_steps  # float in <0; 1>
            random_draw = random.random()

            if training_progress < random_draw:
                # first, sample more often objectives[0], later, prioritize objectives[1]
                yield objectives[0]
            else:
                yield objectives[1]

            self.passed_steps += 1


class InverseSigmoidDecay(Schedule):

    label = "linear_decay"

    passed_steps: int = 0

    def _sample_objectives(self, split: str) -> Iterable[Objective]:
        """
        Sample objectives in inverse sigmoid decay schedule.
        See Bengio et al 2015: https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf
        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the chosen to objectives
        """
        objectives = list(self.objectives[split].values())
        assert len(objectives) == 2, "For LinearDecaySchedule, you need to pass exactly two objectives"

        while True:
            raise NotImplementedError()  # TODO
            training_progress = self.passed_steps / self.args.max_steps  # float in <0; 1>
            random_draw = random.random()

            if training_progress < random_draw:
                # first, sample more often objectives[0], later, prioritize objectives[1]
                yield objectives[0]
            else:
                yield objectives[1]

            self.passed_steps += 1


