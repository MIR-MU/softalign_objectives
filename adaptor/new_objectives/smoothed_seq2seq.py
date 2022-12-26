from ..objectives.objective_base import SupervisedObjective
from ..objectives.seq2seq import Sequence2SequenceMixin


class SmoothedSequence2Sequence(Sequence2SequenceMixin, SupervisedObjective):

    # smoothed loss is already implemented in seq2seq, this class is just for distinguishing objectives labeling
    pass
