import torch
from ray.util.sgd.torch import TrainingOperator


class Operator(TrainingOperator):
    def __init__(self, config=None):
        raise NotImplementedError
        super().__init__(config)
