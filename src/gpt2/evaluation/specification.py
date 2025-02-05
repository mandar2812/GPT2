from typing import Dict

import torch
import torch.nn as nn

from src.gpt2.data import Dataset


class EvaluationSpec(object):
    def initialize(self):
        pass

    def prepare_dataset(self) -> Dataset:
        raise NotImplementedError()

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def eval_objective(
        self, data: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
