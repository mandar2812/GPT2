from tqdm import tqdm
import torch
import torch.nn as nn
from src.gpt2.data import Dataset
from src.gpt2.evaluation import EvaluationSpec, EvaluateConfig
from typing import Optional, Dict


class Evaluator(object):
    def __init__(self, spec: EvaluationSpec, config: EvaluateConfig):
        self.spec = spec
        self.config = config

    def evaluate(self, from_model: Optional[str] = None) -> Dict[str, float]:
        # Initialize evaluation environment and prepare a dataset.
        self.spec.initialize()
        eval_dataset = self.spec.prepare_dataset()

        # Load trained model parameters.
        model = self.spec.construct_model().eval()
        if from_model:
            ckpt = torch.load(from_model, map_location="cpu")
            model.load_state_dict(ckpt["model"])

        # Move the model to GPU device and convert the data type to half
        # precision.
        if self.config.use_gpu:
            model.cuda().half()

        total_metrics = {}
        with tqdm(
            desc="Evaluating GPT2", total=eval_dataset.size(self.config.batch_eval)
        ) as pbar:
            eval_step = 0
            for _ in self.config.iterate():
                batch_metrics = self._eval_step(eval_dataset, model)
                pbar.update(1)
                if batch_metrics is None:
                    break

                # Record the batched metrics.
                for k, v in batch_metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = []
                    total_metrics[k].append(v)
                if (eval_step + 1) % 20 == 0:
                    pbar.set_postfix(
                        {k: sum(v) / len(v) for k, v in total_metrics.items()}
                    )
                eval_step += 1

        return {k: sum(v) / len(v) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_step(
        self, dataset: Dataset, model: nn.Module
    ) -> Optional[Dict[str, float]]:
        try:
            data = dataset.fetch(self.config.batch_eval)
            if self.config.use_gpu:
                data = {k: v.cuda() for k, v in data.items()}

            metrics = self.spec.eval_objective(data, model)
            return {k: v.item() for k, v in metrics.items()}
        except StopIteration:
            return None
