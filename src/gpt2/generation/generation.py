from typing import List, Optional, Tuple

import torch

from src.gpt2.training import Recorder
from src.gpt2.generation import GenerateConfig, GenerationSpec
from src.gpt2.modeling import Past


class Generator(object):
    def __init__(self, spec: GenerationSpec, config: GenerateConfig):
        self.spec = spec
        self.config = config

    def initialize(self, from_model: Optional[str] = None):
        # Initialize generation environment and construct a model.
        self.spec.initialize()
        self.model = self.spec.construct_model().eval()

        # Load trained model parameters.
        if from_model:
            torch.serialization.add_safe_globals([Recorder])
            ckpt = torch.load(from_model, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])

        # Move the model to GPU device and convert the data type to half
        # precision.
        if self.config.use_gpu:
            self.model.cuda().half()

    def generate(self, context: str, chat: bool = False) -> str:
        input_tokens = self.spec.encode_context(context)
        output_tokens: list[int] = []
        current, past = input_tokens, None
        while len(input_tokens + output_tokens) < self.config.context_len:
            # Predict the next word token from the given context.
            probs, past = self._predict_probs(current, past)
            next_word = self._sample_from_top_p(probs)

            # Change the context to the predicted word.
            output_tokens.append(next_word)
            if next_word == self.spec.stop_token:
                break
            current = [next_word]

        result = output_tokens if chat else input_tokens + output_tokens
        return self.spec.decode_tokens(result)

    @torch.no_grad()
    def _predict_probs(
        self, words: List[int], past: Optional[List[Past]] = None
    ) -> Tuple[torch.Tensor, List[Past]]:
        x = torch.tensor(words, dtype=torch.long)
        x = self.spec.decorate_sequence(
            x, offset=past[0][0].size(-2) if past is not None else 0
        )

        if self.config.use_gpu:
            logits, past = self.model(x.cuda(), past)
            logits = logits.cpu().float()
        else:
            logits, past = self.model(x, past)

        return (logits[-1, :] / self.config.temperature).softmax(-1), past

    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        probs, indices = probs.sort(descending=True)

        mask = probs.cumsum(-1) > self.config.nucleus_prob
        mask[0] = False
        probs.masked_fill_(mask, 0)
        probs /= probs.sum()

        # Sample from filtered distribution.
        return indices[probs.multinomial(1)[0]].item()
