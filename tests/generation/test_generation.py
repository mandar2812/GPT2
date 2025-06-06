import torch
import numpy as np
from src.gpt2.modeling import Transformer
from src.gpt2.generation import GenerationSpec, GenerateConfig, Generator


def test_generator_sample_from_top_p():
    generator = Generator(
        spec=None,
        config=GenerateConfig(
            context_len=0, nucleus_prob=0.5, temperature=1.0, use_gpu=False
        ),
    )

    probs = torch.tensor([0.1, 0.2, 0.15, 0.08, 0.25, 0.22])
    for _ in range(100):
        assert generator._sample_from_top_p(probs) in [1, 4, 5]


def test_generator_predict_probs_output_shape():
    spec = GenerationSpec()
    spec.construct_model = lambda: Transformer(
        layers=2,
        pad_idx=0,
        words=80,
        heads=2,
        dims=16,
        rate=4,
        dropout=0,
        bidirectional=False,
    )

    # Create generator with simple specification.
    generator = Generator(
        spec=spec,
        config=GenerateConfig(
            context_len=0, nucleus_prob=0.5, temperature=1.0, use_gpu=False
        ),
    )
    generator.initialize()

    probs1, past1 = generator._predict_probs([np.random.randint(80) for _ in range(8)])
    probs2, past2 = generator._predict_probs(
        [generator._sample_from_top_p(probs1)], past1
    )

    assert probs1.shape == (80,)
    assert probs2.shape == (80,)

    for p in past1:
        assert p[0].shape == (8, 16)
        assert p[1].shape == (8, 16)
    for p in past2:
        assert p[0].shape == (9, 16)
        assert p[1].shape == (9, 16)


def test_generator_generate():
    class TestGenerationSpec(GenerationSpec):
        def construct_model(self):
            return Transformer(
                layers=2,
                pad_idx=0,
                words=80,
                heads=2,
                dims=16,
                rate=4,
                dropout=0,
                bidirectional=False,
            )

        def encode_context(self, context):
            return list(range(len(context.split())))

        def decode_tokens(self, tokens):
            return " ".join(str(t) for t in tokens)

        @property
        def stop_token(self) -> int:
            return 79

    spec = TestGenerationSpec()

    # Create generator with simple specification.
    generator = Generator(
        spec=spec,
        config=GenerateConfig(
            context_len=50, nucleus_prob=0.5, temperature=1.0, use_gpu=False
        ),
    )
    generator.initialize()

    for t in generator.generate("a b c d e").split():
        assert int(t) < 80
