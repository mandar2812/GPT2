import argparse
import torch.nn as nn
import re
from tokenizers import Tokenizer, models, decoders
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from src.gpt2.data import Vocab
from src.gpt2.modeling import Transformer
from src.gpt2.generation import GenerationSpec, GenerateConfig, Generator
from src.gpt2.generate_sentences import GPT2GenerationSpec
from typing import List


class GPT2ChatSpec(GPT2GenerationSpec):
    def __init__(
        self,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        message_boundaries: tuple[str, str] = ("<chmsg>", "</chmsg>"),
        user_token: str = "<user>",
        assistant_token: str = "<assistant>",
    ):
        super(GPT2ChatSpec, self).__init__(
            tokenizer_path,
            seq_len,
            layers,
            heads,
            dims,
            rate,
        )
        self.message_start, self.message_end = message_boundaries
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.user_pattern = re.compile(rf"{self.user_token}.*?{self.message_end}(.*?)")

    def encode_context(self, context: str) -> List[int]:
        return self.tokenizer.encode(
            f"{self.message_start}{self.user_token}{self.vocab.bos_token}"
            + f"{context}{self.vocab.eos_token}{self.message_end}"
        ).ids

    def decode_tokens(self, tokens: List[int]) -> str:
        response = self.tokenizer.decode(tokens)
        return response


def chat_with_gpt2_model(args: argparse.Namespace):
    spec = GPT2ChatSpec(
        tokenizer_path=args.tokenizer_path,
        seq_len=args.context_len,
        layers=args.layers,
        heads=args.heads,
        dims=args.dims,
        rate=args.rate,
    )
    config = GenerateConfig(
        context_len=args.context_len, nucleus_prob=args.nucleus_prob, use_gpu=args.use_gpu
    )

    generator = Generator(spec, config)
    generator.initialize(from_model=args.model_path)

    while True:
        print(generator.generate(input(">>"), chat=True))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "chat", help="Chat with GPT-2 model"
    )

    parser.add_argument("--tokenizer_path", required=True, help="tokenizer file path")
    parser.add_argument(
        "--model_path", required=True, help="trained GPT-2 model file path"
    )

    group = parser.add_argument_group("Model configurations")
    group.add_argument(
        "--context_len", default=64, type=int, help="maximum context length"
    )
    group.add_argument(
        "--layers", default=12, type=int, help="number of transformer layers"
    )
    group.add_argument(
        "--heads", default=16, type=int, help="number of multi-heads in attention layer"
    )
    group.add_argument(
        "--dims",
        default=1024,
        type=int,
        help="dimension of representation in each layer",
    )
    group.add_argument(
        "--rate",
        default=4,
        type=int,
        help="increase rate of dimensionality in bottleneck",
    )

    group = parser.add_argument_group("Generation options")
    group.add_argument(
        "--nucleus_prob",
        default=0.85,
        type=float,
        help="probability threshold for nucleus sampling",
    )
    group.add_argument(
        "--use_gpu", action="store_true", help="use gpu device in inferencing"
    )

    parser.set_defaults(func=chat_with_gpt2_model)
