import argparse
import json
import os
from typing import List

import torch.nn as nn
from tokenizers import Tokenizer

from src.gpt2.data import Vocab
from src.gpt2.generation import GenerateConfig, GenerationSpec, Generator
from src.gpt2.modeling import Transformer


class GPT2GenerationSpec(GenerationSpec):
    def __init__(
        self,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        dropout: float,
        scaled_softmax: bool
    ):
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.scaled_softmax = scaled_softmax

    def initialize(self):
        self.tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.vocab = Vocab(vocab=self.tokenizer.get_vocab())

    def construct_model(self) -> nn.Module:
        return Transformer(
            layers=self.layers,
            pad_idx=self.vocab.pad_idx,
            words=len(self.vocab),
            heads=self.heads,
            dims=self.dims,
            rate=self.rate,
            dropout=self.dropout,
            scaled_softmax=self.scaled_softmax,
            bidirectional=False,
        )

    @property
    def stop_token(self) -> int:
        return self.vocab.eos_idx

    def encode_context(self, context: str) -> List[int]:
        return self.tokenizer.encode(context).ids

    def decode_tokens(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


def generate_sentence_with_gpt2_model(args: argparse.Namespace):
    if args.from_model_config:
        with open(
            args.from_model_config, "r", encoding="utf-8"
        ) as f:
            model_config = json.load(f)
            model_kwargs = {
                "layers": model_config["layers"],
                "heads": model_config["heads"],
                "dims": model_config["dims"],
                "rate": model_config["rate"],
                "dropout": model_config["dropout"],
                "scaled_softmax": model_config["scaled_softmax"],
            }
    else:
        model_kwargs = {
            "layers": args.layers,
            "heads": args.heads,
            "dims": args.dims,
            "rate": args.rate,
            "dropout": args.dropout,
            "scaled_softmax": args.scaled_softmax,
        }
    spec = GPT2GenerationSpec(
        tokenizer_path=args.tokenizer_path,
        seq_len=args.context_len,
        **model_kwargs,
    )
    config = GenerateConfig(
        context_len=args.context_len,
        nucleus_prob=args.nucleus_prob,
        use_gpu=args.use_gpu,
    )

    generator = Generator(spec, config)
    generator.initialize(from_model=args.model_path)

    while True:
        print(generator.generate(input(">>")))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "generate", help="generate sentences with GPT-2 model"
    )

    parser.add_argument("--tokenizer_path", required=True, help="tokenizer file path")
    parser.add_argument(
        "--model_path", required=True, help="trained GPT-2 model file path"
    )

    group = parser.add_argument_group("Model configurations")
    group.add_argument(
        "--from_model_config",
        default=None,
        help="load model config from a json file",
    )
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
    group.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="probability that each element is dropped",
    )
    group.add_argument(
        "--scaled_softmax",
        action="store_true",
        help="use scaled softmax in attention layer",
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

    parser.set_defaults(func=generate_sentence_with_gpt2_model)
