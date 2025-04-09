import argparse
import os
import json
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer

from src.gpt2.data import Dataset, QATokenizedCorpus, Vocab
from src.gpt2.modeling import Transformer
from src.gpt2.training import TrainConfig, Trainer, TrainingSpec
from src.gpt2.utils import fusing
from src.gpt2.train_model import GPT2TrainingSpec


class QASFTSpec(GPT2TrainingSpec):

    def __init__(
        self,
        train_corpus: str,
        eval_corpus: str,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        dropout: float,
        scaled_softmax: bool,
        base_lr: float,
        wd_rate: float,
        total_steps: int,
        use_grad_ckpt: bool,
        message_boundaries: tuple[str, str] = ("<chmsg>", "</chmsg>"),
        user_token: str = "<user>",
        assistant_token: str = "<assistant>",
    ):
        super(QASFTSpec, self).__init__(
            train_corpus=train_corpus,
            eval_corpus=eval_corpus,
            tokenizer_path=tokenizer_path,
            seq_len=seq_len,
            layers=layers,
            heads=heads,
            dims=dims,
            rate=rate,
            dropout=dropout,
            scaled_softmax=scaled_softmax,
            base_lr=base_lr,
            wd_rate=wd_rate,
            total_steps=total_steps,
            use_grad_ckpt=use_grad_ckpt,
        )
        self.message_start, self.message_end = message_boundaries
        self.user_token = user_token
        self.assistant_token = assistant_token

    def initialize(self):
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.vocab = Vocab(vocab=self.tokenizer.get_vocab())
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = QATokenizedCorpus(
            self.train_corpus,
            self.vocab,
            seq_len=self.seq_len,
            message_boundaries=(self.message_start, self.message_end),
            user_token=self.user_token,
            assistant_token=self.assistant_token,
        )
        eval_dataset = QATokenizedCorpus(
            self.eval_corpus,
            self.vocab,
            seq_len=self.seq_len,
            message_boundaries=(self.message_start, self.message_end),
            user_token=self.user_token,
            assistant_token=self.assistant_token,
        )
        return train_dataset, eval_dataset

    def train_objective(
        self, data: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        logits = model(data["input"], use_grad_ckpt=self.use_grad_ckpt)
        # 7️⃣ Apply loss mask
        target_ids = data["output"].clone()
        target_ids[data["loss_mask"] == 0] = -100  # Ignore user query tokens
        loss = self.criterion(logits.transpose(1, 2), target_ids)
        return {"loss": loss}

    def eval_objective(
        self, data: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data["input"], past=None)
        # 7️⃣ Apply loss mask
        target_ids = data["output"].clone()
        target_ids[data["loss_mask"] == 0] = -100  # Ignore user query tokens
        loss = self.criterion(logits.transpose(1, 2), target_ids)
        return {"loss": loss}


class RLFTSpec(TrainingSpec):
    def __init__(
        self,
        train_corpus: str,
        eval_corpus: str,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        dropout: float,
        scaled_softmax: bool,
        base_lr: float,
        wd_rate: float,
        total_steps: int,
        use_grad_ckpt: bool,
        max_context_len: int,
        message_boundaries: tuple[str, str] = ("<chmsg>", "</chmsg>"),
        user_token: str = "<user>",
        assistant_token: str = "<assistant>",
    ):
        self.train_corpus = train_corpus
        self.eval_corpus = eval_corpus
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.scaled_softmax = scaled_softmax
        self.base_lr = base_lr
        self.wd_rate = wd_rate
        self.total_steps = total_steps
        self.use_grad_ckpt = use_grad_ckpt
        self.max_context_len = max_context_len
        self.message_start, self.message_end = message_boundaries
        self.user_token = user_token
        self.assistant_token = assistant_token

    def initialize(self):
        self.tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.vocab = Vocab(vocab=self.tokenizer.get_vocab())
        self.chmsg_token_id = self.vocab[self.message_start]
        self.assistant_token_id = self.vocab[self.assistant_token]
        self.end_chmsg_token_id = self.vocab[self.message_end]
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",  # Ensures masked tokens don’t contribute
        )

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = QATokenizedCorpus(
            self.train_corpus,
            self.vocab,
            seq_len=self.seq_len,
            message_boundaries=(self.message_start, self.message_end),
            user_token=self.user_token,
            assistant_token=self.assistant_token,
        )
        eval_dataset = QATokenizedCorpus(
            self.eval_corpus,
            self.vocab,
            seq_len=self.seq_len,
            message_boundaries=(self.message_start, self.message_end),
            user_token=self.user_token,
            assistant_token=self.assistant_token,
        )
        return train_dataset, eval_dataset

    def construct_model(self) -> nn.Module:
        return Transformer(
            layers=self.layers,
            pad_idx=self.vocab.pad_idx,
            words=len(self.vocab),
            heads=self.heads,
            dims=self.dims,
            rate=self.rate,
            dropout=self.dropout,
            bidirectional=False,
            scaled_softmax=self.scaled_softmax,
        )

    def model_config(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "pad_idx": self.vocab.pad_idx,
            "words": len(self.vocab),
            "heads": self.heads,
            "dims": self.dims,
            "rate": self.rate,
            "dropout": self.dropout,
            "bidirectional": False,
            "scaled_softmax": self.scaled_softmax,
        }

    def create_optimizer(self, params):
        optimizer = fusing.Adam(params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50000, T_mult=1, eta_min=1e-6
        )
        return optimizer, scheduler

    def slice_assistant_response(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts only the assistant's response from logits, based on where the input sequence ends.
        Prepends `<chmsg> <assistant>` and appends `</chmsg>`.
        """
        batch_size, seq_length, vocab_size = input_ids.shape

        # Compute where the assistant response starts
        input_lengths = (input_ids != self.vocab.pad_idx).sum(
            dim=1
        )  # Shape: (batch_size,)

        # Slice out the assistant's response logits
        response_logits = []
        for i in range(batch_size):
            response_logits.append(
                logits[i, input_lengths[i] :, :]
            )  # Take from input end onward

        # Convert list to tensor (pad dynamically if necessary)
        response_logits = torch.nn.utils.rnn.pad_sequence(
            response_logits, batch_first=True, padding_value=float("-inf")
        )

        # Construct prefix `<chmsg> <assistant>`
        assistant_prefix = torch.full(
            (batch_size, 2, vocab_size), float("-inf"), device=logits.device
        )
        assistant_prefix[:, 0, self.chmsg_token_id] = 0
        assistant_prefix[:, 1, self.assistant_token_id] = 0

        # Construct suffix `</chmsg>`
        end_token = torch.full(
            (batch_size, 1, vocab_size), float("-inf"), device=logits.device
        )
        end_token[:, 0, self.end_chmsg_token_id] = 0

        # Concatenate everything
        adjusted_logits = torch.cat(
            [assistant_prefix, response_logits, end_token], dim=1
        )

        return adjusted_logits

    def train_objective(
        self, data: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Finetune the model on QA sequences. The model should learn to generate
        the assistant's response autoregressively.
        """

        batch_size = data["input"].shape[0]
        max_seq_len = self.max_context_len  # Maximum sequence length

        # 1️⃣ Start with user query as input
        generated = data["input"].clone()

        # 2️⃣ Autoregressive loop to generate the assistant's response
        for _ in range(max_seq_len - data["input"].shape[1]):
            logits = model(generated, use_grad_ckpt=self.use_grad_ckpt)  # [B, L, V]

            # Get the next token (greedy decoding)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat((generated, next_token), dim=1)  # Append next token

            # Stop early if all batch elements have generated `</s>`
            if torch.all(next_token.squeeze(-1) == self.vocab[self.message_end]):
                break

        # 3️⃣ Determine maximum sequence length
        max_len = max(generated.shape[1], data["output"].shape[1])

        # 4️⃣ Pad `logits` if shorter than `max_len`
        if logits.shape[1] < max_len:
            pad_size = max_len - logits.shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size, logits.shape[2]),  # Shape: [B, pad_size, V]
                float("-inf"),  # Logits should be masked
                device=logits.device,
            )
            logits = torch.cat((logits, pad_tensor), dim=1)

        # 5️⃣ Pad `generated` if shorter than `max_len`
        if generated.shape[1] < max_len:
            pad_size = max_len - generated.shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size),
                self.vocab.pad_idx,
                device=generated.device,
            )
            generated = torch.cat((generated, pad_tensor), dim=1)

        # 6️⃣ Pad `target_ids` if shorter than `max_len`
        if data["output"].shape[1] < max_len:
            pad_size = max_len - data["output"].shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size),
                -100,  # Mask for loss
                device=data["output"].device,
            )
            data["output"] = torch.cat((data["output"], pad_tensor), dim=1)

            pad_mask = torch.zeros(
                (batch_size, pad_size), device=data["loss_mask"].device
            )
            data["loss_mask"] = torch.cat((data["loss_mask"], pad_mask), dim=1)

        # 7️⃣ Apply loss mask
        target_ids = data["output"].clone()
        target_ids[data["loss_mask"] == 0] = -100  # Ignore user query tokens

        # 8️⃣ Compute loss
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        return {"loss": loss}

    def eval_objective(
        self, data: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Finetune the model on QA sequences. The model should learn to generate
        the assistant's response autoregressively.
        """

        batch_size = data["input"].shape[0]
        max_seq_len = self.max_context_len  # Maximum sequence length

        # 1️⃣ Start with user query as input
        generated = data["input"].clone()

        past = None
        # 2️⃣ Autoregressive loop to generate the assistant's response
        for _ in range(max_seq_len - data["input"].shape[1]):
            logits, past = model(generated, past=past)  # [B, L, V]

            # Get the next token (greedy decoding)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat((generated, next_token), dim=1)  # Append next token

            # Stop early if all batch elements have generated `</s>`
            if torch.all(next_token.squeeze(-1) == self.vocab[self.message_end]):
                break

        # 3️⃣ Determine maximum sequence length
        max_len = max(generated.shape[1], data["output"].shape[1])

        # 4️⃣ Pad `logits` if shorter than `max_len`
        if logits.shape[1] < max_len:
            pad_size = max_len - logits.shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size, logits.shape[2]),  # Shape: [B, pad_size, V]
                float("-inf"),  # Logits should be masked
                device=logits.device,
            )
            logits = torch.cat((logits, pad_tensor), dim=1)

        # 5️⃣ Pad `generated` if shorter than `max_len`
        if generated.shape[1] < max_len:
            pad_size = max_len - generated.shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size),
                self.vocab.pad_idx,
                device=generated.device,
            )
            generated = torch.cat((generated, pad_tensor), dim=1)

        # 6️⃣ Pad `target_ids` if shorter than `max_len`
        if data["output"].shape[1] < max_len:
            pad_size = max_len - data["output"].shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size),
                -100,  # Mask for loss
                device=data["output"].device,
            )
            data["output"] = torch.cat((data["output"], pad_tensor), dim=1)

            pad_mask = torch.zeros(
                (batch_size, pad_size), device=data["loss_mask"].device
            )
            data["loss_mask"] = torch.cat((data["loss_mask"], pad_mask), dim=1)

        # 7️⃣ Apply loss mask
        target_ids = data["output"].clone()
        target_ids[data["loss_mask"] == 0] = -100  # Ignore user query tokens

        # 8️⃣ Compute loss
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        return {"loss": loss}


def train_qa_model(args: argparse.Namespace):
    if args.from_model_config:
        with open(
            os.path.join(args.corpus_dir, args.from_model_config), "r", encoding="utf-8"
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
    spec = QASFTSpec(
        train_corpus=os.path.join(args.corpus_dir, args.train_corpus),
        eval_corpus=os.path.join(args.corpus_dir, args.eval_corpus),
        tokenizer_path=os.path.join(args.corpus_dir, args.tokenizer_path),
        seq_len=args.seq_len,
        **model_kwargs,
        base_lr=args.base_lr,
        wd_rate=args.wd_rate,
        total_steps=args.total_steps,
        use_grad_ckpt=args.use_grad_ckpt,
    )

    config = TrainConfig(
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        total_steps=args.total_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_version_steps=args.save_version_steps,
        save_model_path=os.path.join(args.corpus_dir, args.save_model_path),
        save_checkpoint_path=os.path.join(args.corpus_dir, args.save_checkpoint_path),
        description="Fine-tuning GPT-2 for QA",
        log_format="train/loss: {train_loss:.4f}, eval/loss: {eval_loss:.4f}",
        use_amp=args.use_amp,
        gpus=args.gpus,
    )

    Trainer(spec, config).train(
        from_checkpoint=(
            os.path.join(args.corpus_dir, args.from_checkpoint)
            if args.from_checkpoint
            else None
        ),
        from_pretrained=(
            os.path.join(args.corpus_dir, args.from_pretrained)
            if args.from_pretrained
            else None
        ),
    )


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("finetune", help="fine-tune GPT-2 model for QA")

    group = parser.add_argument_group("Corpus and vocabulary")
    group.add_argument(
        "corpus_dir", help="root directory of corpus files", default=os.getcwd()
    )
    group.add_argument(
        "--train_corpus", required=True, help="training corpus file path"
    )
    group.add_argument(
        "--eval_corpus", required=True, help="evaluation corpus file path"
    )
    group.add_argument("--tokenizer_path", required=True, help="tokenizer file path")

    group = parser.add_argument_group("Model configurations")
    group.add_argument(
        "--from_model_config",
        default=None,
        help="load model config from a json file",
    )
    group.add_argument(
        "--seq_len", default=64, type=int, help="maximum sequence length"
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

    group = parser.add_argument_group("Training and evaluation")
    group.add_argument(
        "--batch_train", default=64, type=int, help="number of training batch size"
    )
    group.add_argument(
        "--batch_eval", default=64, type=int, help="number of evaluation batch size"
    )
    group.add_argument(
        "--base_lr", default=1e-4, type=float, help="default learning rate"
    )
    group.add_argument("--wd_rate", default=1e-2, type=float, help="weight decay rate")

    group.add_argument(
        "--total_steps",
        default=1000000,
        type=int,
        help="number of total training steps",
    )
    group.add_argument(
        "--eval_steps",
        default=500,
        type=int,
        help="period to evaluate model and record metrics",
    )
    group.add_argument(
        "--save_steps",
        default=1000,
        type=int,
        help="period to save training state to checkpoint",
    )
    group.add_argument(
        "--save_version_steps",
        default=-1,
        type=int,
        help="period to save a versioned/branched model.",
    )

    group = parser.add_argument_group("Saving and restoring")
    group.add_argument(
        "--save_model_path",
        default="model.pth",
        help="save trained model weights to the file",
    )
    group.add_argument(
        "--save_checkpoint_path",
        default="checkpoint.pth",
        help="save training state to the checkpoint file",
    )
    group.add_argument(
        "--from_checkpoint",
        default=None,
        help="load last training state from checkpoint file",
    )
    group.add_argument(
        "--from_pretrained",
        default=None,
        help="initialize parameters from pretrained model",
    )

    group = parser.add_argument_group("Extensions")
    group.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed-precision in training",
    )
    group.add_argument(
        "--use_grad_ckpt",
        action="store_true",
        help="use gradient checkpointing in transformer layers",
    )
    group.add_argument(
        "--gpus",
        default=None,
        type=int,
        help="number of gpu devices to use in training",
    )

    parser.set_defaults(func=train_qa_model)
