import json
import re
import torch
from src.gpt2.data import Dataset, Vocab
from typing import Dict, Any, List, Optional, Generator


class TokenizedCorpus(Dataset):
    def __init__(
        self,
        corpus_path: str,
        vocab: Vocab,
        seq_len: int,
        repeat: bool = True,
        overlap: Optional[int] = None,
    ):
        self.corpus_fp = open(corpus_path, "r", encoding="utf-8")
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat
        self.overlap = (
            overlap if overlap is not None else seq_len // 2
        )  # Default to 50% overlap

    def skip(self, count: int):
        for _ in range(count):
            if not self.corpus_fp.readline():
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                self.corpus_fp.readline()

    def process_tokenized_record(self, tokens: str) -> dict[str: list[int]]:
        """Convert tokenized text to indices."""
        # Use token indices rather than the token names directly.
        indices = [self.vocab[t] for t in tokens.split()]
        stride = self.seq_len - self.overlap

        if len(indices) + 2 <= self.seq_len:
            # If the sequence is shorter than seq_len - 2, pad it
            window = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
            window += [self.vocab.pad_idx] * (self.seq_len - len(window))
            return {"input": window[:-1], "output": window[1:]}
        else:
            for start in range(0, len(indices), stride):
                window = indices[start : start + (self.seq_len - 2)]

                # Add BOS only for the first window
                if start == 0:
                    window = [self.vocab.bos_idx] + window

                # Add EOS only for the last window
                if start + (self.seq_len - 2) >= len(indices):
                    window.append(self.vocab.eos_idx)

                window += [self.vocab.pad_idx] * (self.seq_len - len(window))
                return {"input": window[:-1], "output": window[1:]}

    def _fetch_one(self) -> Generator[Dict[str, List[int]], None, None]:
        while True:
            # Read subword-tokenized sequence from corpus.
            line = self.corpus_fp.readline()
            if not line:
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                continue
            tokens = json.loads(line)["tokens"]
            record = self.process_tokenized_record(tokens)
            if not record:
                continue # Skip malformed records
            yield record

    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        data = []
        generator = self._fetch_one()

        if batch is None:
            data.append(next(generator))
        else:
            while len(data) < batch:
                data.append(next(generator))

        data = {k: [d[k] for d in data] for k in data[0]}
        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def where(self) -> Dict[str, Any]:
        return {"offset": self.corpus_fp.tell()}

    def assign(self, where: Dict[str, Any]):
        self.corpus_fp.seek(where["offset"])

    def size(self, batch: Optional[int] = None) -> int:
        """Calculate the number of patterns in the corpus."""
        self.corpus_fp.seek(0)

        def _num_patterns(line: str) -> int:
            tokens = json.loads(line)["tokens"]
            return (
                (len(tokens.split()) - self.seq_len) // (self.seq_len - self.overlap)
            ) + 1

        num_patterns = sum(_num_patterns(line) for line in self.corpus_fp)
        self.corpus_fp.seek(0)
        if batch is None:
            return num_patterns
        else:
            return num_patterns // batch


class QATokenizedCorpus(TokenizedCorpus):
    """Question Answer fine-tuning dataset where each line contains a pre-tokenized conversation."""

    def __init__(
        self,
        corpus_path: str,
        vocab: Vocab,
        seq_len: int,
        repeat: bool = True,
        overlap: Optional[int] = None,
        message_boundaries: tuple[str, str] = ("<chmsg>", "</chmsg>"),
        user_token: str = "<user>",
        assistant_token: str = "<assistant>",
    ):
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.message_start, self.message_end = message_boundaries

        # Regex pattern to extract user and assistant messages
        self.user_pattern = re.compile(rf"{self.user_token}(.*?){self.message_end}")
        self.assistant_pattern = re.compile(
            rf"{self.assistant_token}(.*?){self.message_end}"
        )
        super(QATokenizedCorpus, self).__init__(
            corpus_path, vocab, seq_len, repeat=repeat, overlap=overlap
        )

    def process_tokenized_record(self, tokens: str) -> dict[str: list[int]]:
        if self.message_start not in tokens:
            # Treat this as a normal pretraining record
            record = super(QATokenizedCorpus, self).process_tokenized_record(tokens)
            loss_mask = [1] * len(record["output"])
            return {**record, "loss_mask": loss_mask}

        # Treat as a QA/chat record with user and assistant roles
        # Extract user and assistant messages
        user_match = self.user_pattern.search(tokens)
        assistant_match = self.assistant_pattern.search(tokens)

        if not user_match or not assistant_match:
            return {}

        question_tokens = user_match.group(1).strip().split()
        answer_tokens = assistant_match.group(1).strip().split()

        # Convert tokens to indices
        question_indices = [self.vocab[t] for t in question_tokens]
        answer_indices = [self.vocab[t] for t in answer_tokens]

        # Define input (only user message) and output (entire sequence)
        input_ids = (
            [self.vocab[t] for t in [self.message_start, self.user_token]]
            + question_indices
            + [self.vocab[self.message_end]]
        )
        sequence_indices = (
            input_ids
            + [self.vocab[t] for t in [self.message_start, self.assistant_token]]
            + answer_indices
            + [self.vocab[self.message_end]]
        )

        # Loss mask: 1 for assistant response, 0 otherwise
        loss_mask = [0] * len(input_ids) + [0, 0] + [1] * len(answer_indices) + [0]

        # Now do a sliding window of seq_len if the length is less than seq_len
        stride = self.seq_len - self.overlap
        if len(sequence_indices) <= self.seq_len:
            window = sequence_indices + [self.vocab.pad_idx] * (
                self.seq_len - len(sequence_indices)
            )
            loss_mask = loss_mask + [0] * (self.seq_len - len(loss_mask))
            return {
                "input": window[:-1],
                "output": window[1:],
                "loss_mask": loss_mask[1:],
            }
        else:
            for start in range(0, len(sequence_indices), stride):
                window = sequence_indices[start : start + self.seq_len]
                window += [self.vocab.pad_idx] * (self.seq_len - len(window))
                loss_mask_window = loss_mask[start : start + self.seq_len]
                loss_mask_window += [0] * (self.seq_len - len(loss_mask_window))
                return {
                    "input": window[:-1],
                    "output": window[1:],
                    "loss_mask": loss_mask_window[1:],
                }
