import torch
from src.gpt2.data import Dataset, Vocab
from typing import Dict, Any, List, Optional, Generator


class TokenizedCorpus(Dataset):
    def __init__(
        self, corpus_path: str, vocab: Vocab, seq_len: int, repeat: bool = True, overlap: int = None
    ):
        self.corpus_fp = open(corpus_path, "r", encoding="utf-8")
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat
        self.overlap = overlap if overlap is not None else seq_len // 2  # Default to 50% overlap

    def skip(self, count: int):
        for _ in range(count):
            if not self.corpus_fp.readline():
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                self.corpus_fp.readline()

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

            # Use token indices rather than the token names directly.
            indices = [self.vocab[t] for t in line.split()]
            stride = self.seq_len - self.overlap

            if len(indices) + 2 <= self.seq_len:
                # If the sequence is shorter than seq_len - 2, pad it
                window = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
                window += [self.vocab.pad_idx] * (self.seq_len - len(window))
                yield {"input": window[:-1], "output": window[1:]}
            else:
                for start in range(0, len(indices), stride):
                    window = indices[start : start + (self.seq_len - 2)]
                    window = [self.vocab.bos_idx] + window  # Ensure each window starts with BOS
                    if start + (self.seq_len - 2) >= len(indices):
                        window.append(self.vocab.eos_idx)  # Only last window gets EOS
                    window += [self.vocab.pad_idx] * (self.seq_len - len(window))
                    yield {"input": window[:-1], "output": window[1:]}

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
