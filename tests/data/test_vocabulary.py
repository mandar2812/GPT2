from io import StringIO
from src.gpt2.data import Vocab

_FAKE_VOCAB = {
    "[UNK]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[PAD]": 3,
    "TOKEN#1": 4,
    "TOKEN#2": 5,
    "TOKEN#3": 6,
    "TOKEN#4": 7,
}


def test_vocab_getitem():
    vocab = Vocab(
        vocab=_FAKE_VOCAB,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    # Get index by token.
    assert vocab["[BOS]"] == 1
    assert vocab["[EOS]"] == 2
    assert vocab["[PAD]"] == 3
    assert vocab["[UNK]"] == 0
    assert vocab["TOKEN#1"] == 4
    assert vocab["TOKEN#2"] == 5
    assert vocab["TOKEN#3"] == 6
    assert vocab["TOKEN#4"] == 7

    # Get token by index.
    assert vocab[1] == "[BOS]"
    assert vocab[2] == "[EOS]"
    assert vocab[3] == "[PAD]"
    assert vocab[0] == "[UNK]"
    assert vocab[4] == "TOKEN#1"
    assert vocab[5] == "TOKEN#2"
    assert vocab[6] == "TOKEN#3"
    assert vocab[7] == "TOKEN#4"


def test_vocab_contains():
    vocab = Vocab(
        vocab=_FAKE_VOCAB,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    # The vocabulary must contain the belows.
    assert "[BOS]" in vocab
    assert "[EOS]" in vocab
    assert "[PAD]" in vocab
    assert "[UNK]" in vocab
    assert "TOKEN#1" in vocab
    assert "TOKEN#2" in vocab
    assert "TOKEN#3" in vocab
    assert "TOKEN#4" in vocab

    # These are not defined in the vocabulary.
    assert "TOKEN#5" not in vocab
    assert "TOKEN#6" not in vocab
    assert "TOKEN#7" not in vocab
    assert "TOKEN#8" not in vocab


def test_vocab_len():
    vocab = Vocab(
        vocab=_FAKE_VOCAB,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    assert len(vocab) == 8


def test_vocab_properties():
    vocab = Vocab(
        vocab=_FAKE_VOCAB,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    # Get indices of special tokens by properties.
    assert vocab.unk_idx == 0
    assert vocab.bos_idx == 1
    assert vocab.eos_idx == 2
    assert vocab.pad_idx == 3
