"""
train_bpe.py
=============
Stage 2 of the pipeline: train a Byte-Pair Encoding tokenizer on the extracted corpus.

LIBRARY CHOICE: Hugging Face `tokenizers` vs. SentencePiece
-------------------------------------------------------------
Both are valid. We use `tokenizers` here because:
  - It's Rust-backed and very fast to train/encode even on 100MB+ corpora.
  - It exposes the BPE merge list and vocabulary directly as JSON, which we use
    later to power the frontend's "what tokens exist" explorer.
  - Pre/post-processing (byte-level pre-tokenization, special token handling) is
    declarative and easy to reason about.

SentencePiece is an equally good choice and is what's used by T5, LLaMA, etc.
Its main advantages: built-in lossless detokenization via the unigram language model
option, and no separate pre-tokenization step needed since it treats text as a raw
unicode stream. If you'd rather use SentencePiece, swap this module for:

    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input="corpus.txt", model_prefix="bpe_tokenizer", vocab_size=32000,
        model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

The rest of this pipeline (evaluation, API, frontend) expects a `tokenizer.json` file
in the HF `tokenizers` format, so if you switch to SentencePiece you'll need to either
convert the resulting model or adapt evaluation/evaluate.py and api/main.py accordingly.

WHAT THIS DOES
--------------
1. Initializes a BPE model with byte-level pre-tokenization (so any unicode input,
   including emoji and non-Latin scripts, can always be encoded -- no OOV at the byte
   level, which is why this is the same strategy used by GPT-2/GPT-3/GPT-4).
2. Configures special tokens: <unk>, <s> (BOS), </s> (EOS), <pad>.
3. Trains on the corpus with configurable vocab_size and min_frequency.
4. Saves the resulting tokenizer to a single tokenizer.json, plus a human-readable
   training_config.json recording the hyperparameters used (for reproducibility).

USAGE
-----
    python train_bpe.py \
        --corpus ../corpus/en_wikipedia_corpus.txt \
        --vocab-size 32000 \
        --min-frequency 2 \
        --out-dir ../tokenizer_output
"""

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


def build_tokenizer() -> Tokenizer:
    """Construct an untrained BPE tokenizer with byte-level pre-tokenization.

    Byte-level pre-tokenization (the same scheme GPT-2 uses) means the
    tokenizer first maps raw text into a sequence of bytes-as-unicode-chars,
    then learns BPE merges over that alphabet. This guarantees universal
    coverage -- any input, in any language or script, can be encoded without
    ever hitting an <unk>, because every possible byte has a representation.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # NFKC normalization folds visually-equivalent unicode sequences (e.g.
    # full-width vs. half-width characters, combining accents) into a single
    # canonical form before tokenization -- reduces needless vocabulary
    # fragmentation from encoding quirks rather than real linguistic variety.
    tokenizer.normalizer = NFKC()

    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def train(
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list,
    out_dir: Path,
) -> dict:
    tokenizer = build_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        # Byte-level alphabet must be seeded into the initial vocabulary, or
        # the trainer may not reserve slots for every raw byte value -- this
        # is what byte_level.alphabet() provides.
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=True,
    )

    start = time.time()
    print(f"[train] Training BPE on {corpus_path} "
          f"(vocab_size={vocab_size}, min_frequency={min_frequency})...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    elapsed = time.time() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    actual_vocab_size = tokenizer.get_vocab_size()

    config = {
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "pre_tokenizer": "ByteLevel",
        "normalizer": "NFKC",
        "model": "BPE",
        "corpus_file": str(corpus_path),
        "corpus_size_bytes": corpus_path.stat().st_size,
        "training_seconds": elapsed,
        "tokenizer_file": str(tokenizer_path),
    }

    config_path = out_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[train] Done in {elapsed:.1f}s. Actual vocab size: {actual_vocab_size}")
    print(f"[train] Saved tokenizer to {tokenizer_path}")
    print(f"[train] Saved config to {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text corpus.")
    parser.add_argument("--corpus", required=True, help="Path to the training corpus .txt file")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out-dir", default="../tokenizer_output")
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=SPECIAL_TOKENS,
        help="Special tokens to reserve in the vocabulary, e.g. <unk> <s> </s> <pad>",
    )
    args = parser.parse_args()

    train(
        corpus_path=Path(args.corpus),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
