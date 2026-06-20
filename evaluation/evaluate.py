"""
evaluate.py
============
Stage 3 of the pipeline: evaluate the trained BPE tokenizer against standard
benchmarks and report tokens/word, encode/decode speed, and compression ratio.

BENCHMARKS USED: wikitext-103 and ptb (see benchmarks.py for how each is
sourced -- both route around dataset repos on the Hub whose legacy loading
scripts break under current `datasets`/`huggingface_hub` versions).

IF YOU WANT A DIFFERENT SECOND BENCHMARK
----------------------------------------
If you want a different second benchmark, follow the pattern in
benchmarks.py: any loader that returns a `list[str]` of document/line
strings works with `evaluate_on_corpus()` below, regardless of where the
text comes from (Hub, GitHub, a local file).

METRICS DEFINED
----------------
- tokens_per_word: (total subword tokens) / (total whitespace-split words).
  Lower is "better" in the sense of fewer tokens needed per unit of language --
  but this must be read alongside vocab size, since a larger vocabulary
  mechanically produces a lower tokens/word at the cost of model embedding size.

- compression_ratio: (total UTF-8 bytes of raw text) / (total token count).
  This tells you, on average, how many bytes of text one token "buys" you --
  directly relevant to context-window economics (more bytes per token = more
  effective context for the same token budget).

- encode_chars_per_sec / decode_chars_per_sec: throughput, measured by encoding
  the whole benchmark text and timing it. Reported as both raw seconds and a
  normalized chars/sec figure so results are comparable across benchmarks of
  different sizes.

USAGE
-----
    python evaluate.py \
        --tokenizer-path ../tokenizer_output/tokenizer.json \
        --benchmarks wikitext-103 ptb \
        --max-docs 2000 \
        --out ../tokenizer_output/evaluation_report.json
"""

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer
from benchmarks import BENCHMARK_LOADERS


def evaluate_on_corpus(tokenizer: Tokenizer, texts: list) -> dict:
    """Compute tokens/word, compression ratio, and encode/decode throughput
    for a tokenizer over a list of text documents."""

    total_words = 0
    total_tokens = 0
    total_bytes = 0
    all_token_ids = []

    encode_start = time.perf_counter()
    for text in texts:
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
        total_words += len(text.split())
        total_bytes += len(text.encode("utf-8"))
        all_token_ids.append(encoding.ids)
    encode_elapsed = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    for ids in all_token_ids:
        _ = tokenizer.decode(ids)
    decode_elapsed = time.perf_counter() - decode_start

    total_chars = sum(len(t) for t in texts)

    return {
        "num_documents": len(texts),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes_utf8": total_bytes,
        "tokens_per_word": round(total_tokens / total_words, 4) if total_words else None,
        "compression_ratio_bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else None,
        "encode_seconds": round(encode_elapsed, 4),
        "decode_seconds": round(decode_elapsed, 4),
        "encode_chars_per_sec": round(total_chars / encode_elapsed, 1) if encode_elapsed > 0 else None,
        "decode_chars_per_sec": round(total_chars / decode_elapsed, 1) if decode_elapsed > 0 else None,
    }


def run_evaluation(tokenizer_path: Path, benchmark_names: list, max_docs: int) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "benchmarks": {},
    }

    for name in benchmark_names:
        if name not in BENCHMARK_LOADERS:
            print(f"[eval] WARNING: unknown benchmark '{name}', skipping. "
                  f"Available: {list(BENCHMARK_LOADERS.keys())}")
            continue

        print(f"[eval] Loading benchmark '{name}'...")
        texts = BENCHMARK_LOADERS[name](max_docs=max_docs)
        print(f"[eval] Loaded {len(texts)} documents. Evaluating...")

        result = evaluate_on_corpus(tokenizer, texts)
        report["benchmarks"][name] = result
        print(f"[eval] {name}: tokens/word={result['tokens_per_word']}, "
              f"compression={result['compression_ratio_bytes_per_token']} bytes/token, "
              f"encode_speed={result['encode_chars_per_sec']} chars/sec")

    return report


def print_summary_table(report: dict):
    """Print a Markdown-formatted table summarizing results across benchmarks."""
    print("\n| Benchmark | Tokens/Word | Compression (bytes/token) | Encode (chars/s) | Decode (chars/s) |")
    print("|---|---|---|---|---|")
    for name, r in report["benchmarks"].items():
        print(f"| {name} | {r['tokens_per_word']} | {r['compression_ratio_bytes_per_token']} "
              f"| {r['encode_chars_per_sec']:,.0f} | {r['decode_chars_per_sec']:,.0f} |")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a BPE tokenizer on standard benchmarks.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["wikitext-103", "ptb"])
    parser.add_argument("--max-docs", type=int, default=2000,
                         help="Cap number of benchmark documents (keeps eval fast; set None-like via 0 for all)")
    parser.add_argument("--out", default="../tokenizer_output/evaluation_report.json")
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs > 0 else None
    report = run_evaluation(Path(args.tokenizer_path), args.benchmarks, max_docs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print_summary_table(report)
    print(f"\n[eval] Full report saved to {out_path}")


if __name__ == "__main__":
    main()
