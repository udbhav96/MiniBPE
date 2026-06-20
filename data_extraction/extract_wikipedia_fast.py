"""
extract_wikipedia_fast.py
==========================
FAST-PATH ALTERNATIVE to extract_wikipedia.py.

WHY THIS EXISTS
----------------
extract_wikipedia.py streams the raw XML dump, which is the "correct" approach for
a from-scratch pipeline but requires downloading several GB before you extract a
single byte. If you're iterating on hyperparameters and just need a representative
100MB corpus quickly, use this script instead: it streams the pre-cleaned
`wikimedia/wikipedia` dataset from the Hugging Face Hub, which has already had
wikitext markup stripped, and stops once the size limit is hit -- so you never
download more than you asked for.

TRADE-OFF vs. the XML dump approach:
    + Much faster to get started (no multi-GB download, streaming via HF datasets)
    + Already cleaned (no mwparserfromhell pass needed)
    - Snapshot date is whatever HF last published (usually within the last few months)
    - Slightly less control over the exact cleaning rules applied
    - Adds a dependency on the `datasets` library and HF Hub availability

Both scripts produce the same output format (a single .txt file + manifest.json),
so they're interchangeable inputs to tokenizer_training/train_bpe.py.

USAGE
-----
    python extract_wikipedia_fast.py --lang en --size-limit-mb 100 --out-dir ./corpus
"""

import argparse
import json
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        "Missing dependency 'datasets'. Install with:\n    pip install datasets\n"
    )

# HF dataset snapshot naming follows wikimedia/wikipedia with a date + language config,
# e.g. "20231101.en". We use load_dataset's streaming mode so we never materialize
# the full split (which is many GB) on disk.
DATASET_NAME = "wikimedia/wikipedia"


def main():
    parser = argparse.ArgumentParser(description="Fast Wikipedia extraction via HF datasets.")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--size-limit-mb", type=float, default=100)
    parser.add_argument("--out-dir", default="./corpus")
    parser.add_argument("--min-article-chars", type=int, default=500)
    parser.add_argument("--snapshot", default="20231101",
                         help="HF wikimedia/wikipedia dump date config, e.g. 20231101")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.lang}_wikipedia_corpus.txt"
    manifest_path = out_dir / "extraction_manifest.json"

    config_name = f"{args.snapshot}.{args.lang}"
    print(f"[extract-fast] Streaming dataset config '{config_name}' from {DATASET_NAME}...")

    # streaming=True avoids downloading the entire split up front -- we pull
    # examples lazily and stop as soon as we've written enough text.
    ds = load_dataset(DATASET_NAME, config_name, split="train", streaming=True)

    size_limit_bytes = int(args.size_limit_mb * 1024 * 1024)
    stats = {"articles_kept": 0, "articles_skipped": 0, "bytes_written": 0, "started_at": time.time()}

    with open(out_path, "w", encoding="utf-8") as out_f:
        for example in ds:
            title = example.get("title", "")
            text = example.get("text", "")

            if len(text) < args.min_article_chars:
                stats["articles_skipped"] += 1
                continue

            block = f"\n\n=== {title} ===\n\n{text}\n"
            encoded = block.encode("utf-8")
            out_f.write(block)
            stats["bytes_written"] += len(encoded)
            stats["articles_kept"] += 1

            if stats["bytes_written"] >= size_limit_bytes:
                break

            if stats["articles_kept"] % 2000 == 0:
                mb = stats["bytes_written"] / (1024 * 1024)
                print(f"\r[extract-fast] kept={stats['articles_kept']} written={mb:.1f}MB", end="")

    print()
    stats["finished_at"] = time.time()
    stats["elapsed_seconds"] = stats["finished_at"] - stats["started_at"]
    stats["lang"] = args.lang
    stats["size_limit_mb"] = args.size_limit_mb
    stats["output_file"] = str(out_path)
    stats["source"] = f"{DATASET_NAME} config={config_name} (streaming)"
    stats["license_note"] = (
        "Wikipedia text is licensed CC BY-SA 4.0. Redistribution requires attribution "
        "and share-alike compliance. See https://creativecommons.org/licenses/by-sa/4.0/"
    )

    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[extract-fast] Done. {stats['articles_kept']} articles, "
          f"{stats['bytes_written']/(1024*1024):.2f}MB written to {out_path}")


if __name__ == "__main__":
    main()
