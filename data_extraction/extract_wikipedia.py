"""
extract_wikipedia.py
=====================
Stage 1 of the pipeline: build a plain-text training corpus from Wikipedia.

WHAT THIS DOES
--------------
1. Downloads (or reuses a cached) Wikipedia XML dump for a given language edition.
2. Streams the bz2-compressed XML with an incremental parser (no full decompression
   into memory -- dumps are tens of GB uncompressed).
3. For each <page>, pulls the <text> (raw wikitext) out of the latest <revision>.
4. Strips wikitext markup (templates, tables, refs, links, headings) down to
   clean prose using `mwparserfromhell`.
5. Filters out non-article pages (redirects, disambiguation stubs, very short pages).
6. Writes cleaned articles, one per line is NOT used (articles are multi-paragraph) --
   instead we write each article separated by a document boundary, and stop the moment
   the cumulative byte size hits the configured limit.

WHY A STREAMING PARSER
-----------------------
The full English Wikipedia dump is ~20GB compressed / ~90GB+ uncompressed. We only need
a fraction of that (e.g. 100MB of clean text) for a tokenizer training corpus -- BPE's
vocabulary statistics saturate long before you need the whole dump. Streaming with
iterparse + bz2 lets us stop as soon as we have enough, and never costs us full-dump
disk space.

USAGE
-----
    python extract_wikipedia.py \
        --lang en \
        --size-limit-mb 100 \
        --out-dir ./corpus \
        --min-article-chars 500

This will produce:
    ./corpus/en_wikipedia_corpus.txt      (the cleaned text, capped at size-limit-mb)
    ./corpus/extraction_manifest.json     (stats: articles kept/skipped, byte counts, timing)
"""

import argparse
import bz2
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import mwparserfromhell
except ImportError:
    print(
        "Missing dependency 'mwparserfromhell'. Install with:\n"
        "    pip install mwparserfromhell\n",
        file=sys.stderr,
    )
    raise

# Wikimedia dump index. We pick the latest "pages-articles" multistream dump,
# which contains current revisions only (no full edit history) -- this is the
# correct dump variant for text-corpus extraction; the full-history dumps are
# 10-20x larger and contain mostly redundant old revisions we don't want.
DUMP_URL_TEMPLATE = (
    "https://dumps.wikimedia.org/{lang}wiki/latest/"
    "{lang}wiki-latest-pages-articles.xml.bz2"
)

# Namespace used by MediaWiki XML export format.
MEDIAWIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def download_dump(lang: str, cache_dir: Path) -> Path:
    """Download the XML dump if not already cached locally. Dumps are large
    (multi-GB), so we always reuse a cached copy rather than re-downloading."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dump_path = cache_dir / f"{lang}wiki-latest-pages-articles.xml.bz2"

    if dump_path.exists():
        print(f"[extract] Using cached dump at {dump_path}")
        return dump_path

    url = DUMP_URL_TEMPLATE.format(lang=lang)
    print(f"[extract] Downloading dump from {url}")
    print("[extract] NOTE: full dumps are several GB; this can take a while.")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r[extract] {pct:5.1f}% downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dump_path, reporthook=_progress)
    print()
    return dump_path


def clean_wikitext(raw_wikitext: str) -> str:
    """Strip MediaWiki markup down to plain prose.

    Removes: templates ({{...}}), tables, file/image links, category links,
    reference tags, HTML comments, and heading markup -- while keeping the
    human-readable link text (e.g. [[Paris|the capital]] -> "the capital").
    """
    wikicode = mwparserfromhell.parse(raw_wikitext)

    # Strip templates/tables before getting plain text -- these are the main
    # source of leftover markup noise (infoboxes, citation templates, etc).
    for template in wikicode.filter_templates():
        try:
            wikicode.remove(template)
        except ValueError:
            pass  # already removed as part of a larger node

    text = wikicode.strip_code(normalize=True, collapse=True)

    # Drop residual reference/footnote artifacts and excess whitespace.
    text = re.sub(r"\[\d+\]", "", text)          # leftover [1] [2] citation markers
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse runs of blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse runs of spaces/tabs
    return text.strip()


def is_valid_article(title: str, wikitext: str, min_chars: int) -> bool:
    """Filter out redirects, disambiguation pages, and stubs that would add
    noise rather than useful language-modeling signal."""
    if wikitext is None:
        return False
    lowered = wikitext.lower().lstrip()
    if lowered.startswith("#redirect"):
        return False
    if "{{disambig" in lowered or "{{disambiguation" in lowered:
        return False
    if title.startswith(("Category:", "Template:", "File:", "Wikipedia:", "Portal:", "Help:")):
        return False
    if len(wikitext) < min_chars:
        return False
    return True


def stream_extract(
    dump_path: Path,
    out_path: Path,
    size_limit_bytes: int,
    min_article_chars: int,
) -> dict:
    """Stream-parse the bz2 XML dump and write cleaned article text to out_path
    until size_limit_bytes is reached. Returns a manifest dict with stats."""

    stats = {
        "articles_kept": 0,
        "articles_skipped": 0,
        "bytes_written": 0,
        "started_at": time.time(),
    }

    page_tag = f"{MEDIAWIKI_NS}page"
    title_tag = f"{MEDIAWIKI_NS}title"
    revision_tag = f"{MEDIAWIKI_NS}revision"
    text_tag = f"{MEDIAWIKI_NS}text"

    with bz2.open(dump_path, "rb") as bz_stream, open(out_path, "w", encoding="utf-8") as out_f:
        # iterparse streams the XML tree node-by-node instead of loading it
        # fully into memory -- essential for multi-GB dump files.
        context = ET.iterparse(bz_stream, events=("end",))

        for event, elem in context:
            if elem.tag != page_tag:
                continue

            title_elem = elem.find(title_tag)
            revision_elem = elem.find(revision_tag)
            title = title_elem.text if title_elem is not None else ""

            raw_text = None
            if revision_elem is not None:
                text_elem = revision_elem.find(text_tag)
                raw_text = text_elem.text if text_elem is not None else None

            if is_valid_article(title, raw_text, min_article_chars):
                cleaned = clean_wikitext(raw_text)
                if len(cleaned) >= min_article_chars:
                    block = f"\n\n=== {title} ===\n\n{cleaned}\n"
                    encoded = block.encode("utf-8")
                    out_f.write(block)
                    stats["bytes_written"] += len(encoded)
                    stats["articles_kept"] += 1
                else:
                    stats["articles_skipped"] += 1
            else:
                stats["articles_skipped"] += 1

            # CRITICAL: clear the element to free memory. Without this,
            # iterparse retains every parsed node and memory grows unbounded
            # over a multi-GB file.
            elem.clear()

            if stats["bytes_written"] >= size_limit_bytes:
                break

            if (stats["articles_kept"] + stats["articles_skipped"]) % 2000 == 0:
                mb_written = stats["bytes_written"] / (1024 * 1024)
                print(
                    f"\r[extract] kept={stats['articles_kept']} "
                    f"skipped={stats['articles_skipped']} "
                    f"written={mb_written:.1f}MB",
                    end="",
                )
                sys.stdout.flush()

    print()
    stats["finished_at"] = time.time()
    stats["elapsed_seconds"] = stats["finished_at"] - stats["started_at"]
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract a plain-text corpus from Wikipedia.")
    parser.add_argument("--lang", default="en", help="Wikipedia language code, e.g. en, fr, ja")
    parser.add_argument("--size-limit-mb", type=float, default=100,
                         help="Stop once this many MB of cleaned text have been written")
    parser.add_argument("--out-dir", default="./corpus", help="Output directory")
    parser.add_argument("--cache-dir", default="./dump_cache", help="Where to cache the downloaded dump")
    parser.add_argument("--min-article-chars", type=int, default=500,
                         help="Skip articles shorter than this many characters (filters stubs)")
    parser.add_argument("--dump-path", default=None,
                         help="Optional: path to an already-downloaded dump file, skips download")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.lang}_wikipedia_corpus.txt"
    manifest_path = out_dir / "extraction_manifest.json"

    dump_path = Path(args.dump_path) if args.dump_path else download_dump(args.lang, Path(args.cache_dir))

    size_limit_bytes = int(args.size_limit_mb * 1024 * 1024)
    print(f"[extract] Extracting up to {args.size_limit_mb}MB of {args.lang} Wikipedia text...")

    stats = stream_extract(dump_path, out_path, size_limit_bytes, args.min_article_chars)
    stats["lang"] = args.lang
    stats["size_limit_mb"] = args.size_limit_mb
    stats["output_file"] = str(out_path)
    stats["license_note"] = (
        "Wikipedia text is licensed CC BY-SA 4.0. Redistribution of this corpus or "
        "derived models must comply with attribution and share-alike terms. "
        "See https://creativecommons.org/licenses/by-sa/4.0/"
    )

    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[extract] Done. {stats['articles_kept']} articles, "
          f"{stats['bytes_written']/(1024*1024):.2f}MB written to {out_path}")
    print(f"[extract] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
