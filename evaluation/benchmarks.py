"""
benchmarks.py
==============
Loads standard tokenizer-evaluation benchmark datasets.

We use two genuinely different benchmarks:

  1. WikiText-103  - long-form, clean Wikipedia articles (test split). Good for
     measuring tokenizer behavior on prose similar to (but disjoint from) our
     training corpus.
  2. Penn Treebank (PTB) - older, smaller, different register (1989 WSJ news
     text) with distinct tokenization conventions (e.g. pre-split contractions,
     numbers replaced with <unk>). Good for measuring how the tokenizer
     generalizes to out-of-domain text.

WHY NOT WIKITEXT-2 AS THE SECOND BENCHMARK
---------------------------------------------
An earlier version of this file used WikiText-2 here, reasoning that its much
smaller size would stress-test the tokenizer differently. That reasoning was
wrong in practice: WikiText-2's test split is not an independently-sampled
subset -- it shares the same test article boundaries as WikiText-103, just
with a smaller *training* set. Evaluating on both with a max_docs cap just
re-evaluates the same text twice (confirmed by comparing the two splits
directly: identical row counts, identical first 2000+ entries). So it added
zero signal. PTB, sourced as plain text below, is genuinely different text
with a different register and predates Wikipedia-style writing, so it's a
real out-of-domain check.

WHY PTB IS LOADED AS PLAIN TEXT, NOT VIA `datasets`
------------------------------------------------------
The `ptb_text_only` repo on the Hub only ships a legacy Python loading script,
which current `datasets` versions refuse to execute (no more implicit
`trust_remote_code`). Rather than depend on a dataset that's being phased out
upstream, we fetch the original plain-text PTB files directly from the
`wojzaremba/lstm` GitHub repo, which has hosted the canonical preprocessed PTB
benchmark files (the same files used by Mikolov's and Zaremba's original RNN
language-modeling papers) for years. No Hub config quirks, no Python code
execution.

Both loaders return a `list[str]` of document/line strings, the contract the
rest of the eval pipeline (`evaluate.py`) expects. If you'd rather plug in
LAMBADA or another benchmark, add a loader function here following the same
signature and register it in BENCHMARK_LOADERS below.
"""

import urllib.request

from datasets import load_dataset

PTB_TEST_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"


def load_wikitext103(split: str = "test", max_docs: int = None) -> list:
    """Returns a list of non-empty text lines from WikiText-103 test split.

    NOTE: the original `wikitext` dataset repo on the Hub ships a legacy
    loading script (no namespace) that newer `datasets`/`huggingface_hub`
    versions fail to parse (HfUriError: "Repository id must be
    'namespace/name'"). `Salesforce/wikitext` is the actively maintained
    parquet mirror with identical contents/splits and avoids that bug.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    lines = [row["text"] for row in ds if row["text"].strip()]
    if max_docs:
        lines = lines[:max_docs]
    return lines


def load_ptb(split: str = "test", max_docs: int = None) -> list:
    """Returns a list of non-empty lines from the Penn Treebank test set,
    fetched as plain text directly from GitHub (no `datasets` library, no
    Hub config, no legacy loading script).

    Note: only the "test" split is wired up here, since that's what
    evaluate.py uses by default. If you want train/valid too, swap
    ptb.test.txt for ptb.train.txt / ptb.valid.txt in PTB_TEST_URL.
    """
    if split != "test":
        raise ValueError(f"load_ptb only supports split='test' currently, got '{split}'")

    with urllib.request.urlopen(PTB_TEST_URL) as response:
        raw_text = response.read().decode("utf-8")

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    if max_docs:
        lines = lines[:max_docs]
    return lines


BENCHMARK_LOADERS = {
    "wikitext-103": load_wikitext103,
    "ptb": load_ptb,
}
