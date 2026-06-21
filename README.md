# BPE Lab — a from-scratch byte-pair encoding tokenizer pipeline

End-to-end pipeline: crawl Wikipedia → train a BPE tokenizer → evaluate it against
standard benchmarks → serve it through an interactive web UI.

```
bpe-tokenizer-pipeline/
├── requirements.txt              # Python deps for stages 1-3 + API
├── data_extraction/
│   ├── extract_wikipedia.py      # Stage 1 (primary): streams the raw XML dump
│   └── extract_wikipedia_fast.py # Stage 1 (fast path): streams HF's pre-cleaned dataset
├── tokenizer_training/
│   └── train_bpe.py              # Stage 2: trains the BPE model
├── evaluation/
│   ├── benchmarks.py             # Loaders for WikiText-103 / Penn Treebank
│   └── evaluate.py               # Stage 3: computes tokens/word, compression, speed
├── api/
│   └── main.py                   # FastAPI server bridging the tokenizer to the frontend
├── frontend/                     # Next.js + Tailwind v4 app (the UI)
│   └── src/
│       ├── app/page.tsx          # Main page
│       ├── components/           # TokenStream, BenchmarkTable, CompareView
│       └── lib/api.ts            # Typed client for the FastAPI backend
├── scripts/
│   └── run_pipeline.sh           # Runs stages 1-3 end to end
└── corpus/ , tokenizer_output/   # Generated at runtime (gitignored)
```

## 1. Setup

### Python backend (stages 1-3 + API)

```bash
cd bpe-tokenizer-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## 2. Run the pipeline

The defaults below match the brief: English Wikipedia, 100MB corpus, 32,000
vocabulary, min frequency 2, special tokens `<unk> <s> </s> <pad>`.

**One-shot (recommended):**

```bash
chmod +x scripts/run_pipeline.sh
LANG_CODE=en SIZE_LIMIT_MB=100 VOCAB_SIZE=32000 MIN_FREQUENCY=2 ./scripts/run_pipeline.sh
```

**Or stage by stage**, if you want to inspect intermediate outputs:

```bash
# Stage 1 — extraction. Two options, pick one:

# (a) Fast path: pre-cleaned HF dataset, good for iterating quickly
python data_extraction/extract_wikipedia_fast.py --lang en --size-limit-mb 100 --out-dir ./corpus

# (b) From-scratch path: raw XML dump, slower first run (downloads the full
#     compressed dump, several GB) but no dependency on a third-party snapshot
python data_extraction/extract_wikipedia.py --lang en --size-limit-mb 100 --out-dir ./corpus

# Stage 2 — train the tokenizer
python tokenizer_training/train_bpe.py \
    --corpus ./corpus/en_wikipedia_corpus.txt \
    --vocab-size 50257 --min-frequency 2 \
    --out-dir ./tokenizer_output

# Stage 3 — evaluate
python evaluation/evaluate.py \
    --tokenizer-path ./tokenizer_output/tokenizer.json \
    --benchmarks wikitext-103 ptb \
    --out ./tokenizer_output/evaluation_report.json
```

This produces:
- `corpus/en_wikipedia_corpus.txt` + `extraction_manifest.json`
- `tokenizer_output/tokenizer.json` + `training_config.json`
- `tokenizer_output/evaluation_report.json`

### Why two extraction scripts?

| | `extract_wikipedia.py` (XML dump) | `extract_wikipedia_fast.py` (HF dataset) |
|---|---|---|
| First-run cost | Downloads full compressed dump (several GB) before extracting anything | Streams only what's needed; stops at the size limit |
| Cleaning | You control every regex/markup-stripping rule via `mwparserfromhell` | Already cleaned by the HF dataset maintainers |
| Reproducibility | Exact dump date you choose | Whatever snapshot HF last published |
| Best for | "From scratch," full control, production pipelines | Fast iteration on hyperparameters |

Both produce an identical output format, so they're interchangeable inputs to
Stage 2.

## 3. Run the API + frontend

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000`. The frontend reads `NEXT_PUBLIC_API_URL` from
`frontend/.env.local` (defaults to `http://localhost:8000`).

## 4. Hyperparameters

All configurable via CLI flags on `train_bpe.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--vocab-size` | 32000 | Target vocabulary size (actual may land slightly below if the corpus doesn't have enough distinct merges) |
| `--min-frequency` | 2 | Minimum pair frequency required to perform a merge |
| `--special-tokens` | `<unk> <s> </s> <pad>` | Reserved tokens, always assigned the lowest ids |

## 5. Evaluation methodology

We benchmark on **WikiText-103** (clean long-form Wikipedia prose, disjoint
from training data) and **Penn Treebank** (older, smaller, different
register — 1989 financial newswire, with numbers and rare words already
replaced by `<unk>` — a genuine out-of-domain check, fetched as plain text
directly from GitHub rather than via the Hub, since the `ptb_text_only`
dataset repo there only ships a legacy script that current `datasets`
versions refuse to run). For each:

- **Tokens/word** — total subword tokens ÷ whitespace-split words. Expect roughly
  1.2–1.6 for a 32k vocab trained on similar-domain text, higher (worse) on PTB
  since its register differs from the Wikipedia training corpus.
- **Compression ratio (bytes/token)** — how many raw UTF-8 bytes one token
  represents on average. Directly relevant to context-window economics.
- **Encode/decode speed** — chars/sec throughput, measured end-to-end including
  Python-level overhead, not just the Rust core.

Sample table shape (`evaluate.py` prints this and saves it to JSON):

| Benchmark | Tokens/Word | Bytes/Token | Encode (chars/s) | Decode (chars/s) |
|---|---|---|---|---|
| wikitext-103 | 1.31 | 4.02 | ~600k | ~3.5M |
| ptb | 1.58 | 3.21 | ~550k | ~3.2M |

(Actual numbers depend on your corpus and vocab size — run `evaluate.py` to
generate real figures for your trained tokenizer.)

## 6. Frontend

Built with Next.js 16 (App Router) + TypeScript + Tailwind v4. Three views:

1. **Live token stream** — type text, see it tokenized in real time. Each
   token renders as a chip; width scales with how many characters it
   compressed, color cycles for visual separation between adjacent tokens.
   Hover a chip to see its token id and byte offsets.
2. **GPT-2 comparison** — sends the same input to both tokenizers via the
   `/compare` endpoint and shows token count delta + side-by-side chips.
3. **Benchmark table** — renders `evaluation_report.json` from Stage 3.




