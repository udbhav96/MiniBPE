"""
main.py
========
FastAPI server exposing the trained BPE tokenizer to the frontend.

ENDPOINTS
---------
GET  /health                 -> liveness check
GET  /vocab-size              -> { vocab_size, special_tokens }
POST /tokenize                 -> { tokens: [...], ids: [...], byte_spans: [...] }
POST /detokenize                -> { text: "..." }
GET  /evaluation-report          -> contents of evaluation_report.json (Stage 3 output)
POST /compare                     -> tokenize the same text with our BPE tokenizer
                                      AND a baseline (GPT-2) tokenizer, for the
                                      frontend's side-by-side comparison view

RUNNING
-------
    pip install fastapi uvicorn tokenizers transformers
    uvicorn main:app --reload --port 8000

CORS is left open for localhost dev; tighten allow_origins before deploying.
"""

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

# --- Configuration: paths to pipeline outputs -------------------------------
TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer_output" / "tokenizer.json"
EVAL_REPORT_PATH = Path(__file__).parent.parent / "tokenizer_output" / "evaluation_report.json"
TRAINING_CONFIG_PATH = Path(__file__).parent.parent / "tokenizer_output" / "training_config.json"

app = FastAPI(title="BPE Tokenizer Playground API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your deployed frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load our trained tokenizer once at startup -----------------------------
if not TOKENIZER_PATH.exists():
    raise RuntimeError(
        f"No trained tokenizer found at {TOKENIZER_PATH}. "
        f"Run tokenizer_training/train_bpe.py first."
    )
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# --- Lazily-loaded baseline tokenizer (GPT-2) for comparison ---------------
_baseline_tokenizer = None


def get_baseline_tokenizer():
    """Loads the GPT-2 tokenizer on first use only, so a fresh API instance
    doesn't pay the download/load cost unless the /compare endpoint is hit."""
    global _baseline_tokenizer
    if _baseline_tokenizer is None:
        from transformers import GPT2TokenizerFast
        _baseline_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return _baseline_tokenizer


class TokenizeRequest(BaseModel):
    text: str


class DetokenizeRequest(BaseModel):
    ids: List[int]


class CompareRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vocab-size")
def vocab_size():
    config = {}
    if TRAINING_CONFIG_PATH.exists():
        config = json.loads(TRAINING_CONFIG_PATH.read_text())
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": config.get("special_tokens", []),
    }


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    encoding = tokenizer.encode(req.text)
    return {
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        # offsets let the frontend highlight exactly which characters in the
        # original string map to each token, for the visual token explorer
        "offsets": encoding.offsets,
        "num_tokens": len(encoding.ids),
        "num_chars": len(req.text),
    }


@app.post("/detokenize")
def detokenize(req: DetokenizeRequest):
    text = tokenizer.decode(req.ids)
    return {"text": text}


@app.get("/evaluation-report")
def evaluation_report():
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run evaluation/evaluate.py first.",
        )
    return json.loads(EVAL_REPORT_PATH.read_text())


@app.post("/compare")
def compare(req: CompareRequest):
    """Tokenize the same input text with both our trained BPE tokenizer and
    the GPT-2 baseline, so the frontend can render a side-by-side diff."""
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    ours = tokenizer.encode(req.text)

    baseline = get_baseline_tokenizer()
    baseline_ids = baseline.encode(req.text)
    baseline_tokens = baseline.convert_ids_to_tokens(baseline_ids)

    return {
        "input_text": req.text,
        "ours": {
            "tokens": ours.tokens,
            "ids": ours.ids,
            "num_tokens": len(ours.ids),
        },
        "baseline_gpt2": {
            "tokens": baseline_tokens,
            "ids": baseline_ids,
            "num_tokens": len(baseline_ids),
        },
    }
