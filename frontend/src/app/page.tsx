"use client";

import { useEffect, useState } from "react";
import {
  tokenize,
  compareWithBaseline,
  getEvaluationReport,
  getVocabInfo,
  type TokenizeResult,
  type CompareResult,
  type EvaluationReport,
  type VocabInfo,
} from "@/lib/api";
import { TokenStream } from "@/components/TokenStream";
import { BenchmarkTable } from "@/components/BenchmarkTable";
import { CompareView } from "@/components/CompareView";

const SAMPLE_TEXT =
  "Byte-pair encoding builds its vocabulary by repeatedly merging the most frequent pair of symbols in a corpus.";

export default function Home() {
  const [input, setInput] = useState(SAMPLE_TEXT);
  const [result, setResult] = useState<TokenizeResult | null>(null);
  const [vocabInfo, setVocabInfo] = useState<VocabInfo | null>(null);
  const [evalReport, setEvalReport] = useState<EvaluationReport | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load vocab info + evaluation report once on mount.
  useEffect(() => {
    getVocabInfo().then(setVocabInfo).catch(() => setVocabInfo(null));
    getEvaluationReport().then(setEvalReport).catch(() => setEvalReport(null));
  }, []);

  // Debounced live tokenization as the user types.
  useEffect(() => {
    if (!input.trim()) {
      setResult(null);
      return;
    }
    const handle = setTimeout(() => {
      tokenize(input)
        .then((r) => {
          setResult(r);
          setError(null);
        })
        .catch((e) => setError(e.message));
    }, 250);
    return () => clearTimeout(handle);
  }, [input]);

  async function handleCompare() {
    if (!input.trim()) return;
    setCompareLoading(true);
    try {
      const r = await compareWithBaseline(input);
      setCompareResult(r);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCompareLoading(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-12 md:py-16">
      {/* ---- Hero ---- */}
      <header className="mb-10 space-y-3">
        <p className="font-data text-xs uppercase tracking-[0.2em] text-cyan">
          BPE Lab
        </p>
        <h1 className="font-display text-4xl font-medium leading-tight text-paper md:text-5xl">
          Watch your text become tokens.
        </h1>
        <p className="max-w-xl font-body text-slate">
          A byte-pair encoding tokenizer trained from scratch on a Wikipedia corpus.
          Type below to see exactly how it splits your text — then compare it against GPT-2.
        </p>
        {vocabInfo && (
          <p className="font-data text-xs text-slate">
            vocab size: <span className="text-paper">{vocabInfo.vocab_size.toLocaleString()}</span>
            {"  ·  "}
            special tokens: <span className="text-paper">{vocabInfo.special_tokens.join(", ")}</span>
          </p>
        )}
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}. Is the API running at the configured NEXT_PUBLIC_API_URL?
        </div>
      )}

      {/* ---- Live token input ---- */}
      <section className="mb-12 space-y-3">
        <label htmlFor="text-input" className="font-display text-sm font-medium text-paper">
          Your text
        </label>
        <textarea
          id="text-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={4}
          placeholder="Type or paste anything..."
          className="w-full resize-none rounded-lg border border-slate-dim bg-surface px-4 py-3 font-body text-paper placeholder:text-slate focus:border-cyan focus:outline-none"
        />
        <TokenStream result={result} />
      </section>

      {/* ---- Comparison with GPT-2 ---- */}
      <section className="mb-12 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-xl font-medium text-paper">Compare with GPT-2</h2>
          <button
            onClick={handleCompare}
            disabled={compareLoading || !input.trim()}
            className="rounded-md bg-cyan px-4 py-2 font-body text-sm font-medium text-ink transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {compareLoading ? "Comparing…" : "Run comparison"}
          </button>
        </div>
        <CompareView result={compareResult} />
      </section>

      {/* ---- Evaluation benchmarks ---- */}
      <section className="space-y-4">
        <h2 className="font-display text-xl font-medium text-paper">Benchmark results</h2>
        <p className="font-body text-sm text-slate">
          Tokens per word, byte compression, and throughput measured against standard
          language-modeling benchmarks.
        </p>
        <BenchmarkTable report={evalReport} />
      </section>

      <footer className="mt-16 border-t border-slate-dim pt-6 font-data text-xs text-slate">
        Trained on a Wikipedia corpus (CC BY-SA 4.0). Tokenizer: byte-level BPE via
        Hugging Face <code className="text-paper">tokenizers</code>.
      </footer>
    </main>
  );
}
