/**
 * api.ts
 * ------
 * Thin client around the FastAPI backend (api/main.py). Centralizes the base
 * URL and response typing so components don't each hand-roll fetch calls.
 *
 * Set NEXT_PUBLIC_API_URL in .env.local to point at your running backend,
 * e.g. http://localhost:8000 for local dev or your deployed API origin.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface TokenizeResult {
  tokens: string[];
  ids: number[];
  offsets: [number, number][];
  num_tokens: number;
  num_chars: number;
}

export interface VocabInfo {
  vocab_size: number;
  special_tokens: string[];
}

export interface BenchmarkResult {
  num_documents: number;
  total_words: number;
  total_tokens: number;
  total_chars: number;
  total_bytes_utf8: number;
  tokens_per_word: number;
  compression_ratio_bytes_per_token: number;
  encode_seconds: number;
  decode_seconds: number;
  encode_chars_per_sec: number;
  decode_chars_per_sec: number;
}

export interface EvaluationReport {
  tokenizer_path: string;
  vocab_size: number;
  benchmarks: Record<string, BenchmarkResult>;
}

export interface CompareResult {
  input_text: string;
  ours: { tokens: string[]; ids: number[]; num_tokens: number };
  baseline_gpt2: { tokens: string[]; ids: number[]; num_tokens: number };
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function tokenize(text: string): Promise<TokenizeResult> {
  const res = await fetch(`${API_BASE}/tokenize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse<TokenizeResult>(res);
}

export async function detokenize(ids: number[]): Promise<{ text: string }> {
  const res = await fetch(`${API_BASE}/detokenize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  return handleResponse<{ text: string }>(res);
}

export async function getVocabInfo(): Promise<VocabInfo> {
  const res = await fetch(`${API_BASE}/vocab-size`);
  return handleResponse<VocabInfo>(res);
}

export async function getEvaluationReport(): Promise<EvaluationReport> {
  const res = await fetch(`${API_BASE}/evaluation-report`);
  return handleResponse<EvaluationReport>(res);
}

export async function compareWithBaseline(text: string): Promise<CompareResult> {
  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse<CompareResult>(res);
}
