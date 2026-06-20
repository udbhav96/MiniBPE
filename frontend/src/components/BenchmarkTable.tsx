import type { EvaluationReport } from "@/lib/api";

/**
 * BenchmarkTable
 * --------------
 * Renders the Stage 3 evaluation_report.json as a readable comparison table:
 * one row per benchmark (WikiText-103, PTB, ...), columns for the metrics
 * that matter when judging a tokenizer — tokens/word, compression, and
 * encode/decode throughput.
 */

const METRIC_COLUMNS: {
  key: keyof EvaluationReport["benchmarks"][string];
  label: string;
  format: (v: number) => string;
}[] = [
  { key: "tokens_per_word", label: "Tokens / word", format: (v) => v.toFixed(3) },
  {
    key: "compression_ratio_bytes_per_token",
    label: "Bytes / token",
    format: (v) => v.toFixed(3),
  },
  {
    key: "encode_chars_per_sec",
    label: "Encode speed",
    format: (v) => `${(v / 1000).toFixed(0)}k chars/s`,
  },
  {
    key: "decode_chars_per_sec",
    label: "Decode speed",
    format: (v) => `${(v / 1000).toFixed(0)}k chars/s`,
  },
];

export function BenchmarkTable({ report }: { report: EvaluationReport | null }) {
  if (!report) {
    return (
      <div className="rounded-lg border border-dashed border-slate-dim p-6 text-sm text-slate">
        No evaluation report loaded yet. Run{" "}
        <code className="font-data text-cyan">evaluation/evaluate.py</code> to generate one.
      </div>
    );
  }

  const benchmarkNames = Object.keys(report.benchmarks);

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-dim">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-slate-dim bg-surface-raised">
            <th className="px-4 py-3 font-display font-medium text-paper">Benchmark</th>
            {METRIC_COLUMNS.map((col) => (
              <th key={col.key} className="px-4 py-3 font-data text-xs uppercase tracking-wide text-slate">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {benchmarkNames.map((name, i) => {
            const row = report.benchmarks[name];
            return (
              <tr
                key={name}
                className={`border-b border-slate-dim/60 ${i % 2 === 0 ? "bg-surface" : "bg-ink"}`}
              >
                <td className="px-4 py-3 font-medium text-paper">{name}</td>
                {METRIC_COLUMNS.map((col) => (
                  <td key={col.key} className="px-4 py-3 font-data text-cyan">
                    {col.format(row[col.key] as number)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
