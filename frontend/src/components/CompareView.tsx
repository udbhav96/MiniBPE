"use client";

import type { CompareResult } from "@/lib/api";

/**
 * CompareView
 * -----------
 * Side-by-side comparison of our trained tokenizer vs. the GPT-2 baseline
 * on the same input text. Token count delta is the headline number — it's
 * the most legible single signal for "did our domain-specific vocabulary
 * actually help."
 */

function TokenRow({ tokens }: { tokens: string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {tokens.map((t, i) => (
        <span
          key={i}
          className="rounded border border-slate-dim bg-surface px-1.5 py-0.5 font-data text-xs text-paper"
        >
          {t.replace(/Ġ/g, "·").replace(/^Ġ/, "·")}
        </span>
      ))}
    </div>
  );
}

export function CompareView({ result }: { result: CompareResult | null }) {
  if (!result) {
    return (
      <div className="rounded-lg border border-dashed border-slate-dim p-6 text-sm text-slate">
        Enter text above and run a comparison to see how this tokenizer stacks up against GPT-2.
      </div>
    );
  }

  const { ours, baseline_gpt2 } = result;
  const delta = ours.num_tokens - baseline_gpt2.num_tokens;
  const deltaLabel =
    delta === 0
      ? "Identical token count"
      : delta < 0
        ? `${Math.abs(delta)} fewer tokens than GPT-2`
        : `${delta} more tokens than GPT-2`;

  return (
    <div className="space-y-4">
      <div className="flex items-baseline gap-3">
        <span className="font-display text-2xl font-medium text-paper">{deltaLabel}</span>
        <span
          className={`font-data text-sm ${delta <= 0 ? "text-cyan" : "text-amber"}`}
        >
          ({ours.num_tokens} vs {baseline_gpt2.num_tokens})
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2 rounded-lg border border-cyan/30 bg-surface p-4">
          <div className="flex items-center justify-between">
            <h4 className="font-display text-sm font-medium text-cyan">Our BPE tokenizer</h4>
            <span className="font-data text-xs text-slate">{ours.num_tokens} tokens</span>
          </div>
          <TokenRow tokens={ours.tokens} />
        </div>

        <div className="space-y-2 rounded-lg border border-slate-dim bg-surface p-4">
          <div className="flex items-center justify-between">
            <h4 className="font-display text-sm font-medium text-paper">GPT-2 baseline</h4>
            <span className="font-data text-xs text-slate">{baseline_gpt2.num_tokens} tokens</span>
          </div>
          <TokenRow tokens={baseline_gpt2.tokens} />
        </div>
      </div>
    </div>
  );
}
