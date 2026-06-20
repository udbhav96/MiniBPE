"use client";

import { useMemo, useState } from "react";
import type { TokenizeResult } from "@/lib/api";

/**
 * TokenStream
 * -----------
 * The signature visual of the page. Renders each token as a chip whose:
 *   - WIDTH encodes how many characters of the original text it compressed
 *     (a wider chip "bought back" more characters per token — exactly the
 *     economic intuition compression_ratio captures, made visible).
 *   - COLOR cycles through the Frost accent set (cyan / amber / slate-blue)
 *     by token id band, purely so adjacent tokens are visually distinguishable
 *     when several short ones sit side by side — not a meaningful encoding,
 *     just a way to "see the seams" between merges.
 * Hovering a chip reveals its token id and byte offsets in the data face (DM Mono).
 */

const CHIP_PALETTE = [
  { bg: "bg-cyan/15", border: "border-cyan/40", text: "text-cyan" },
  { bg: "bg-amber/15", border: "border-amber/40", text: "text-amber" },
  { bg: "bg-slate/20", border: "border-slate/50", text: "text-paper" },
];

function paletteFor(tokenId: number) {
  return CHIP_PALETTE[tokenId % CHIP_PALETTE.length];
}

export function TokenStream({ result }: { result: TokenizeResult | null }) {
  const [hovered, setHovered] = useState<number | null>(null);

  const chips = useMemo(() => {
    if (!result) return [];
    return result.tokens.map((tok, i) => {
      const [start, end] = result.offsets[i] ?? [0, 0];
      const charSpan = Math.max(1, end - start);
      // Map char span (typically 1-12) to a width scale so longer merges
      // visibly take up more horizontal space.
      const widthScale = Math.min(1 + charSpan * 0.18, 4);
      return {
        token: tok,
        id: result.ids[i],
        start,
        end,
        charSpan,
        widthScale,
      };
    });
  }, [result]);

  if (!result) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-slate-dim text-slate">
        <p className="font-body text-sm">Type something above to see it tokenized.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5 rounded-lg border border-slate-dim bg-surface p-4">
        {chips.map((chip, i) => {
          const palette = paletteFor(chip.id);
          const isHovered = hovered === i;
          return (
            <span
              key={i}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              style={{ minWidth: `${chip.widthScale * 0.55}rem` }}
              className={`relative flex cursor-default items-center justify-center rounded-md border px-2 py-1 font-data text-[13px] leading-tight transition-all ${palette.bg} ${palette.border} ${palette.text} ${
                isHovered ? "scale-105 shadow-lg shadow-cyan/10" : ""
              }`}
              title={`id ${chip.id} · chars [${chip.start}, ${chip.end})`}
            >
              {chip.token.replace(/Ġ/g, "·")}
              {isHovered && (
                <span className="absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded border border-slate-dim bg-surface-raised px-2 py-0.5 font-data text-[11px] text-slate">
                  id {chip.id} · {chip.charSpan} chars
                </span>
              )}
            </span>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 font-data text-xs text-slate">
        <span>
          <span className="text-paper">{result.num_tokens}</span> tokens
        </span>
        <span>
          <span className="text-paper">{result.num_chars}</span> chars
        </span>
        <span>
          <span className="text-paper">
            {(result.num_chars / Math.max(1, result.num_tokens)).toFixed(2)}
          </span>{" "}
          chars/token
        </span>
      </div>
    </div>
  );
}
