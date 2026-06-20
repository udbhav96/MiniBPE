import type { Metadata } from "next";
import { Space_Grotesk, Instrument_Sans, DM_Mono } from "next/font/google";
import "./globals.css";

// Display face — used sparingly, for headings and the hero only.
const spaceGrotesk = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "700"],
});

// Body / UI face — everything else: labels, copy, buttons.
const instrumentSans = Instrument_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

// Utility / data face — token ids, byte offsets, counts, code.
const dmMono = DM_Mono({
  variable: "--font-data",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "BPE Lab — byte-pair tokenizer playground",
  description:
    "Train, inspect, and benchmark a byte-pair encoding tokenizer built from a Wikipedia corpus.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${instrumentSans.variable} ${dmMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-ink text-paper font-body">
        {children}
      </body>
    </html>
  );
}
