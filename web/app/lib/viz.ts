/**
 * Chart primitives: palette roles, scales, formatting, hover state.
 *
 * Colors are referenced as CSS custom properties rather than literal hex so a
 * single token swap re-themes light and dark. Continuous ramps are built with
 * `color-mix()` on those same tokens, which keeps the dark steps correct
 * without a second JS palette.
 */

import { useCallback, useState } from "react";

/** Categorical slots in fixed order. Assign in sequence; never cycle. */
export const SERIES = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
] as const;

/** The three impact components read as distinct series (slots 1–3). */
export const COMPONENT_COLOR = {
  net: SERIES[0],
  offense: SERIES[1],
  defense: SERIES[2],
} as const;

/** The three models read as distinct series (slots 1–3). */
export const MODEL_COLOR = {
  aio: SERIES[0],
  spm: SERIES[1],
  rapm: SERIES[2],
} as const;

/**
 * Diverging fill for a value's polarity around zero. `t` is the value divided
 * by the scale bound; the midpoint resolves to neutral gray so "about zero"
 * reads as nothing rather than as a weak hue.
 */
export function polarity(value: number, bound: number) {
  const t = Math.max(-1, Math.min(1, bound > 0 ? value / bound : 0));
  const weight = Math.round(Math.abs(t) * 100);
  return `color-mix(in oklab, ${t >= 0 ? "var(--pos)" : "var(--neg)"} ${weight}%, var(--mid))`;
}

export const polarityToken = (value: number) =>
  value >= 0 ? "var(--pos)" : "var(--neg)";

/** Sequential blue for magnitude, light -> dark. `t` in [0,1]. */
export function magnitude(t: number) {
  const weight = Math.round(Math.max(0, Math.min(1, t)) * 100);
  return `color-mix(in oklab, var(--seq-700) ${weight}%, var(--seq-100))`;
}

/* ------------------------------------------------------------- formatting - */

export const fmtRating = (value: number | undefined | null) =>
  value === undefined || value === null || Number.isNaN(value)
    ? "—"
    : `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;

export const fmtDelta = (value: number | undefined) =>
  value === undefined || Number.isNaN(value)
    ? "—"
    : `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;

export const fmtInt = (value: number | undefined | null) =>
  value === undefined || value === null
    ? "—"
    : Math.round(value).toLocaleString("en-US");

export const fmtCompact = (value: number) =>
  Math.abs(value) >= 1000
    ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}K`
    : String(Math.round(value));

export const fmtPct = (value: number | null | undefined) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : `${Math.round(value)}`;

export const ordinalSuffix = (value: number) => {
  const rounded = Math.round(value);
  const mod100 = rounded % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${rounded}th`;
  return `${rounded}${["th", "st", "nd", "rd"][rounded % 10] ?? "th"}`;
};

/* ----------------------------------------------------------------- scales - */

export type Scale = (value: number) => number;

export function linear(
  domain: [number, number],
  range: [number, number],
): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** Round a raw extent out to clean tick values. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    const base = Number.isFinite(min) ? min : 0;
    return [base - 1, base, base + 1];
  }
  const raw = (max - min) / Math.max(1, count - 1);
  const magnitudeStep = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / magnitudeStep;
  const step =
    magnitudeStep *
    (normalized >= 5 ? 5 : normalized >= 2.5 ? 2.5 : normalized >= 2 ? 2 : 1);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 1e-6; value += step) {
    ticks.push(Math.abs(value) < step * 1e-6 ? 0 : Number(value.toFixed(6)));
  }
  return ticks;
}

/** A symmetric bound around zero, rounded out — the right frame for polarity. */
export function symmetricBound(values: number[], minimum = 1) {
  const peak = values.reduce(
    (most, value) => Math.max(most, Math.abs(value)),
    0,
  );
  return Math.max(minimum, Math.ceil(peak * 2) / 2);
}

export function extent(values: number[]): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  return Number.isFinite(low) ? [low, high] : [0, 1];
}

/** Pad an extent by a fraction of its span so marks never touch the frame. */
export function padExtent(
  [low, high]: [number, number],
  fraction = 0.06,
): [number, number] {
  const span = high - low || 1;
  return [low - span * fraction, high + span * fraction];
}

/* ---------------------------------------------------------------- tooltip - */

export type TooltipState<T> = { x: number; y: number; datum: T } | null;

/**
 * Hover/focus state for a chart. The tooltip only ever enhances: every value it
 * shows is also reachable from a direct label or the table view, and focus
 * produces the same readout as the pointer.
 */
export function useTooltip<T>() {
  const [tip, setTip] = useState<TooltipState<T>>(null);
  const show = useCallback(
    (datum: T, x: number, y: number) => setTip({ datum, x, y }),
    [],
  );
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

/** Clamp a tooltip inside its container so it never runs off the card. */
export function tipStyle(x: number, y: number, width: number, box = 150) {
  const flip = x + box + 18 > width;
  return {
    left: `${flip ? x - box - 12 : x + 12}px`,
    top: `${Math.max(0, y - 12)}px`,
    width: `${box}px`,
  } as const;
}

/** Convert a pointer event to chart-local coordinates in viewBox units. */
export function localPoint(
  event: { clientX: number; clientY: number },
  target: SVGSVGElement,
  vw: number,
  vh: number,
) {
  const box = target.getBoundingClientRect();
  return {
    x: ((event.clientX - box.left) / box.width) * vw,
    y: ((event.clientY - box.top) / box.height) * vh,
  };
}
