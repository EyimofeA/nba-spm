"use client";

import { ReactNode, useCallback, useRef, useState } from "react";

/* ---------------------------------------------------------------- hover ---- */

type TipState<T> = { x: number; y: number; width: number; datum: T } | null;

/**
 * Hover and focus state for one chart, in pixels relative to the chart body.
 * Focus produces the same readout as the pointer, so the tooltip is reachable
 * from the keyboard and never becomes the only path to a value. The container
 * width rides along so the card can flip itself away from the right edge.
 */
export function useChartTip<T>() {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<TipState<T>>(null);

  const at = useCallback((datum: T, clientX: number, clientY: number) => {
    const box = bodyRef.current?.getBoundingClientRect();
    if (!box) return;
    setTip({
      datum,
      x: clientX - box.left,
      y: clientY - box.top,
      width: box.width,
    });
  }, []);

  const clear = useCallback(() => setTip(null), []);

  /** Spread onto a mark's hit target. */
  const bind = useCallback(
    (datum: T) => ({
      onPointerMove: (event: React.PointerEvent) =>
        at(datum, event.clientX, event.clientY),
      onPointerLeave: clear,
      onFocus: (event: React.FocusEvent<SVGElement>) => {
        const box = event.currentTarget.getBoundingClientRect();
        at(datum, box.left + box.width / 2, box.top + box.height / 2);
      },
      onBlur: clear,
    }),
    [at, clear],
  );

  return { bodyRef, tip, bind, at, clear };
}

export function Tooltip({
  x,
  y,
  container,
  width = 158,
  children,
}: {
  x: number;
  y: number;
  container: number;
  width?: number;
  children: ReactNode;
}) {
  // Flip to the left of the pointer near the right edge so the card never clips.
  const flip = x + width + 28 > container;
  return (
    <div
      className="tooltip"
      style={{
        width,
        left: 0,
        top: 0,
        transform: `translate(${x}px, ${y}px) translate(${flip ? `-${width + 14}px` : "14px"}, -50%)`,
      }}
      role="status"
    >
      {children}
    </div>
  );
}

/** One tooltip row: value leads, series name follows, keyed by a short stroke. */
export function TipRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="tt-row">
      {color && <i className="key" style={{ background: color }} />}
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

/* --------------------------------------------------------------- legend ---- */

export type LegendItem = { label: string; color: string; muted?: boolean };

/**
 * A legend is always present for two or more series. `shape` mirrors the mark:
 * a rect for bars and areas, a short stroke for lines.
 */
export function Legend({
  items,
  shape = "swatch",
  onToggle,
  hidden,
}: {
  items: LegendItem[];
  shape?: "swatch" | "key";
  onToggle?: (label: string) => void;
  hidden?: Set<string>;
}) {
  if (items.length < 2) return null;
  return (
    <div className="legend">
      {items.map((item) =>
        onToggle ? (
          <button
            key={item.label}
            type="button"
            aria-pressed={!hidden?.has(item.label)}
            onClick={() => onToggle(item.label)}
          >
            <i className={shape} style={{ background: item.color }} />
            {item.label}
          </button>
        ) : (
          <span className="item" key={item.label}>
            <i className={shape} style={{ background: item.color }} />
            {item.label}
          </span>
        ),
      )}
    </div>
  );
}

/** Continuous legend for the diverging scale — the scatter's color channel. */
export function ScaleLegend({
  low,
  high,
  caption,
}: {
  low: string;
  high: string;
  caption?: string;
}) {
  return (
    <div className="scale-legend">
      {caption && <span>{caption}</span>}
      <span>{low}</span>
      <i className="ramp" aria-hidden="true" />
      <span>{high}</span>
    </div>
  );
}

/* --------------------------------------------------------------- figure ---- */

/**
 * A chart card. Every chart ships a table view alongside it: that is the relief
 * channel for the light-mode slots that sit under 3:1 on the surface, and it
 * keeps every value reachable without hovering.
 */
export function Figure({
  kicker,
  title,
  note,
  controls,
  legend,
  table,
  tableLabel = "Table",
  defaultView = "chart",
  children,
}: {
  kicker?: string;
  title: string;
  note?: ReactNode;
  controls?: ReactNode;
  legend?: ReactNode;
  table?: ReactNode;
  tableLabel?: string;
  defaultView?: "chart" | "table";
  children: ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table">(defaultView);
  return (
    <section className="card">
      <div className="card-head">
        <div>
          {kicker && <p className="kicker">{kicker}</p>}
          <h2>{title}</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {controls}
          {table && (
            <div
              className="segmented"
              role="group"
              aria-label={`${title} view`}
            >
              <button
                type="button"
                aria-pressed={view === "chart"}
                onClick={() => setView("chart")}
              >
                Chart
              </button>
              <button
                type="button"
                aria-pressed={view === "table"}
                onClick={() => setView("table")}
              >
                {tableLabel}
              </button>
            </div>
          )}
        </div>
      </div>
      {view === "chart" ? (
        <>
          {legend && <div style={{ marginTop: 12 }}>{legend}</div>}
          <div style={{ marginTop: legend ? 0 : 14 }}>{children}</div>
        </>
      ) : (
        <div style={{ marginTop: 14, maxHeight: 420, overflow: "auto" }}>
          {table}
        </div>
      )}
      {note && <p className="note">{note}</p>}
    </section>
  );
}

/** Wraps a chart body so absolutely-positioned tooltips anchor correctly. */
export function ChartBody({
  bodyRef,
  children,
}: {
  bodyRef: React.RefObject<HTMLDivElement | null>;
  children: ReactNode;
}) {
  return (
    <div className="chart" ref={bodyRef}>
      {children}
    </div>
  );
}
