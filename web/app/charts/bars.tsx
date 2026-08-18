"use client";

import {
  COMPONENT_COLOR,
  fmtInt,
  fmtRating,
  linear,
  niceTicks,
  polarity,
  symmetricBound,
} from "../lib/viz";
import { ChartBody, TipRow, Tooltip, useChartTip } from "./frame";

export type ImpactDatum = {
  id: number;
  name: string;
  team: string | null;
  season: number;
  offense: number;
  defense: number;
  net: number;
  poss: number;
};

/**
 * Ranked offense/defense bars around a shared zero. Offense and defense are
 * grouped rather than stacked: either side can be negative, and a stack would
 * draw the second segment back over the first. Net is direct-labelled at the
 * row end; the per-side values live in the tooltip and the table view.
 */
export function ImpactBars({
  rows,
  onSelect,
  highlight,
}: {
  rows: ImpactDatum[];
  onSelect?: (datum: ImpactDatum) => void;
  highlight?: number;
}) {
  const { bodyRef, tip, bind } = useChartTip<ImpactDatum>();
  if (!rows.length) return <div className="empty">No rows for this slice.</div>;

  const width = 880;
  const rowHeight = 30;
  const barHeight = 9; // two 9px bars + a 2px surface gap sit inside the row
  const pad = { left: 172, right: 64, top: 14, bottom: 30 };
  const height = pad.top + rows.length * rowHeight + pad.bottom;

  // Anchored at zero but only as wide as the data: a forced symmetric domain
  // leaves half the plot empty whenever a leaderboard is all-positive.
  const values = rows.flatMap((row) => [row.offense, row.defense]);
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const margin = (high - low) * 0.03 || 0.5;
  const x = linear(
    [low - margin, high + margin],
    [pad.left, width - pad.right],
  );
  const zero = x(0);
  const ticks = niceTicks(low - margin, high + margin, 6);

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Offense and defense by player, points per 100 possessions"
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className={tick === 0 ? "zero-line" : "grid-line"}
              x1={x(tick)}
              x2={x(tick)}
              y1={pad.top}
              y2={height - pad.bottom}
            />
            <text
              className="tick"
              x={x(tick)}
              y={height - pad.bottom + 16}
              textAnchor="middle"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}

        {rows.map((row, index) => {
          const top = pad.top + index * rowHeight;
          const centre = top + rowHeight / 2;
          const active = highlight === row.id;
          const bars: [number, string][] = [
            [row.offense, COMPONENT_COLOR.offense],
            [row.defense, COMPONENT_COLOR.defense],
          ];
          return (
            <g
              key={`${row.id}-${row.season}`}
              className={highlight !== undefined && !active ? "dim" : undefined}
            >
              <text
                className="cat-label"
                x={pad.left - 12}
                y={centre + 4}
                textAnchor="end"
              >
                {row.name.length > 25 ? `${row.name.slice(0, 24)}…` : row.name}
              </text>
              {bars.map(([value, color], side) => {
                const barY = centre - barHeight - 1 + side * (barHeight + 2);
                const start = Math.min(zero, x(value));
                const size = Math.max(1, Math.abs(x(value) - zero));
                // 4px round on the data end, square where it meets the baseline.
                const radius = Math.min(4, size);
                const positive = value >= 0;
                return (
                  <path
                    key={side}
                    className="bar"
                    fill={color}
                    d={
                      positive
                        ? `M${start},${barY} h${size - radius} a${radius},${radius} 0 0 1 ${radius},${radius} v${barHeight - radius * 2} a${radius},${radius} 0 0 1 -${radius},${radius} h-${size - radius} z`
                        : `M${start + size},${barY} h-${size - radius} a${radius},${radius} 0 0 0 -${radius},${radius} v${barHeight - radius * 2} a${radius},${radius} 0 0 0 ${radius},${radius} h${size - radius} z`
                    }
                  />
                );
              })}
              <text
                className="mark-value"
                x={width - pad.right + 10}
                y={centre + 4}
                textAnchor="start"
              >
                {fmtRating(row.net)}
              </text>
              {/* The whole row is the hit target, so nothing needs a precise landing. */}
              <rect
                className="hit"
                x={0}
                y={top}
                width={width}
                height={rowHeight}
                tabIndex={0}
                role={onSelect ? "button" : undefined}
                aria-label={`${row.name}, net ${fmtRating(row.net)}`}
                onClick={() => onSelect?.(row)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect?.(row);
                  }
                }}
                {...bind(row)}
              />
            </g>
          );
        })}
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={176}>
          <div className="tt-head">
            {tip.datum.team ?? "—"} · {tip.datum.season}
          </div>
          <div className="tt-title">{tip.datum.name}</div>
          <TipRow
            label="Offense"
            value={fmtRating(tip.datum.offense)}
            color={COMPONENT_COLOR.offense}
          />
          <TipRow
            label="Defense"
            value={fmtRating(tip.datum.defense)}
            color={COMPONENT_COLOR.defense}
          />
          <TipRow label="Net" value={fmtRating(tip.datum.net)} />
          <div className="tt-note">{fmtInt(tip.datum.poss)} possessions</div>
        </Tooltip>
      )}
    </ChartBody>
  );
}

/** Shared legend items for every offense/defense bar chart. */
export const impactLegend = [
  { label: "Offense", color: COMPONENT_COLOR.offense },
  { label: "Defense", color: COMPONENT_COLOR.defense },
];

/* -------------------------------------------------------------------------- */

export type SwarmDatum = {
  id: number;
  name: string;
  value: number;
  team: string | null;
};

/**
 * Where every rated player sits on one axis, with one player picked out. Dots
 * carry the diverging scale (polarity around zero), so the colour restates the
 * position rather than adding a second variable; a scale legend accompanies it.
 */
export function Distribution({
  rows,
  highlight,
  label,
  onSelect,
}: {
  rows: SwarmDatum[];
  highlight?: number;
  label: string;
  onSelect?: (datum: SwarmDatum) => void;
}) {
  const { bodyRef, tip, bind } = useChartTip<SwarmDatum>();
  if (!rows.length)
    return <div className="empty">No distribution for this slice.</div>;

  const width = 880;
  const pad = { left: 16, right: 16, top: 16, bottom: 28 };
  const laneHeight = 96;
  const height = pad.top + laneHeight + pad.bottom;
  const bound = symmetricBound(
    rows.map((row) => row.value),
    2,
  );
  const x = linear([-bound, bound], [pad.left, width - pad.right]);
  const ticks = niceTicks(-bound, bound, 7);
  const centre = pad.top + laneHeight / 2;

  // Bucket by x, then fan each bucket out from the centre line: a cheap
  // beeswarm that keeps every dot visible without a force simulation.
  const radius = 3.4;
  const columns = new Map<number, number>();
  const placed = rows
    .slice()
    .sort((a, b) => a.value - b.value)
    .map((row) => {
      const px = x(row.value);
      const bucket = Math.round(px / (radius * 1.7));
      const depth = columns.get(bucket) ?? 0;
      columns.set(bucket, depth + 1);
      const offset =
        Math.ceil(depth / 2) * (radius * 1.85) * (depth % 2 === 0 ? 1 : -1);
      return { ...row, px, py: centre + offset };
    });
  const chosen = placed.find((row) => row.id === highlight);

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className={tick === 0 ? "zero-line" : "grid-line"}
              x1={x(tick)}
              x2={x(tick)}
              y1={pad.top}
              y2={pad.top + laneHeight}
            />
            <text
              className="tick"
              x={x(tick)}
              y={height - pad.bottom + 17}
              textAnchor="middle"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}
        {placed.map((row) => {
          const active = row.id === highlight;
          return (
            <circle
              key={row.id}
              className={active ? "dot" : "dot ghost"}
              cx={row.px}
              cy={row.py}
              r={active ? 6 : radius}
              fill={polarity(row.value, bound)}
            />
          );
        })}
        {chosen && (
          <>
            <line
              className="crosshair"
              x1={chosen.px}
              x2={chosen.px}
              y1={pad.top}
              y2={pad.top + laneHeight}
            />
            <text
              className="mark-value"
              x={chosen.px}
              y={pad.top - 3}
              textAnchor={
                chosen.px > width - 120
                  ? "end"
                  : chosen.px < 120
                    ? "start"
                    : "middle"
              }
            >
              {chosen.name} {fmtRating(chosen.value)}
            </text>
          </>
        )}
        {/* Nearest-point hit lanes: the pointer only has to be closest. */}
        {placed.map((row, index) => {
          const previous = placed[index - 1];
          const next = placed[index + 1];
          const left = previous ? (row.px + previous.px) / 2 : pad.left;
          const right = next ? (row.px + next.px) / 2 : width - pad.right;
          return (
            <rect
              key={`hit-${row.id}`}
              className="hit"
              x={left}
              y={pad.top}
              width={Math.max(1, right - left)}
              height={laneHeight}
              tabIndex={-1}
              onClick={() => onSelect?.(row)}
              {...bind(row)}
            />
          );
        })}
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={166}>
          <div className="tt-head">{tip.datum.team ?? "—"}</div>
          <div className="tt-title">{tip.datum.name}</div>
          <TipRow label={label} value={fmtRating(tip.datum.value)} />
        </Tooltip>
      )}
    </ChartBody>
  );
}
