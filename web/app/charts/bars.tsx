"use client";

import {
  COMPONENT_COLOR,
  fmtInt,
  fmtRating,
  linear,
  niceTicks,
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
