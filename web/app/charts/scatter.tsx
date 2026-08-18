"use client";

import { useRef } from "react";
import {
  SERIES,
  fmtInt,
  fmtRating,
  linear,
  localPoint,
  niceTicks,
  padExtent,
  polarity,
  symmetricBound,
} from "../lib/viz";
import { ChartBody, TipRow, Tooltip, useChartTip } from "./frame";

export type LandscapeDatum = {
  id: number;
  name: string;
  team: string | null;
  season: number;
  offense: number;
  defense: number;
  net: number;
  poss: number;
};

/** The diverging domain the plot uses, so a scale legend can match it. */
export const netBoundFor = (rows: { net: number }[]) =>
  symmetricBound(
    rows.map((row) => row.net),
    3,
  );

const QUADRANTS = [
  { label: "Two-way", at: "top-right" as const },
  { label: "Defense first", at: "top-left" as const },
  { label: "Offense first", at: "bottom-right" as const },
  { label: "Below average", at: "bottom-left" as const },
];

/**
 * Offense against defense, split at zero on both axes. Colour carries the
 * diverging net scale — a restatement of position on the diagonal rather than a
 * new variable, which is what makes the two-way corner readable at a glance —
 * and radius carries exposure, so small samples look small. One shared axis per
 * dimension; never a second y-scale.
 */
export function Landscape({
  rows,
  highlight,
  onSelect,
  labelTop = 6,
}: {
  rows: LandscapeDatum[];
  highlight?: number;
  onSelect?: (datum: LandscapeDatum) => void;
  labelTop?: number;
}) {
  const { bodyRef, tip, at, clear } = useChartTip<LandscapeDatum>();
  const svgRef = useRef<SVGSVGElement>(null);
  if (!rows.length)
    return <div className="empty">No player-seasons in this slice.</div>;

  const width = 880;
  const height = 520;
  const pad = { left: 62, right: 26, top: 24, bottom: 52 };

  const xBound = symmetricBound(
    rows.map((row) => row.offense),
    2,
  );
  const yBound = symmetricBound(
    rows.map((row) => row.defense),
    2,
  );
  const netBound = netBoundFor(rows);
  const x = linear([-xBound, xBound], [pad.left, width - pad.right]);
  const y = linear([-yBound, yBound], [height - pad.bottom, pad.top]);

  const possBound = Math.max(...rows.map((row) => row.poss), 1);
  const radius = (poss: number) =>
    3 + Math.sqrt(Math.max(0, poss) / possBound) * 5;

  const placed = rows.map((row) => ({
    ...row,
    px: x(row.offense),
    py: y(row.defense),
  }));
  const chosen = placed.find((row) => row.id === highlight);
  // Direct-label only the extremes; the rest live in the tooltip and table.
  const labelled = new Set(
    [...placed]
      .sort((a, b) => b.net - a.net)
      .slice(0, labelTop)
      .map((row) => row.id),
  );
  if (chosen) labelled.add(chosen.id);

  function nearest(event: React.PointerEvent<SVGRectElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const point = localPoint(event, svg, width, height);
    let best: (typeof placed)[number] | undefined;
    let bestDistance = Infinity;
    for (const row of placed) {
      const distance = (row.px - point.x) ** 2 + (row.py - point.y) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = row;
      }
    }
    if (best && bestDistance < 60 ** 2) at(best, event.clientX, event.clientY);
    else clear();
  }

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Offense against defense by player-season"
      >
        {niceTicks(-xBound, xBound, 7).map((tick) => (
          <g key={`x${tick}`}>
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
              y={height - pad.bottom + 17}
              textAnchor="middle"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}
        {niceTicks(-yBound, yBound, 6).map((tick) => (
          <g key={`y${tick}`}>
            <line
              className={tick === 0 ? "zero-line" : "grid-line"}
              x1={pad.left}
              x2={width - pad.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text
              className="tick"
              x={pad.left - 9}
              y={y(tick) + 4}
              textAnchor="end"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}

        {QUADRANTS.map((quadrant) => {
          const right = quadrant.at.endsWith("right");
          const top = quadrant.at.startsWith("top");
          return (
            <text
              key={quadrant.label}
              className="quadrant-label"
              x={right ? width - pad.right - 8 : pad.left + 8}
              y={top ? pad.top + 14 : height - pad.bottom - 8}
              textAnchor={right ? "end" : "start"}
            >
              {quadrant.label}
            </text>
          );
        })}

        {placed.map((row) => (
          <circle
            key={`${row.id}-${row.season}`}
            className={row.id === highlight ? "dot" : "dot ghost"}
            cx={row.px}
            cy={row.py}
            r={row.id === highlight ? radius(row.poss) + 3 : radius(row.poss)}
            fill={polarity(row.net, netBound)}
          />
        ))}

        {placed
          .filter((row) => labelled.has(row.id))
          .map((row) => (
            <text
              key={`label-${row.id}`}
              className="mark-label"
              x={row.px}
              y={row.py - radius(row.poss) - 6}
              textAnchor="middle"
            >
              {row.name.split(" ").slice(-1)[0]}
            </text>
          ))}

        {/* Axis titles sit outside the plot so they never collide with the
            quadrant labels, which own the four corners. */}
        <text
          className="axis-title"
          x={(pad.left + width - pad.right) / 2}
          y={height - 6}
          textAnchor="middle"
        >
          Offense →
        </text>
        <text
          className="axis-title"
          transform="rotate(-90)"
          x={-(pad.top + height - pad.bottom) / 2}
          y={13}
          textAnchor="middle"
        >
          Defense →
        </text>

        {/* Nearest-point layer: the pointer only has to be closest, not exact. */}
        <rect
          className="hit"
          x={pad.left}
          y={pad.top}
          width={width - pad.left - pad.right}
          height={height - pad.top - pad.bottom}
          onPointerMove={nearest}
          onPointerLeave={clear}
          onClick={() => tip && onSelect?.(tip.datum)}
        />
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={178}>
          <div className="tt-head">
            {tip.datum.team ?? "—"} · {tip.datum.season}
          </div>
          <div className="tt-title">{tip.datum.name}</div>
          <TipRow label="Offense" value={fmtRating(tip.datum.offense)} />
          <TipRow label="Defense" value={fmtRating(tip.datum.defense)} />
          <TipRow label="Net" value={fmtRating(tip.datum.net)} />
          <div className="tt-note">{fmtInt(tip.datum.poss)} possessions</div>
        </Tooltip>
      )}
    </ChartBody>
  );
}

/* -------------------------------------------------------------------------- */

export type RoleDatum = {
  id: number;
  name: string;
  team: string | null;
  x: number;
  y: number;
  role: string;
};

/**
 * The behavioural role map. Roles run to six categories, which is past the
 * three-slot cap for a scatter (any two dots can sit side by side), so this uses
 * emphasis instead of six hues: one role at a time is lifted into the accent and
 * the rest recede. Identity comes from the picker, never from colour alone.
 */
export function RoleMap({
  rows,
  emphasis,
  highlight,
  onSelect,
}: {
  rows: RoleDatum[];
  emphasis?: string;
  highlight?: number;
  onSelect?: (datum: RoleDatum) => void;
}) {
  const { bodyRef, tip, at, clear } = useChartTip<RoleDatum>();
  const svgRef = useRef<SVGSVGElement>(null);
  if (!rows.length)
    return <div className="empty">No role map for this season.</div>;

  const width = 880;
  const height = 500;
  const pad = 42;

  const [x0, x1] = padExtent([
    Math.min(...rows.map((r) => r.x)),
    Math.max(...rows.map((r) => r.x)),
  ]);
  const [y0, y1] = padExtent([
    Math.min(...rows.map((r) => r.y)),
    Math.max(...rows.map((r) => r.y)),
  ]);
  const x = linear([x0, x1], [pad, width - pad]);
  const y = linear([y0, y1], [height - pad, pad]);

  const placed = rows.map((row) => ({ ...row, px: x(row.x), py: y(row.y) }));
  const chosen = placed.find((row) => row.id === highlight);

  function nearest(event: React.PointerEvent<SVGRectElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const point = localPoint(event, svg, width, height);
    let best: (typeof placed)[number] | undefined;
    let bestDistance = Infinity;
    for (const row of placed) {
      const distance = (row.px - point.x) ** 2 + (row.py - point.y) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = row;
      }
    }
    if (best && bestDistance < 50 ** 2) at(best, event.clientX, event.clientY);
    else clear();
  }

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Player role cluster map"
      >
        <line
          className="grid-line"
          x1={width / 2}
          x2={width / 2}
          y1={pad}
          y2={height - pad}
        />
        <line
          className="grid-line"
          x1={pad}
          x2={width - pad}
          y1={height / 2}
          y2={height / 2}
        />

        {placed.map((row) => {
          const lit = !emphasis || row.role === emphasis;
          const isChosen = row.id === highlight;
          return (
            <circle
              key={row.id}
              className={isChosen ? "dot" : "dot ghost"}
              cx={row.px}
              cy={row.py}
              r={isChosen ? 8 : lit ? 4.5 : 3}
              fill={
                isChosen ? SERIES[1] : lit ? SERIES[0] : "var(--text-muted)"
              }
              opacity={isChosen ? 1 : lit ? 0.78 : 0.2}
            />
          );
        })}

        {chosen && (
          <text
            className="mark-label"
            x={chosen.px}
            y={chosen.py - 14}
            textAnchor="middle"
          >
            {chosen.name}
          </text>
        )}

        <text
          className="axis-title"
          x={width - pad}
          y={height - 12}
          textAnchor="end"
        >
          Role axis 1 →
        </text>
        <text className="axis-title" x={pad} y={pad - 14}>
          ↑ Role axis 2
        </text>

        <rect
          className="hit"
          x={pad}
          y={pad}
          width={width - pad * 2}
          height={height - pad * 2}
          onPointerMove={nearest}
          onPointerLeave={clear}
          onClick={() => tip && onSelect?.(tip.datum)}
        />
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={172}>
          <div className="tt-head">{tip.datum.team ?? "—"}</div>
          <div className="tt-title">{tip.datum.name}</div>
          <TipRow label="Role" value={tip.datum.role} />
        </Tooltip>
      )}
    </ChartBody>
  );
}
