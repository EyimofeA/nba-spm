"use client";

import { useMemo } from "react";
import { fmtRating, linear, niceTicks, padExtent } from "../lib/viz";
import { ChartBody, TipRow, Tooltip, useChartTip } from "./frame";

export type LinePoint = { x: number; y: number };
export type LineSeries = { label: string; color: string; points: LinePoint[] };

type Snapped = {
  x: number;
  rows: { label: string; color: string; y: number }[];
};

/**
 * Trend over time for one to four series. A vertical crosshair snaps to the
 * nearest x and one tooltip reports every series there, so the pointer never
 * has to land on a stroke. End labels are dropped when they would collide —
 * nudging them apart detaches a label from its line, and the legend already
 * carries identity.
 */
export function MultiLine({
  series,
  height = 260,
  xFormat = String,
  yFormat = fmtRating,
  xTitle,
  area = false,
  markEvery = 1,
  selected,
  onSelectX,
  zeroBaseline = true,
}: {
  series: LineSeries[];
  height?: number;
  xFormat?: (value: number) => string;
  yFormat?: (value: number) => string;
  xTitle?: string;
  area?: boolean;
  markEvery?: number;
  selected?: number;
  onSelectX?: (x: number) => void;
  zeroBaseline?: boolean;
}) {
  const { bodyRef, tip, bind } = useChartTip<Snapped>();

  const live = series.filter((item) => item.points.length > 0);
  const xs = useMemo(
    () =>
      [...new Set(live.flatMap((s) => s.points.map((p) => p.x)))].sort(
        (a, b) => a - b,
      ),
    [live],
  );

  if (!live.length || !xs.length)
    return <div className="empty">No series to plot.</div>;

  const width = 880;
  const labelRoom = live.length > 1 ? 74 : 52;
  // The bottom band holds the tick row and, below it, the axis title.
  const pad = { left: 46, right: labelRoom, top: 20, bottom: xTitle ? 48 : 32 };

  const values = live.flatMap((s) => s.points.map((p) => p.y));
  const rawLow = zeroBaseline ? Math.min(0, ...values) : Math.min(...values);
  const rawHigh = zeroBaseline ? Math.max(0, ...values) : Math.max(...values);
  const [low, high] = padExtent([rawLow, rawHigh], 0.12);
  const yTicks = niceTicks(low, high, 5);

  const x = linear([xs[0], xs[xs.length - 1]], [pad.left, width - pad.right]);
  const y = linear([low, high], [height - pad.bottom, pad.top]);
  // A single x would collapse the scale; centre it instead.
  const px = (value: number) =>
    xs.length === 1 ? (pad.left + width - pad.right) / 2 : x(value);

  const ends = live
    .map((s) => {
      const last = s.points[s.points.length - 1];
      return { label: s.label, color: s.color, x: px(last.x), y: y(last.y) };
    })
    .sort((a, b) => a.y - b.y);
  const endsCollide = ends.some(
    (end, index) => index > 0 && Math.abs(end.y - ends[index - 1].y) < 12,
  );
  const showEndLabels = live.length > 1 && !endsCollide;

  const snapped: Snapped[] = xs.map((value) => ({
    x: value,
    rows: live.flatMap((s) => {
      const found = s.points.find((p) => p.x === value);
      return found ? [{ label: s.label, color: s.color, y: found.y }] : [];
    }),
  }));

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={live.map((s) => s.label).join(", ")}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
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

        {area && live.length === 1 && (
          <path
            className="series-area"
            fill={live[0].color}
            d={`M${px(live[0].points[0].x)},${y(0)} ${live[0].points
              .map((p) => `L${px(p.x)},${y(p.y)}`)
              .join(
                " ",
              )} L${px(live[0].points[live[0].points.length - 1].x)},${y(0)} Z`}
          />
        )}

        {live.map((s) => (
          <g key={s.label}>
            <path
              className="series-line"
              stroke={s.color}
              d={s.points
                .map((p, index) => `${index ? "L" : "M"}${px(p.x)},${y(p.y)}`)
                .join(" ")}
            />
            {s.points
              .filter(
                (_, index) =>
                  index % markEvery === 0 || index === s.points.length - 1,
              )
              .map((p) => (
                <circle
                  key={p.x}
                  className="dot"
                  cx={px(p.x)}
                  cy={y(p.y)}
                  r={selected === p.x ? 5.5 : 4}
                  fill={s.color}
                />
              ))}
          </g>
        ))}

        {showEndLabels &&
          ends.map((end) => (
            <text
              key={end.label}
              className="mark-label"
              x={end.x + 10}
              y={end.y + 4}
            >
              {end.label}
            </text>
          ))}
        {live.length === 1 && (
          <text className="mark-value" x={ends[0].x + 10} y={ends[0].y + 4}>
            {yFormat(live[0].points[live[0].points.length - 1].y)}
          </text>
        )}

        {xs.map((value) => (
          <text
            key={value}
            className="tick"
            x={px(value)}
            y={height - pad.bottom + 18}
            textAnchor="middle"
            style={
              selected === value
                ? { fill: "var(--text-primary)", fontWeight: 600 }
                : undefined
            }
          >
            {xFormat(value)}
          </text>
        ))}
        {/* Centred below the tick row so it never collides with the last tick. */}
        {xTitle && (
          <text
            className="axis-title"
            x={(pad.left + width - pad.right) / 2}
            y={height - 6}
            textAnchor="middle"
          >
            {xTitle}
          </text>
        )}

        {tip && (
          <line
            className="crosshair"
            x1={px(tip.datum.x)}
            x2={px(tip.datum.x)}
            y1={pad.top}
            y2={height - pad.bottom}
          />
        )}

        {/* One hit band per x — readers aim at a season, never at a 2px line. */}
        {snapped.map((snap, index) => {
          const previous = snapped[index - 1];
          const next = snapped[index + 1];
          const left = previous ? (px(snap.x) + px(previous.x)) / 2 : pad.left;
          const right = next
            ? (px(snap.x) + px(next.x)) / 2
            : width - pad.right;
          return (
            <rect
              key={snap.x}
              className="hit"
              x={left}
              y={pad.top}
              width={Math.max(1, right - left)}
              height={height - pad.bottom - pad.top}
              tabIndex={0}
              role={onSelectX ? "button" : undefined}
              aria-label={`${xFormat(snap.x)}: ${snap.rows.map((r) => `${r.label} ${yFormat(r.y)}`).join(", ")}`}
              onClick={() => onSelectX?.(snap.x)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelectX?.(snap.x);
                }
              }}
              {...bind(snap)}
            />
          );
        })}
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={168}>
          <div className="tt-head">{xTitle ?? "At"}</div>
          <div className="tt-title">{xFormat(tip.datum.x)}</div>
          {tip.datum.rows.map((row) => (
            <TipRow
              key={row.label}
              label={row.label}
              value={yFormat(row.y)}
              color={row.color}
            />
          ))}
        </Tooltip>
      )}
    </ChartBody>
  );
}
