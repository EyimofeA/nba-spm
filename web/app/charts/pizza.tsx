"use client";

import { SERIES, ordinalSuffix } from "../lib/viz";
import { ChartBody, TipRow, Tooltip, useChartTip } from "./frame";

export type Slice = {
  key: string;
  label: string;
  group: SkillGroup;
  value: number;
};
export type SkillGroup = "offense" | "defense" | "shared";

/** Three groups, three categorical slots, assigned in fixed order. */
export const GROUP_COLOR: Record<SkillGroup, string> = {
  offense: SERIES[0],
  defense: SERIES[1],
  shared: SERIES[2],
};
export const GROUP_LABEL: Record<SkillGroup, string> = {
  offense: "Offense",
  defense: "Defense",
  shared: "Both ends",
};

/** Every skill the snapshot publishes, grouped and ordered around the ring. */
export const SKILLS: { key: string; label: string; group: SkillGroup }[] = [
  { key: "shooting", label: "Shooting", group: "offense" },
  { key: "spacing", label: "Spacing", group: "offense" },
  { key: "creation", label: "Creation", group: "offense" },
  { key: "security", label: "Security", group: "offense" },
  { key: "rim_pressure", label: "Rim pressure", group: "offense" },
  { key: "rebounding", label: "Rebounding", group: "shared" },
  { key: "shot_defense", label: "Shot defense", group: "defense" },
  { key: "disruption", label: "Disruption", group: "defense" },
  { key: "suppression", label: "Suppression", group: "defense" },
];

const polar = (cx: number, cy: number, radius: number, degrees: number) => {
  const radians = ((degrees - 90) * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(radians),
    y: cy + radius * Math.sin(radians),
  };
};

function wedgePath(
  cx: number,
  cy: number,
  radius: number,
  from: number,
  to: number,
) {
  const start = polar(cx, cy, radius, from);
  const end = polar(cx, cy, radius, to);
  const large = to - from > 180 ? 1 : 0;
  return `M${cx},${cy} L${start.x},${start.y} A${radius},${radius} 0 ${large} 1 ${end.x},${end.y} Z`;
}

/**
 * Season-relative percentile as a wedge ring. Radius carries the value and hue
 * carries the skill group. Only the strongest and weakest wedges are
 * direct-labelled — a number on every wedge goes unread — and the table view
 * carries the rest, which is also the relief for the light-mode slots that sit
 * under 3:1 on the surface.
 */
export function Pizza({
  slices,
  size = 400,
}: {
  slices: Slice[];
  size?: number;
}) {
  const { bodyRef, tip, bind } = useChartTip<Slice>();
  if (slices.length < 3)
    return <div className="empty">No skill profile for this season.</div>;

  // Category labels need horizontal room on both sides but only a line of it
  // above and below, so the frame is wider than it is tall.
  const boxW = size + 156;
  const boxH = size + 76;
  const cx = boxW / 2;
  const cy = boxH / 2;
  const radius = size / 2;
  const step = 360 / slices.length;
  const gap = 0.9; // the surface showing between wedges does the separating

  const ranked = [...slices].sort((a, b) => b.value - a.value);
  const flagged = new Set(
    [...ranked.slice(0, 2), ...ranked.slice(-2)].map((slice) => slice.key),
  );

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        viewBox={`0 0 ${boxW} ${boxH}`}
        role="img"
        aria-label="Skill percentile profile"
      >
        {[25, 50, 75, 100].map((ring) => (
          <circle
            key={ring}
            className="grid-line"
            cx={cx}
            cy={cy}
            r={(radius * ring) / 100}
            fill="none"
          />
        ))}

        {slices.map((slice, index) => {
          const from = index * step + gap;
          const to = (index + 1) * step - gap;
          const value = Math.max(0, Math.min(100, slice.value));
          const mid = (from + to) / 2;
          const labelPoint = polar(cx, cy, radius + 22, mid);
          const valuePoint = polar(cx, cy, (radius * value) / 100 + 15, mid);
          const anchor =
            Math.abs(labelPoint.x - cx) < 14
              ? "middle"
              : labelPoint.x > cx
                ? "start"
                : "end";
          return (
            <g key={slice.key}>
              <path
                d={wedgePath(cx, cy, (radius * value) / 100, from, to)}
                fill={GROUP_COLOR[slice.group]}
                opacity={tip && tip.datum.key !== slice.key ? 0.55 : 0.9}
              />
              <text
                className="cat-label"
                x={labelPoint.x}
                y={labelPoint.y + 4}
                textAnchor={anchor}
              >
                {slice.label}
              </text>
              {flagged.has(slice.key) && (
                <text
                  className="mark-value"
                  x={valuePoint.x}
                  y={valuePoint.y + 4}
                  textAnchor="middle"
                >
                  {Math.round(slice.value)}
                </text>
              )}
              {/* The wedge's own sector is the hit target, out to the full radius. */}
              <path
                className="hit"
                d={wedgePath(cx, cy, radius + 14, from - gap, to + gap)}
                tabIndex={0}
                aria-label={`${slice.label}: ${ordinalSuffix(slice.value)} percentile`}
                {...bind(slice)}
              />
            </g>
          );
        })}
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y} container={tip.width} width={162}>
          <div className="tt-head">{GROUP_LABEL[tip.datum.group]}</div>
          <div className="tt-title">{tip.datum.label}</div>
          <TipRow
            label="Percentile"
            value={ordinalSuffix(tip.datum.value)}
            color={GROUP_COLOR[tip.datum.group]}
          />
        </Tooltip>
      )}
    </ChartBody>
  );
}

export const pizzaLegend = (
  ["offense", "shared", "defense"] as SkillGroup[]
).map((group) => ({
  label: GROUP_LABEL[group],
  color: GROUP_COLOR[group],
}));
