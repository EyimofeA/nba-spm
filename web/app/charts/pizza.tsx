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

/** Exact feature families behind the public descriptive profile. */
export const SKILL_DEFINITIONS: Record<string, string> = {
  shooting: "Relative true shooting, shooting proficiency, and playtype-adjusted TS (zTS).",
  spacing: "Crafted spacing score.",
  creation: "Box creation, passing, assist/load, creation/load, and potential assists.",
  security: "Turnovers relative to load, including live-ball and bad-pass turnovers (inverted).",
  rim_pressure: "Free throws, rim frequency, fouls drawn, and points of contact drawn.",
  rebounding: "Offensive and defensive rebounds, rebound contests, and recovered blocks.",
  shot_defense: "Rim points saved, defender FG differentials, and scorer-adjusted matchup shot value.",
  disruption: "Steals, blocks, deflections, charges, and scorer-adjusted turnovers forced.",
  suppression: "Scorer-adjusted suppression of attempts, threes, assists, and shooting fouls.",
};

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

/** Overlay two season-relative skill profiles without changing the underlying scale. */
export function RadarComparison({
  left,
  right,
  leftName,
  rightName,
  size = 420,
}: {
  left: Slice[];
  right: Slice[];
  leftName: string;
  rightName: string;
  size?: number;
}) {
  const byKey = (slices: Slice[]) => new Map(slices.map((slice) => [slice.key, slice.value]));
  const leftValues = byKey(left);
  const rightValues = byKey(right);
  const skills = SKILLS.filter((skill) => leftValues.has(skill.key) && rightValues.has(skill.key));
  if (skills.length < 3) return <div className="empty">Both players need a skill profile for this season.</div>;
  const box = size + 92;
  const center = box / 2;
  const radius = size * 0.34;
  const point = (index: number, value: number) => polar(center, center, radius * Math.max(0, Math.min(100, value)) / 100, index * 360 / skills.length);
  const polygon = (values: Map<string, number>) => skills.map((skill, index) => { const p = point(index, values.get(skill.key) ?? 0); return `${p.x},${p.y}`; }).join(" ");
  // Keep the comparison inside the shared chart shell.  The shell owns the
  // theme-aware SVG label fills; without it, SVG falls back to black text.
  return <div className="chart"><svg viewBox={`0 0 ${box} ${box}`} role="img" aria-label={`Skill comparison: ${leftName} and ${rightName}`}>
    {[25, 50, 75, 100].map((ring) => <circle key={ring} className="grid-line" cx={center} cy={center} r={radius * ring / 100} fill="none" />)}
    {skills.map((skill, index) => { const edge = polar(center, center, radius, index * 360 / skills.length); const label = polar(center, center, radius + 24, index * 360 / skills.length); return <g key={skill.key}><line className="grid-line" x1={center} y1={center} x2={edge.x} y2={edge.y} /><text className="cat-label" x={label.x} y={label.y + 4} textAnchor={Math.abs(label.x - center) < 10 ? "middle" : label.x > center ? "start" : "end"}>{skill.label}</text></g>; })}
    <polygon points={polygon(leftValues)} fill="var(--series-1)" opacity="0.22" stroke="var(--series-1)" strokeWidth="2" />
    <polygon points={polygon(rightValues)} fill="var(--series-2)" opacity="0.2" stroke="var(--series-2)" strokeWidth="2" />
  </svg></div>;
}
