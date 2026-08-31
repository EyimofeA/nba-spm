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

/** Every skill the snapshot publishes, grouped in the fingerprint. */
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

const labelLines = (label: string) => {
  const words = label.split(" ");
  return words.length > 1
    ? [words.slice(0, -1).join(" "), words.at(-1) ?? ""]
    : [label];
};

/**
 * A basketball scouting-card fingerprint. Each vertical lane is a
 * season-relative percentile: offense at one end of the court, shared skills
 * at centre court, and defense at the other. Direct labels keep every value
 * readable without a hover; the focusable lane adds the fuller tooltip.
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

  const boxW = Math.max(size + 80, 360);
  const boxH = Math.max(size + 64, 360);
  const inset = 28;
  const top = 30;
  const bandH = (boxH - top - 28) / 3;
  const centerX = boxW / 2;
  const courtMid = top + bandH * 1.5;
  const groups: SkillGroup[] = ["offense", "shared", "defense"];

  return (
    <ChartBody bodyRef={bodyRef}>
      <svg
        viewBox={`0 0 ${boxW} ${boxH}`}
        role="img"
        aria-label="Basketball skill percentile fingerprint"
      >
        <rect
          x={inset / 2}
          y={12}
          width={boxW - inset}
          height={boxH - 24}
          rx={10}
          fill="var(--surface-2)"
          stroke="var(--border)"
        />
        <line className="grid-line" x1={inset} x2={boxW - inset} y1={courtMid} y2={courtMid} />
        <circle className="grid-line" cx={centerX} cy={courtMid} r={20} fill="none" />
        <path
          d={`M${centerX - 42},${courtMid - 20}a42,42 0 0,0 84,0M${centerX - 42},${courtMid + 20}a42,42 0 0,1 84,0`}
          fill="none"
          stroke="var(--grid)"
          strokeWidth="1"
        />
        <text className="axis-title" x={inset} y={24}>SKILL FINGERPRINT · SEASON PERCENTILES</text>

        {groups.map((group, groupIndex) => {
          const skills = slices.filter((slice) => slice.group === group);
          if (!skills.length) return null;
          const y = top + groupIndex * bandH;
          const meterTop = y + 43;
          const meterH = Math.max(24, bandH - 82);
          const laneW = Math.min(86, (boxW - inset * 2) / skills.length);
          const startX = centerX - (laneW * skills.length) / 2;
          return (
            <g key={group}>
              <line
                x1={inset}
                x2={boxW - inset}
                y1={y + 30}
                y2={y + 30}
                stroke={GROUP_COLOR[group]}
                strokeOpacity="0.35"
              />
              <text
                x={inset}
                y={y + 20}
                fill={GROUP_COLOR[group]}
                fontFamily="var(--mono)"
                fontSize="10"
                fontWeight="600"
                letterSpacing="1.4"
              >
                {GROUP_LABEL[group].toUpperCase()}
              </text>
              {skills.map((slice, index) => {
                const value = Math.max(0, Math.min(100, slice.value));
                const x = startX + laneW * index + laneW / 2;
                const fillH = value === 0 ? 0 : Math.max(2, (meterH * value) / 100);
                const fillY = meterTop + meterH - fillH;
                const faded = Boolean(tip && tip.datum.key !== slice.key);
                return (
                  <g key={slice.key} opacity={faded ? 0.48 : 1}>
                    <rect
                      x={x - 12}
                      y={meterTop}
                      width={24}
                      height={meterH}
                      rx={12}
                      fill="var(--page)"
                      stroke="var(--grid)"
                    />
                    <rect
                      x={x - 9}
                      y={fillY}
                      width={18}
                      height={fillH}
                      rx={9}
                      fill={GROUP_COLOR[group]}
                    />
                    <circle cx={x} cy={fillY} r={5} fill="var(--surface-1)" stroke={GROUP_COLOR[group]} strokeWidth="2" />
                    <text className="mark-value" x={x} y={meterTop - 8} textAnchor="middle">
                      {Math.round(value)}
                    </text>
                    <text className="cat-label" x={x} y={meterTop + meterH + 17} textAnchor="middle">
                      {labelLines(slice.label).map((line, lineIndex) => (
                        <tspan key={lineIndex} x={x} dy={lineIndex ? 11 : 0}>{line}</tspan>
                      ))}
                    </text>
                    <rect
                      className="hit"
                      x={x - laneW / 2}
                      y={y + 31}
                      width={laneW}
                      height={bandH - 31}
                      rx={5}
                      tabIndex={0}
                      aria-label={`${slice.label}: ${ordinalSuffix(slice.value)} percentile`}
                      {...bind(slice)}
                    />
                  </g>
                );
              })}
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
  const skills = left.filter((skill) => rightValues.has(skill.key));
  if (skills.length < 3) return <div className="empty">Both players need a skill profile for this season.</div>;
  const box = size + 92;
  const center = box / 2;
  const radius = size * 0.34;
  const point = (index: number, value: number) => polar(center, center, radius * Math.max(0, Math.min(100, value)) / 100, index * 360 / skills.length);
  const polygon = (values: Map<string, number>) => skills.map((skill, index) => { const p = point(index, values.get(skill.key) ?? 0); return `${p.x},${p.y}`; }).join(" ");
  // Keep the comparison inside the shared chart shell.  The shell owns the
  // theme-aware SVG label fills; without it, SVG falls back to black text.
  return <div className="chart"><div className="radar-legend" aria-hidden="true"><span><i className="radar-key radar-key-left" />{leftName}</span><span><i className="radar-key radar-key-right" />{rightName}</span></div><svg viewBox={`0 0 ${box} ${box}`} role="img" aria-label={`Skill comparison: ${leftName} and ${rightName}`}>
    {[25, 50, 75, 100].map((ring) => <circle key={ring} className="grid-line" cx={center} cy={center} r={radius * ring / 100} fill="none" />)}
    {skills.map((skill, index) => { const edge = polar(center, center, radius, index * 360 / skills.length); const label = polar(center, center, radius + 24, index * 360 / skills.length); return <g key={skill.key}><line className="grid-line" x1={center} y1={center} x2={edge.x} y2={edge.y} /><text className="cat-label" x={label.x} y={label.y + 4} textAnchor={Math.abs(label.x - center) < 10 ? "middle" : label.x > center ? "start" : "end"}>{skill.label}</text></g>; })}
    <polygon points={polygon(leftValues)} fill="var(--series-1)" opacity="0.22" stroke="var(--series-1)" strokeWidth="2" />
    <polygon points={polygon(rightValues)} fill="var(--series-2)" opacity="0.2" stroke="var(--series-2)" strokeWidth="2" />
  </svg></div>;
}
