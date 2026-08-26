"use client";

import { useMemo, useState } from "react";
import { Figure, Legend } from "../charts/frame";
import { Pizza, RadarComparison, Slice } from "../charts/pizza";
import {
  LocalGameSkillRow,
  LocalPlayerSkills,
  LocalSkillDefinition,
  LocalSkillIndex,
  LocalSkillRow,
} from "../lib/data";

type Mode = "career" | "season";
type ValueMode = "estimate" | "raw";

const PROFILE_LABELS: Record<string, string> = {
  rim_finishing: "Rim finishing",
  three_point_shooting: "Three-point shooting",
  free_throw_shooting: "Free throws",
  shotmaking: "Shotmaking",
  shooting_context: "Shooting context",
  creation: "Creation",
  passing: "Passing",
  ball_security: "Ball security",
  rebounding: "Rebounding",
  rim_defense: "Rim defense",
  perimeter_defense: "Perimeter defense",
  disruption: "Disruption",
};

function value(row: LocalSkillRow, mode: ValueMode) {
  return mode === "estimate" ? row[1] : row[2];
}

function format(value: number | null, definition: LocalSkillDefinition) {
  if (value === null) return "—";
  if (definition.unit === "percent") return `${value.toFixed(1)}%`;
  if (definition.unit === "points_per_shot") return value.toFixed(3);
  if (definition.unit === "percentage_points") return `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp`;
  if (definition.unit.includes("points")) return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
  return value.toFixed(2);
}

function profileSlices(player: LocalPlayerSkills, season: number): Slice[] {
  const profile = player.profiles.find((row) => row.season === season);
  if (!profile) return [];
  return Object.entries(PROFILE_LABELS).flatMap(([key, label]) => {
    const metric = profile[key];
    if (typeof metric !== "number") return [];
    const group = key.includes("defense") || key === "disruption"
      ? "defense"
      : key === "rebounding"
        ? "shared"
        : "offense";
    return [{ key, label, group, value: metric } as Slice];
  });
}

export function PlayerSkills({
  index,
  player,
  comparison,
  season,
}: {
  index: LocalSkillIndex;
  player: LocalPlayerSkills;
  comparison: LocalPlayerSkills | null;
  season: number;
}) {
  const [skill, setSkill] = useState("three_point_pct");
  const [mode, setMode] = useState<Mode>("career");
  const [valueMode, setValueMode] = useState<ValueMode>("estimate");
  const definitions = useMemo(
    () => new Map(index.definitions.map((item) => [item.key, item])),
    [index.definitions],
  );
  const definition = definitions.get(skill) ?? index.definitions[0];
  const selected = player.skills[skill];
  const rows = selected?.rows ?? [];
  const league = new Map(index.league[skill] ?? []);
  const gameRows = player.games[skill] ?? [];
  const comparisonRows = comparison?.skills[skill]?.rows ?? [];
  const hasGames = index.gameSkills.includes(skill) && gameRows.length > 0;
  const leftProfile = profileSlices(player, season);
  const rightProfile = comparison ? profileSlices(comparison, season) : [];

  const tableRows = index.definitions.flatMap((item) => {
    const row = player.skills[item.key]?.rows.find((entry) => entry[0] === season);
    return row ? [{ definition: item, row }] : [];
  });

  return (
    <section className="local-skills" aria-labelledby="current-skills-heading">
      <div className="section-head">
        <div>
          <p className="kicker">Skills · localhost</p>
          <h2 id="current-skills-heading">Current ability</h2>
        </div>
        <span className="meta">Updated {rows.at(-1)?.[7] ?? "—"}</span>
      </div>

      <div className="filters skill-filters">
        <label className="field">
          <span>Skill</span>
          <select value={skill} onChange={(event) => setSkill(event.target.value)}>
            {index.definitions.map((item) => (
              <option key={item.key} value={item.key}>{item.label}</option>
            ))}
          </select>
        </label>
        <div className="field">
          <span>Range</span>
          <div className="segmented" role="group" aria-label="Skill range">
            <button type="button" aria-pressed={mode === "career"} onClick={() => setMode("career")}>Career</button>
            <button type="button" aria-pressed={mode === "season"} onClick={() => setMode("season")}>2026</button>
          </div>
        </div>
        <div className="field">
          <span>Value</span>
          <div className="segmented" role="group" aria-label="Skill value">
            <button type="button" aria-pressed={valueMode === "estimate"} onClick={() => setValueMode("estimate")}>Stabilized</button>
            <button type="button" aria-pressed={valueMode === "raw"} onClick={() => setValueMode("raw")}>{mode === "career" ? "Source" : "Game"}</button>
          </div>
        </div>
      </div>

      <div className="grid">
        <Figure
          kicker={mode === "career" ? "Career" : "2026 games"}
          title={definition.label}
          legend={<Legend items={[
            { label: player.name, color: "var(--series-1)" },
            ...(comparison && mode === "career" ? [{ label: comparison.name, color: "var(--series-2)" }] : []),
            { label: "League", color: "var(--text-muted)" },
          ]} shape="key" />}
          table={
            mode === "career" ? (
              <CareerTable rows={rows} definition={definition} />
            ) : (
              <GameTable rows={gameRows} definition={definition} />
            )
          }
          note={mode === "season" && !hasGames ? "Game-level observations are available for free throws and total three-point shooting." : definition.definition}
        >
          <SkillTrajectory
            mode={mode}
            valueMode={valueMode}
            rows={rows}
            games={hasGames ? gameRows : []}
            comparisonRows={comparisonRows}
            league={league}
            definition={definition}
          />
        </Figure>

        <div className="grid two">
          <Figure
            kicker={`Profile · ${season}`}
            title="Skill percentiles"
            table={<ProfileTable slices={leftProfile} comparison={rightProfile} left={player.name} right={comparison?.name} />}
          >
            {comparison && rightProfile.length >= 3 ? (
              <RadarComparison left={leftProfile} right={rightProfile} leftName={player.name} rightName={comparison.name} />
            ) : (
              <Pizza slices={leftProfile} />
            )}
          </Figure>

          <Figure
            kicker={`All skills · ${season}`}
            title="Exact values"
            defaultView="table"
            table={<SkillTable rows={tableRows} />}
          >
            <div className="empty">Open the table for all 34 estimates.</div>
          </Figure>
        </div>
      </div>
    </section>
  );
}

function SkillTrajectory({
  mode,
  valueMode,
  rows,
  games,
  comparisonRows,
  league,
  definition,
}: {
  mode: Mode;
  valueMode: ValueMode;
  rows: LocalSkillRow[];
  games: LocalGameSkillRow[];
  comparisonRows: LocalSkillRow[];
  league: Map<number, number | null>;
  definition: LocalSkillDefinition;
}) {
  const width = 880;
  const height = 280;
  const pad = { left: 58, right: 28, top: 18, bottom: 42 };
  const career = rows.map((row) => ({ label: String(row[0]), primary: value(row, valueMode), raw: row[2], estimate: row[1], league: league.get(row[0]) ?? null }));
  const season = games.map((row) => ({ label: row.date.slice(5), primary: valueMode === "estimate" ? row.estimate : row.raw, raw: row.raw, estimate: row.estimate, league: league.get(2026) ?? null }));
  const points = mode === "career" ? career : season;
  const comparisonBySeason = new Map(
    comparisonRows.map((row) => [String(row[0]), value(row, valueMode)]),
  );
  const comparable = mode === "career"
    ? points.map((point) => comparisonBySeason.get(point.label) ?? null)
    : [];
  const values = [...points.flatMap((row) => [row.primary, row.raw, row.estimate, row.league]), ...comparable].filter((item): item is number => typeof item === "number");
  if (!points.length || !values.length) return <div className="empty">No trajectory for this skill.</div>;
  let low = Math.min(...values);
  let high = Math.max(...values);
  const span = Math.max(high - low, definition.unit === "percent" ? 5 : 0.1);
  low -= span * 0.12;
  high += span * 0.12;
  const x = (index: number) => pad.left + (points.length === 1 ? (width - pad.left - pad.right) / 2 : index * (width - pad.left - pad.right) / (points.length - 1));
  const y = (metric: number) => pad.top + (high - metric) * (height - pad.top - pad.bottom) / (high - low);
  const line = (metrics: (number | null)[]) => {
    let open = false;
    return metrics.map((metric, index) => {
      if (metric === null) { open = false; return ""; }
      const command = open ? "L" : "M";
      open = true;
      return `${command}${x(index)},${y(metric)}`;
    }).join(" ");
  };
  const ticks = [low, (low + high) / 2, high];
  const labelEvery = Math.max(1, Math.ceil(points.length / 6));
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${definition.label} trajectory`}>
        {ticks.map((tick) => <g key={tick}><line className="grid-line" x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="tick" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{format(tick, definition)}</text></g>)}
        <path className="skill-reference" d={line(points.map((row) => row.league))} />
        <path className="skill-observed-line" d={line(points.map((row) => row.raw))} />
        <path className="series-line" stroke="var(--series-1)" d={line(points.map((row) => row.primary))} />
        {mode === "career" && comparisonRows.length > 0 && <path className="series-line" stroke="var(--series-2)" d={line(comparable)} />}
        {points.map((point, index) => point.raw === null ? null : <circle key={index} className="skill-observation" cx={x(index)} cy={y(point.raw)} r={3} />)}
        {points.map((point, index) => index % labelEvery === 0 || index === points.length - 1 ? <text key={index} className="tick" x={x(index)} y={height - 14} textAnchor={index === 0 ? "start" : index === points.length - 1 ? "end" : "middle"}>{point.label}</text> : null)}
      </svg>
    </div>
  );
}

function CareerTable({ rows, definition }: { rows: LocalSkillRow[]; definition: LocalSkillDefinition }) {
  return <table className="mini"><thead><tr><th scope="col">Season</th><th scope="col">Stabilized</th><th scope="col">Source</th><th scope="col">Volume</th><th scope="col">Percentile</th></tr></thead><tbody>{rows.map((row) => <tr key={row[0]}><td>{row[0]}</td><td>{format(row[1], definition)}</td><td>{format(row[2], definition)}</td><td>{row[3]?.toLocaleString() ?? "—"}</td><td>{row[4]?.toFixed(0) ?? "—"}</td></tr>)}</tbody></table>;
}

function GameTable({ rows, definition }: { rows: LocalGameSkillRow[]; definition: LocalSkillDefinition }) {
  return <table className="mini"><thead><tr><th scope="col">Date</th><th scope="col">Stabilized</th><th scope="col">Game</th><th scope="col">Attempts</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${row.date}-${index}`}><td>{row.date}</td><td>{format(row.estimate, definition)}</td><td>{format(row.raw, definition)}</td><td>{row.opportunities ?? "—"}</td></tr>)}</tbody></table>;
}

function SkillTable({ rows }: { rows: { definition: LocalSkillDefinition; row: LocalSkillRow }[] }) {
  return <table className="mini"><thead><tr><th scope="col">Skill</th><th scope="col">Estimate</th><th scope="col">Source</th><th scope="col">Volume</th><th scope="col">Pctl</th><th scope="col">YoY</th></tr></thead><tbody>{rows.map(({ definition, row }) => <tr key={definition.key}><td><b>{definition.label}</b><span className="sub">{definition.group}</span></td><td>{format(row[1], definition)}</td><td>{format(row[2], definition)}</td><td>{row[3]?.toLocaleString() ?? "—"}</td><td>{row[4]?.toFixed(0) ?? "—"}</td><td>{row[5] === null ? "—" : `${row[5] >= 0 ? "+" : ""}${row[5].toFixed(2)}`}</td></tr>)}</tbody></table>;
}

function ProfileTable({ slices, comparison, left, right }: { slices: Slice[]; comparison: Slice[]; left: string; right?: string }) {
  const comparisonMap = new Map(comparison.map((slice) => [slice.key, slice.value]));
  return <table className="mini"><thead><tr><th scope="col">Profile</th><th scope="col">{left}</th>{right && <th scope="col">{right}</th>}</tr></thead><tbody>{slices.map((slice) => <tr key={slice.key}><td>{slice.label}</td><td>{slice.value.toFixed(0)}</td>{right && <td>{comparisonMap.get(slice.key)?.toFixed(0) ?? "—"}</td>}</tr>)}</tbody></table>;
}
