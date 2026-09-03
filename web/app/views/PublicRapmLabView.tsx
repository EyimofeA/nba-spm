"use client";

import { useEffect, useMemo, useState } from "react";
import { fmtRating, ordinalSuffix } from "../lib/viz";
import { WP_RELEASE_ID } from "../lib/rapmRelease";

type Period = { id: string; label: string; url: string; rows: number };
type Estimand = { id: string; title: string; unit: string; note: string; periods: Period[] };
type Catalog = { schema: string; lineage: { wp_run: string }; estimands: Estimand[] };
type Row = Record<string, string | number | null> & { PLAYER_NAME?: string };

const DISPLAY_ORDER = [
  "annual", "rolling-three", "rolling-five", "current-time-decay",
  "luck-adjusted", "win-probability", "log-odds-win-probability",
  "game-level-pm", "six-factor-annual", "point-channels", "teammate-effects",
  "teammate-efg", "observable-scoring-channels", "observable-finish-channels",
  "coach", "units",
];
const LABELS: Record<string, string> = {
  annual: "1 year", "rolling-three": "3 years", "rolling-five": "5 years",
  "current-time-decay": "Time decay", "luck-adjusted": "Luck adjusted",
  "win-probability": "WP-RAPM", "log-odds-win-probability": "Log-odds WP",
  "game-level-pm": "Game PM",
  "six-factor-annual": "Six factor", "point-channels": "Point channels",
  "teammate-effects": "Teammates", coach: "Coaches", units: "Units",
  "teammate-efg": "Teammate eFG", "observable-scoring-channels": "Scoring channels",
  "observable-finish-channels": "Finish channels",
};
const STANDARD_COLUMNS = [
  ["PLAYER_NAME", "Player"], ["offense", "Off"], ["defense", "Def"], ["net", "RAPM"],
] as const;
const FACTOR_COLUMNS = [
  ["PLAYER_NAME", "Player"],
  ["off_ts", "Off TS"], ["off_tov", "Off TOV"], ["off_reb", "Off REB"],
  ["def_ts", "Def TS"], ["def_tov", "Def TOV"], ["def_reb", "Def REB"],
  ["offense", "Off"], ["defense", "Def"], ["net", "RAPM"],
] as const;
const GAME_PM_COLUMNS = [
  ["PLAYER_NAME", "Player"], ["net", "Game PM"],
] as const;
const POINT_CHANNEL_COLUMNS = [
  ["PLAYER_NAME", "Player"],
  ["one_point_offense", "Off"], ["one_point_defense", "Def"], ["one_point_net", "Net"],
  ["two_point_offense", "Off"], ["two_point_defense", "Def"], ["two_point_net", "Net"],
  ["three_plus_offense", "Off"], ["three_plus_defense", "Def"], ["three_plus_net", "Net"],
  ["offense", "Off"], ["defense", "Def"], ["net", "RAPM"],
] as const;
const TEAMMATE_COLUMNS = [
  ["PLAYER_NAME", "Player"],
  ["teammate_scoring", "Scoring"], ["teammate_turnovers", "TOV prevention"],
  ["teammate_assists", "Assists"], ["teammate_steals", "Steals"],
  ["teammate_blocks", "Blocks"], ["teammate_oreb", "OREB"],
  ["teammate_dreb", "DREB"],
] as const;
const TEAMMATE_EFG_COLUMNS = [
  ["PLAYER_NAME", "Player"], ["teammate_efg_offense", "Teammate eFG"],
  ["shot_defense", "Shot defense"], ["teammate_efg_net", "Net"],
] as const;
const OBSERVABLE_SCORING_COLUMNS = [
  ["PLAYER_NAME", "Player"], ["rim_assists_net", "Rim AST"],
  ["transition_points_net", "Transition"], ["three_point_points_net", "3PT"],
  ["free_throw_points_net", "FT"], ["midrange_attempts_net", "Mid freq"],
  ["rim_points_net", "Rim"],
] as const;
const OBSERVABLE_FINISH_COLUMNS = [
  ["PLAYER_NAME", "Player"], ["playtype_transition_points_net", "Transition"],
  ["playtype_putback_points_net", "Putback"], ["playtype_cut_points_net", "Cut"],
  ["playtype_drive_points_net", "Drive"], ["playtype_pullup_points_net", "Pull-up"],
  ["playtype_post_points_net", "Post-like"], ["playtype_spotup_points_net", "Jump shot"],
  ["playtype_other_points_net", "Other"],
] as const;
const DEFAULT_SORT: Record<string, string> = {
  "teammate-effects": "teammate_scoring",
  "teammate-efg": "teammate_efg_net",
  "observable-scoring-channels": "rim_points_net",
  "observable-finish-channels": "playtype_drive_points_net",
};

const numeric = (row: Row, key: string) => {
  const direct = row[key];
  const annual = row[`rapm_${key}`];
  return typeof direct === "number" ? direct : typeof annual === "number" ? annual : 0;
};
const exposure = (row: Row) => {
  if (typeof row.minutes === "number") return row.minutes;
  if (typeof row.possession_opportunities === "number") return row.possession_opportunities;
  if (row.Poss_Off == null || row.Poss_Def == null) return Number.POSITIVE_INFINITY;
  return Math.min(Number(row.Poss_Off), Number(row.Poss_Def));
};

export function PublicRapmLabView() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [estimandId, setEstimandId] = useState("annual");
  const [periodId, setPeriodId] = useState("2026");
  const [board, setBoard] = useState<{ url: string; rows: Row[]; error?: boolean } | null>(null);
  const [sort, setSort] = useState("net");
  const [ascending, setAscending] = useState(false);
  const [query, setQuery] = useState("");
  const [minimum, setMinimum] = useState(100);

  useEffect(() => {
    fetch(`/data/rapm/catalog.json?v=${WP_RELEASE_ID}`)
      .then((response) => {
        if (!response.ok) throw new Error("RAPM catalog unavailable");
        return response.json() as Promise<Catalog>;
      })
      .then((data) => {
        if (data.lineage?.wp_run !== WP_RELEASE_ID) throw new Error("Stale WP release");
        setCatalog(data);
      })
      .catch(() => setCatalog(null));
  }, []);

  const estimands = useMemo(() => {
    const source = catalog?.estimands ?? [];
    return DISPLAY_ORDER.map((id) => source.find((item) => item.id === id))
      .filter((item): item is Estimand => Boolean(item));
  }, [catalog]);
  const estimand = estimands.find((item) => item.id === estimandId) ?? estimands[0];
  const period = estimand?.periods.find((item) => item.id === periodId) ?? estimand?.periods.at(-1);
  const columns = estimand?.id === "six-factor-annual"
    ? FACTOR_COLUMNS
    : estimand?.id === "teammate-effects"
      ? TEAMMATE_COLUMNS
    : estimand?.id === "teammate-efg"
      ? TEAMMATE_EFG_COLUMNS
    : estimand?.id === "observable-scoring-channels"
      ? OBSERVABLE_SCORING_COLUMNS
    : estimand?.id === "observable-finish-channels"
      ? OBSERVABLE_FINISH_COLUMNS
    : estimand?.id === "point-channels"
      ? POINT_CHANNEL_COLUMNS
      : estimand?.id === "game-level-pm"
        ? GAME_PM_COLUMNS
        : STANDARD_COLUMNS;
  const usesMinutes = estimand?.id === "game-level-pm";
  const formatValue = (value: number) => ["win-probability", "log-odds-win-probability"].includes(estimand?.id ?? "")
    ? `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`
    : fmtRating(value);

  useEffect(() => {
    if (!period) return;
    let live = true;
    fetch(period.url)
      .then((response) => {
        if (!response.ok) throw new Error("RAPM leaderboard unavailable");
        return response.json() as Promise<Row[]>;
      })
      .then((rows) => { if (live) setBoard({ url: period.url, rows }); })
      .catch(() => { if (live) setBoard({ url: period.url, rows: [], error: true }); });
    return () => { live = false; };
  }, [period]);

  const current = board?.url === period?.url ? board : null;
  const qualified = useMemo(() => (current?.rows ?? []).filter((row) => exposure(row) >= minimum), [current, minimum]);
  const visible = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const direction = ascending ? 1 : -1;
    return qualified
      .filter((row) => !needle || String(row.PLAYER_NAME ?? "").toLocaleLowerCase().includes(needle))
      .sort((left, right) => {
        const delta = sort === "PLAYER_NAME"
          ? String(left.PLAYER_NAME ?? "").localeCompare(String(right.PLAYER_NAME ?? ""))
          : numeric(left, sort) - numeric(right, sort);
        return delta * direction;
      });
  }, [ascending, query, qualified, sort]);
  const percentiles = useMemo(() => {
    const byKey = (key: string) => {
      const ordered = [...qualified].sort((left, right) => numeric(left, key) - numeric(right, key));
      return new Map(ordered.map((row, index) => [row, ordered.length <= 1 ? 100 : Math.round((index / (ordered.length - 1)) * 99 + 1)]));
    };
    return { offense: byKey("offense"), defense: byKey("defense"), net: byKey("net") };
  }, [qualified]);

  const selectEstimand = (next: Estimand) => {
    setEstimandId(next.id);
    setPeriodId(next.periods.at(-1)?.id ?? "");
    setSort(DEFAULT_SORT[next.id] ?? "net");
    setAscending(false);
  };
  const setSorting = (next: string) => {
    if (next === sort) setAscending(!ascending);
    else { setSort(next); setAscending(next === "PLAYER_NAME"); }
  };

  if (!catalog || !estimand || !period) return <p className="empty">RAPM Lab unavailable.</p>;

  return (
    <section className="ratings-workbench" aria-labelledby="rapm-lab-heading">
      <header className="ratings-titlebar">
        <div><p className="kicker">RAPM ratings</p><h1 id="rapm-lab-heading">RAPM Lab</h1></div>
      </header>

      <div className="model-tabs" role="tablist" aria-label="RAPM horizon">
        {estimands.map((item) => <button key={item.id} type="button" role="tab" aria-selected={item.id === estimand.id} onClick={() => selectEstimand(item)}>{LABELS[item.id] ?? item.title}</button>)}
      </div>

      <nav className="season-strip ratings-season-strip" aria-label="RAPM season or window">
        {(estimand.id === "units" ? estimand.periods : [...estimand.periods].reverse()).map((item) => <button key={item.id} type="button" aria-pressed={item.id === period.id} onClick={() => setPeriodId(item.id)}>{item.label}</button>)}
      </nav>

      <div className="ratings-toolbar"><div className="filters">
        <label className="field"><span>Find player</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search" /></label>
        <label className="field"><span>{usesMinutes ? "Min min" : "Min poss"}</span><select value={minimum} onChange={(event) => setMinimum(Number(event.target.value))}><option value={100}>100+</option><option value={500}>500+</option><option value={1000}>1,000+</option><option value={2000}>2,000+</option></select></label>
      </div></div>

      <section className="leaderboard-panel">
        <header className="board-head"><div><span>{period.label}</span><h2>{estimand.title}</h2></div></header>
        <p className="note">{estimand.unit}</p>
        {estimand.note && <p className="note">{estimand.note}</p>}
        {!current && <p className="empty" role="status">Loading leaderboard…</p>}
        {current?.error && <p className="empty" role="alert">Leaderboard unavailable.</p>}
        <p className="scroll-hint">Swipe for impact columns →</p>
        <div className="table-wrap"><table className="data rapm-board">
          <thead>
            {estimand.id === "six-factor-annual" && <tr><th className="left">#</th><th className="left">Player</th><th colSpan={3}>Offense factors</th><th colSpan={3}>Defense factors</th><th colSpan={3}>Total impact</th></tr>}
            {estimand.id === "point-channels" && <tr><th className="left">#</th><th className="left">Player</th><th colSpan={3}>One-point</th><th colSpan={3}>Two-point</th><th colSpan={3}>Three-plus</th><th colSpan={3}>Total impact</th></tr>}
            <tr><th className="left">#</th>{columns.map(([key, label]) => <th key={key} className={key === "PLAYER_NAME" ? "left" : undefined} aria-sort={sort === key ? (ascending ? "ascending" : "descending") : undefined}><button type="button" onClick={() => setSorting(key)}>{label}</button></th>)}</tr>
          </thead>
          <tbody>{visible.map((row, index) => <tr key={`${row.PLAYER_ID ?? row.PLAYER_NAME}-${index}`}>
            <td className="rank">{String(index + 1).padStart(2, "0")}</td>
            {columns.map(([key]) => key === "PLAYER_NAME"
              ? <th key={key} className="left name">{row.PLAYER_NAME ?? "—"}</th>
              : ["offense", "defense", "net"].includes(key)
                ? <td key={key} className={`metric-cell ${numeric(row, key) >= 0 ? "positive" : "negative"}`}><b>{formatValue(numeric(row, key))}</b><small>{ordinalSuffix(percentiles[key as "offense" | "defense" | "net"].get(row) ?? 0)}</small></td>
                : <td key={key}>{formatValue(numeric(row, key))}</td>)}
          </tr>)}</tbody>
        </table></div>
      </section>
    </section>
  );
}
