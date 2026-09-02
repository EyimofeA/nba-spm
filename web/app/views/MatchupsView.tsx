"use client";

import { useMemo, useState } from "react";
import { MatchupLabPayload } from "../lib/data";
import { fmtInt, fmtRating } from "../lib/viz";

type Model = "raw" | "scorer_adjusted" | "contextual" | "sequential";
type Side = "net" | "offense" | "defense";
type SortKey = "PLAYER_NAME" | "value" | "reliability" | "exposure";

const MODEL_LABELS: Record<Model, string> = {
  raw: "Raw",
  scorer_adjusted: "Two-way ridge",
  contextual: "Contextual ridge",
  sequential: "Sequential",
};

const modelValue = (row: Record<string, unknown>, model: Model, side: Side) =>
  Number(row[`${model}_${side}`] ?? Number.NaN);

export function MatchupsView({ lab }: { lab: MatchupLabPayload | null }) {
  const [model, setModel] = useState<Model>("contextual");
  const [side, setSide] = useState<Side>("net");
  const [channel, setChannel] = useState("total_scoring");
  const [minimum, setMinimum] = useState(250);
  const [selected, setSelected] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("value");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

  const rows = useMemo(() => {
    if (!lab) return [];
    if (channel !== "total_scoring") {
      return lab.channels
        .filter((row) => row.channel === channel)
        .map((row) => ({
          ...row,
          value: side === "net" ? row.offense + row.defense : row[side],
          exposure: Math.min(row.offense_matchup_possessions, row.defense_matchup_possessions),
        }));
    }
    return lab.players.map((row) => ({
      ...row,
      value: modelValue(row as unknown as Record<string, unknown>, model, side),
      exposure: Math.min(row.offense_matchup_possessions, row.defense_matchup_possessions),
    }));
  }, [channel, lab, model, side]);

  const sorted = useMemo(() => {
    const direction = sortDirection === "asc" ? 1 : -1;
    return rows.filter((row) => row.exposure >= minimum && Number.isFinite(row.value))
      .sort((a, b) => {
        if (sortKey === "PLAYER_NAME") return a.PLAYER_NAME.localeCompare(b.PLAYER_NAME) * direction;
        return (Number(a[sortKey]) - Number(b[sortKey])) * direction;
      });
  }, [minimum, rows, sortDirection, sortKey]);

  function sortBy(key: SortKey) {
    if (key === sortKey) setSortDirection((value) => value === "desc" ? "asc" : "desc");
    else {
      setSortKey(key);
      setSortDirection(key === "PLAYER_NAME" ? "asc" : "desc");
    }
  }
  const history = useMemo(
    () => lab?.history.filter((row) => row.PLAYER_ID === selected).sort((a, b) => a.Season - b.Season) ?? [],
    [lab, selected],
  );
  const validation = useMemo(
    () => lab?.validation.filter((row) => row.model !== "league_average") ?? [],
    [lab],
  );
  const channels = useMemo(
    () => [...new Set(["total_scoring", ...(lab?.channels.map((row) => row.channel) ?? [])])],
    [lab],
  );

  if (!lab) {
    return <section className="card"><p>Matchup Lab data unavailable.</p></section>;
  }

  return (
    <>
      <div className="page-head">
        <div><p className="kicker">Local research</p><h1>Matchup Lab</h1></div>
        <span className="meta">2018–{lab.latest_season}</span>
      </div>

      <div className="filters">
        <label className="field"><span>Model</span>
          <select value={model} disabled={channel !== "total_scoring"} onChange={(event) => setModel(event.target.value as Model)}>
            {Object.entries(MODEL_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="field"><span>Side</span>
          <select value={side} onChange={(event) => setSide(event.target.value as Side)}>
            <option value="net">Net</option><option value="offense">Offense</option><option value="defense">Defense</option>
          </select>
        </label>
        <label className="field"><span>Channel</span>
          <select value={channel} onChange={(event) => setChannel(event.target.value)}>
            {channels.map((value) => <option key={value} value={value}>{value.replaceAll("_", " ")}</option>)}
          </select>
        </label>
        <label className="field"><span>Min exposure</span>
          <select value={minimum} onChange={(event) => setMinimum(Number(event.target.value))}>
            <option value={0}>Any</option><option value={100}>100+</option><option value={250}>250+</option><option value={500}>500+</option>
          </select>
        </label>
      </div>

      <section>
        <div className="section-head"><div><p className="kicker">Leaderboard</p><h2>{channel.replaceAll("_", " ")}</h2></div></div>
        <div className="table-wrap"><table className="data">
          <thead><tr><th>#</th>{([
            ["PLAYER_NAME", "Player"], ["value", "Rating"], ["reliability", "Reliability"], ["exposure", "Exposure"],
          ] as [SortKey, string][]).map(([key, label]) => <th key={key} className={key === "PLAYER_NAME" ? "left" : undefined} aria-sort={sortKey === key ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}><button type="button" onClick={() => sortBy(key)}>{label}</button></th>)}</tr></thead>
          <tbody>{sorted.map((row, index) => (
            <tr key={row.PLAYER_ID} className={selected === row.PLAYER_ID ? "selected" : undefined} onClick={() => setSelected(row.PLAYER_ID)}>
              <td>{index + 1}</td><td className="left name">{row.PLAYER_NAME}</td><td className="headline">{fmtRating(row.value)}</td>
              <td>{Math.round(100 * row.reliability)}%</td><td>{fmtInt(row.exposure)}</td>
            </tr>
          ))}</tbody>
        </table></div>
      </section>

      {history.length > 0 && <section className="card">
        <p className="kicker">Player history</p><h2>{history[0].PLAYER_NAME}</h2>
        <div className="table-wrap"><table className="mini"><thead><tr><th>Season</th><th>Offense</th><th>Defense</th><th>Net</th></tr></thead>
          <tbody>{history.map((row) => <tr key={row.Season}><td>{row.Season}</td><td>{fmtRating(row.raw_offense)}</td><td>{fmtRating(row.raw_defense)}</td><td>{fmtRating(row.raw_net)}</td></tr>)}</tbody>
        </table></div>
      </section>}

      <section className="card">
        <p className="kicker">Chronological validation</p>
        <div className="table-wrap"><table className="mini"><thead><tr><th>Season</th><th>Model</th><th>RMSE</th><th>Correlation</th><th>Slope</th></tr></thead>
          <tbody>{validation.map((row) => <tr key={`${row.Season}-${row.model}`}><td>{row.Season}</td><td>{row.model.replaceAll("_", " ")}</td><td>{row.rmse.toFixed(2)}</td><td>{row.correlation.toFixed(3)}</td><td>{row.calibration_slope.toFixed(2)}</td></tr>)}</tbody>
        </table></div>
      </section>

      <section className="card">
        <p className="kicker">Frequent scorer–defender pairs</p>
        <div className="table-wrap"><table className="mini"><thead><tr><th>Scorer</th><th>Defender</th><th>Exposure</th><th>Points</th></tr></thead>
          <tbody>{lab.pairs.slice(0, 25).map((row) => <tr key={`${row.SCORER_ID}-${row.DEFENDER_ID}`}><td>{row.SCORER_NAME}</td><td>{row.DEFENDER_NAME}</td><td>{row.matchup_possessions.toFixed(1)}</td><td>{row.player_points.toFixed(0)}</td></tr>)}</tbody>
        </table></div>
      </section>

      <p className="note matchup-caveat">Aggregated scorer–listed-defender assignments are not shot-level guarding. This lab remains local until a model clears the chronological gate.</p>
    </>
  );
}
