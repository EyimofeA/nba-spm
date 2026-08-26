"use client";

import { useMemo, useState } from "react";
import { SpmLabPayload } from "../lib/data";
import { fmtRating } from "../lib/viz";

type Metric = "spm" | "aio";
type SortKey = "selected_net" | "selected_offense" | "selected_defense" | "delta_net";

export function SpmLabView({ lab }: { lab: SpmLabPayload | null }) {
  const [season, setSeason] = useState(2026);
  const [metric, setMetric] = useState<Metric>("aio");
  const [sort, setSort] = useState<SortKey>("selected_net");
  const [ascending, setAscending] = useState(false);
  const rows = useMemo(
    () =>
      (lab?.ratings ?? [])
        .filter((row) => row.Season === season && row.metric === metric)
        .sort((a, b) => (a[sort] - b[sort]) * (ascending ? 1 : -1)),
    [ascending, lab, metric, season, sort],
  );
  const sortBy = (key: SortKey) => {
    if (sort === key) setAscending((value) => !value);
    else {
      setSort(key);
      setAscending(false);
    }
  };
  if (!lab) return <p className="empty">SPM research data unavailable.</p>;
  return (
    <div className="grid">
      <div className="page-head">
        <div>
          <p className="kicker">Local research</p>
          <h1>SPM Lab</h1>
        </div>
        <span className="meta">Five-year target</span>
      </div>

      <section className="card">
        <div className="filters">
          <label className="field">
            <span>Season</span>
            <select value={season} onChange={(event) => setSeason(Number(event.target.value))}>
              {lab.seasons.toReversed().map((value) => <option key={value}>{value}</option>)}
            </select>
          </label>
          <div className="field">
            <span>Rating</span>
            <div className="segmented">
              <button type="button" aria-pressed={metric === "spm"} onClick={() => setMetric("spm")}>SPM</button>
              <button type="button" aria-pressed={metric === "aio"} onClick={() => setMetric("aio")}>AIO</button>
            </div>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead><tr>
              <th className="left"><button type="button" disabled>Player</button></th>
              <th><button type="button" onClick={() => sortBy("selected_offense")}>Offense</button></th>
              <th><button type="button" onClick={() => sortBy("selected_defense")}>Defense</button></th>
              <th><button type="button" onClick={() => sortBy("selected_net")}>Net</button></th>
              <th><button type="button" onClick={() => sortBy("delta_net")}>Change</button></th>
            </tr></thead>
            <tbody>{rows.map((row) => <tr key={`${row.metric}-${row.Season}-${row.PLAYER_ID}`}>
              <td className="left name">{row.PLAYER_NAME}</td>
              <td>{fmtRating(row.selected_offense)}</td>
              <td>{fmtRating(row.selected_defense)}</td>
              <td>{fmtRating(row.selected_net)}</td>
              <td>{fmtRating(row.delta_net)}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><div><p className="kicker">Forward selection</p><h2>Feature families</h2></div></div>
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th className="left"><button type="button" disabled>Family</button></th><th>Side</th><th>Features</th><th>RMSE change</th><th>Changer change</th><th>Decision</th></tr></thead>
            <tbody>{lab.decisions.map((row) => <tr key={row.group}>
              <td className="left name">{row.group.replaceAll("_", " ")}</td>
              <td>{row.side}</td><td>{row.feature_count}</td>
              <td>{fmtRating(row.development_mean_rmse_delta)}</td>
              <td>{fmtRating(row.team_changer_mean_rmse_delta)}</td>
              <td>{row.selected ? "Add" : "Reject"}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><div><p className="kicker">Next-season games</p><h2>AIO validation</h2></div></div>
        <table className="mini"><thead><tr><th>Season</th><th>Baseline</th><th>Selected</th><th>Change</th></tr></thead>
          <tbody>{lab.validation.map((row) => <tr key={row.test_season}><td>{row.test_season}</td><td>{row.baseline_rmse.toFixed(3)}</td><td>{row.selected_rmse.toFixed(3)}</td><td>{fmtRating(row.rmse_delta)}</td></tr>)}</tbody>
        </table>
        <p className="note">Season stabilization uses only that season. Five-year pooling happens after each annual estimate is frozen.</p>
      </section>
    </div>
  );
}
