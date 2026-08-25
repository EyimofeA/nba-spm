"use client";

import { useMemo, useState } from "react";
import { MatchupRow, ShotQualityRow } from "../lib/data";
import { fmtInt } from "../lib/viz";

type SortKey =
  | "PLAYER_NAME"
  | "TEAM_ABBREVIATION"
  | "offense_elo"
  | "defense_elo"
  | "net_elo"
  | "matchups";
type QualitySortKey =
  | "PLAYER_NAME"
  | "TEAM_ABBREVIATION"
  | "raw_net_rank"
  | "shot_quality_net_rank"
  | "rank_change"
  | "lineup_offense_shotmaking_per_100_shots"
  | "lineup_defense_contest_per_100_shots"
  | "lineup_net_residual_per_100_shots"
  | "lineup_shots";
type View = "raw" | "shot_quality";

const elo = (value: number) => value.toFixed(0);
const net = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(0)}`;
const residual = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const rankChange = (value: number) => `${value >= 0 ? "+" : ""}${value}`;

/**
 * A deliberately separate surface for the experimental matchup model. Elo is
 * not a points-per-100 measure, so sharing the ratings table would imply a
 * comparability the model has not earned.
 */
export function MatchupsView({
  rows,
  shotQualityRows,
  season,
  seasons,
  onSeason,
}: {
  rows: MatchupRow[];
  shotQualityRows: ShotQualityRow[];
  season: number;
  seasons: number[];
  onSeason: (season: number) => void;
}) {
  const [view, setView] = useState<View>("raw");
  const [minimum, setMinimum] = useState(5000);
  const [qualityMinimum, setQualityMinimum] = useState(2500);
  const [sortKey, setSortKey] = useState<SortKey>("net_elo");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [qualitySortKey, setQualitySortKey] = useState<QualitySortKey>(
    "lineup_net_residual_per_100_shots",
  );
  const [qualityOrder, setQualityOrder] = useState<"asc" | "desc">("desc");

  const shaped = useMemo(
    () =>
      rows
        .map((row) => ({
          ...row,
          matchups: Math.min(
            row.offense_matchup_possessions,
            row.defense_matchup_possessions,
          ),
        }))
        .filter((row) => row.matchups >= minimum),
    [rows, minimum],
  );

  const sorted = useMemo(() => {
    const direction = order === "asc" ? 1 : -1;
    return [...shaped].sort((a, b) => {
      const left = a[sortKey];
      const right = b[sortKey];
      const compare =
        typeof left === "string" && typeof right === "string"
          ? left.localeCompare(right)
          : Number(left) - Number(right);
      return compare * direction || b.net_elo - a.net_elo;
    });
  }, [shaped, sortKey, order]);

  const sortBy = (key: SortKey) => {
    if (key === sortKey) setOrder(order === "desc" ? "asc" : "desc");
    else {
      setSortKey(key);
      setOrder(key === "PLAYER_NAME" || key === "TEAM_ABBREVIATION" ? "asc" : "desc");
    }
  };

  const shapedQuality = useMemo(
    () => shotQualityRows.filter((row) => row.lineup_shots >= qualityMinimum),
    [shotQualityRows, qualityMinimum],
  );
  const sortedQuality = useMemo(() => {
    const direction = qualityOrder === "asc" ? 1 : -1;
    return [...shapedQuality].sort((a, b) => {
      const left = a[qualitySortKey];
      const right = b[qualitySortKey];
      const compare =
        typeof left === "string" && typeof right === "string"
          ? left.localeCompare(right)
          : Number(left) - Number(right);
      return compare * direction || a.shot_quality_net_rank - b.shot_quality_net_rank;
    });
  }, [shapedQuality, qualityOrder, qualitySortKey]);
  const sortQualityBy = (key: QualitySortKey) => {
    if (key === qualitySortKey) setQualityOrder(qualityOrder === "desc" ? "asc" : "desc");
    else {
      setQualitySortKey(key);
      setQualityOrder(key === "PLAYER_NAME" || key === "TEAM_ABBREVIATION" ? "asc" : "desc");
    }
  };

  const columns: { key: SortKey; label: string; left?: boolean }[] = [
    { key: "PLAYER_NAME", label: "Player", left: true },
    { key: "TEAM_ABBREVIATION", label: "Team", left: true },
    { key: "offense_elo", label: "Off Elo" },
    { key: "defense_elo", label: "Def Elo" },
    { key: "net_elo", label: "Net" },
    { key: "matchups", label: "Matchups" },
  ];
  const qualityColumns: { key: QualitySortKey; label: string; left?: boolean }[] = [
    { key: "PLAYER_NAME", label: "Player", left: true },
    { key: "TEAM_ABBREVIATION", label: "Team", left: true },
    { key: "raw_net_rank", label: "Raw rank" },
    { key: "shot_quality_net_rank", label: "SQ rank" },
    { key: "rank_change", label: "Change" },
    { key: "lineup_offense_shotmaking_per_100_shots", label: "Off residual" },
    { key: "lineup_defense_contest_per_100_shots", label: "Def residual" },
    { key: "lineup_net_residual_per_100_shots", label: "SQ net" },
    { key: "lineup_shots", label: "Lineup shots" },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Experimental</p>
          <h1>Matchup Elo</h1>
        </div>
      </div>

      <div className="filters">
        <div className="field">
          <span>View</span>
          <div className="segmented" role="group" aria-label="Matchup model view">
            <button type="button" aria-pressed={view === "raw"} onClick={() => setView("raw")}>
              Raw
            </button>
            <button type="button" aria-pressed={view === "shot_quality"} onClick={() => setView("shot_quality")}>
              Shot quality
            </button>
          </div>
        </div>
        {view === "raw" ? (
        <label className="field">
          <span>Season</span>
          <select value={season} onChange={(event) => onSeason(Number(event.target.value))}>
            {[...seasons].reverse().map((value) => (
              <option key={value} value={value}>
                {value - 1}–{String(value).slice(2)}
              </option>
            ))}
          </select>
        </label>
        ) : (
          <span className="meta">2025–26 only</span>
        )}
        {view === "raw" ? (
        <label className="field">
          <span>Min matchups</span>
          <select value={minimum} onChange={(event) => setMinimum(Number(event.target.value))}>
            <option value={0}>Any</option>
            <option value={2000}>2,000+</option>
            <option value={5000}>5,000+</option>
            <option value={8000}>8,000+</option>
          </select>
        </label>
        ) : (
          <label className="field">
            <span>Min lineup shots</span>
            <select value={qualityMinimum} onChange={(event) => setQualityMinimum(Number(event.target.value))}>
              <option value={0}>Any</option>
              <option value={1000}>1,000+</option>
              <option value={2500}>2,500+</option>
              <option value={4000}>4,000+</option>
            </select>
          </label>
        )}
      </div>

      {view === "raw" ? (
        <>
          <section className="card matchup-intro">
            <p className="kicker">Three-year trailing window</p>
            <p>
              Matchups from the current season count fully; the two earlier seasons
              are weighted by {rows[0]?.time_decay ?? 0.7} and {((rows[0]?.time_decay ?? 0.7) ** 2).toFixed(2)}.
              Elo 1500 is league average. Net is offense plus defense minus 3000.
            </p>
          </section>

          <section>
            <div className="section-head" style={{ marginTop: 18 }}>
              <div>
                <p className="kicker">Player matchups</p>
                <h2>{season - 1}–{String(season).slice(2)}</h2>
              </div>
              <span className="meta">{sorted.length} players</span>
            </div>
            <div className="table-wrap">
              <table className="data">
                <caption className="visually-hidden">Experimental matchup Elo ratings</caption>
                <thead>
                  <tr>
                    <th scope="col" className="left"><button type="button" disabled style={{ cursor: "default" }}>#</button></th>
                    {columns.map((column) => (
                      <th key={column.key} scope="col" className={column.left ? "left" : undefined}>
                        <button type="button" onClick={() => sortBy(column.key)}>
                          {column.label}{sortKey === column.key && <span className="arrow">{order === "asc" ? "▲" : "▼"}</span>}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((row, index) => (
                    <tr key={row.PLAYER_ID}>
                      <td className="rank">{index + 1}</td>
                      <td className="left name">{row.PLAYER_NAME}</td>
                      <td className="left team">{row.TEAM_ABBREVIATION ?? "—"}</td>
                      <td>{elo(row.offense_elo)}</td>
                      <td>{elo(row.defense_elo)}</td>
                      <td className="headline">{net(row.net_elo)}</td>
                      <td>{fmtInt(row.matchups)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <p className="note matchup-caveat">
            This is a matchup-assignment model, not a primary-defender estimate.
            It is useful for iteration; its defensive ranking is not yet validated
            for publication or inclusion in RAPM, SPM, or AIO.
          </p>
        </>
      ) : (
        <>
          <section className="card matchup-intro">
            <p className="kicker">Local research comparison</p>
            <h2>Location-adjusted shot residuals</h2>
            <p>
              This view holds shot location and game context fixed, then estimates
              what the five players on each side add to makes above expectation.
              It is a five-on-five diagnostic, not an individual defender-at-shot model.
              Compare ranks, not Elo against residual points.
            </p>
          </section>
          <section>
            <div className="section-head" style={{ marginTop: 18 }}>
              <div>
                <p className="kicker">Raw versus shot quality</p>
                <h2>2025–26</h2>
              </div>
              <span className="meta">{sortedQuality.length} players</span>
            </div>
            <div className="table-wrap">
              <table className="data">
                <caption className="visually-hidden">Raw matchup and shot-quality diagnostic rank comparison</caption>
                <thead>
                  <tr>
                    <th scope="col" className="left"><button type="button" disabled style={{ cursor: "default" }}>#</button></th>
                    {qualityColumns.map((column) => (
                      <th key={column.key} scope="col" className={column.left ? "left" : undefined}>
                        <button type="button" onClick={() => sortQualityBy(column.key)}>
                          {column.label}{qualitySortKey === column.key && <span className="arrow">{qualityOrder === "asc" ? "▲" : "▼"}</span>}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedQuality.map((row, index) => (
                    <tr key={row.PLAYER_ID}>
                      <td className="rank">{index + 1}</td>
                      <td className="left name">{row.PLAYER_NAME}</td>
                      <td className="left team">{row.TEAM_ABBREVIATION ?? "—"}</td>
                      <td>{row.raw_net_rank}</td>
                      <td>{row.shot_quality_net_rank}</td>
                      <td className="headline">{rankChange(row.rank_change)}</td>
                      <td>{residual(row.lineup_offense_shotmaking_per_100_shots)}</td>
                      <td>{residual(row.lineup_defense_contest_per_100_shots)}</td>
                      <td className="headline">{residual(row.lineup_net_residual_per_100_shots)}</td>
                      <td>{fmtInt(row.lineup_shots)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <p className="note matchup-caveat">
            A positive change means the player ranks higher in the shot-quality diagnostic.
            This diagnostic did not clear its held-out improvement gate, so it remains local only.
          </p>
        </>
      )}
    </>
  );
}
