"use client";

import { useMemo, useState } from "react";
import { MatchupRow } from "../lib/data";
import { fmtInt } from "../lib/viz";

type SortKey =
  | "PLAYER_NAME"
  | "TEAM_ABBREVIATION"
  | "offense_elo"
  | "defense_elo"
  | "net_elo"
  | "matchups";

const elo = (value: number) => value.toFixed(0);
const net = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(0)}`;

/**
 * A deliberately separate surface for the experimental matchup model. Elo is
 * not a points-per-100 measure, so sharing the ratings table would imply a
 * comparability the model has not earned.
 */
export function MatchupsView({
  rows,
  season,
  seasons,
  onSeason,
}: {
  rows: MatchupRow[];
  season: number;
  seasons: number[];
  onSeason: (season: number) => void;
}) {
  const [minimum, setMinimum] = useState(5000);
  const [sortKey, setSortKey] = useState<SortKey>("net_elo");
  const [order, setOrder] = useState<"asc" | "desc">("desc");

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

  const columns: { key: SortKey; label: string; left?: boolean }[] = [
    { key: "PLAYER_NAME", label: "Player", left: true },
    { key: "TEAM_ABBREVIATION", label: "Team", left: true },
    { key: "offense_elo", label: "Off Elo" },
    { key: "defense_elo", label: "Def Elo" },
    { key: "net_elo", label: "Net" },
    { key: "matchups", label: "Matchups" },
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
        <label className="field">
          <span>Min matchups</span>
          <select value={minimum} onChange={(event) => setMinimum(Number(event.target.value))}>
            <option value={0}>Any</option>
            <option value={2000}>2,000+</option>
            <option value={5000}>5,000+</option>
            <option value={8000}>8,000+</option>
          </select>
        </label>
      </div>

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
  );
}
