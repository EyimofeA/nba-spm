"use client";

import { useEffect, useMemo, useState } from "react";
import { Figure, ScaleLegend } from "../charts/frame";
import { Landscape, LandscapeDatum, netBoundFor } from "../charts/scatter";
import {
  Catalog,
  LeaderboardRow,
  ModelId,
  RolePoint,
  RoleSide,
  loadRoleMap,
  possessions,
  rating,
  resolveModel,
} from "../lib/data";
import { fmtRating, symmetricBound } from "../lib/viz";
import { MinPossField, ModelField, SeasonField, TeamField } from "./controls";

type SortKey = "net" | "offense" | "defense" | "prior" | "update" | "name" | "team";

export function RatingsView({
  catalog,
  rows,
  season,
  onSeason,
  model,
  onModel,
  minPoss,
  onMinPoss,
  onPlayer,
}: {
  catalog: Catalog;
  rows: LeaderboardRow[];
  season: number;
  onSeason: (season: number) => void;
  model: ModelId;
  onModel: (model: ModelId) => void;
  minPoss: number;
  onMinPoss: (value: number) => void;
  onPlayer: (id: number) => void;
}) {
  const [team, setTeam] = useState("All");
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [roleFilter, setRoleFilter] = useState("All");
  const [roleRows, setRoleRows] = useState<RolePoint[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("net");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [view, setView] = useState<"table" | "chart">("table");
  const [showParts, setShowParts] = useState(false);

  const active = resolveModel(rows, model);

  useEffect(() => {
    if (!(catalog.catalog.role_seasons[roleSide] ?? []).includes(season)) {
      return;
    }
    let live = true;
    void loadRoleMap(roleSide, season)
      .then((next) => { if (live) setRoleRows(next); })
      .catch(() => { if (live) setRoleRows([]); });
    return () => { live = false; };
  }, [catalog, roleSide, season]);

  const roleNames = useMemo(
    () => [...new Set(roleRows.filter((row) => row.Season === season).map((row) => row.raw_role))].sort(),
    [roleRows, season],
  );
  const rolePlayerIds = useMemo(
    () => new Set(roleRows.filter((row) => row.Season === season && (roleFilter === "All" || row.raw_role === roleFilter)).map((row) => row.PLAYER_ID)),
    [roleRows, roleFilter, season],
  );
  const activeRoleFilter = roleNames.includes(roleFilter) ? roleFilter : "All";
  const teams = useMemo(
    () =>
      [
        ...new Set(
          rows
            .map((row) => row.TEAM_ABBREVIATION)
            .filter((v): v is string => Boolean(v)),
        ),
      ].sort(),
    [rows],
  );

  const shaped = useMemo(
    () =>
      rows
        .filter((row) => possessions(row) >= minPoss)
        .filter((row) => team === "All" || row.TEAM_ABBREVIATION === team)
        .filter((row) => activeRoleFilter === "All" || rolePlayerIds.has(row.PLAYER_ID))
        .map((row) => ({
          id: row.PLAYER_ID,
          name: row.PLAYER_NAME,
          team: row.TEAM_ABBREVIATION,
          season: row.Season,
          offense: rating(row, active.prefix, "offense") ?? 0,
          defense: rating(row, active.prefix, "defense") ?? 0,
          net: rating(row, active.prefix, "net") ?? 0,
          poss: possessions(row),
          prior: typeof row.pulse_prior_net === "number" ? row.pulse_prior_net : 0,
          update: typeof row.lineup_update_net === "number" ? row.lineup_update_net : 0,
        })),
    [rows, minPoss, team, activeRoleFilter, rolePlayerIds, active.prefix],
  );

  const sorted = useMemo(() => {
    const direction = order === "asc" ? 1 : -1;
    return [...shaped].sort((a, b) => {
      const compare =
        sortKey === "name"
          ? a.name.localeCompare(b.name)
          : sortKey === "team"
            ? (a.team ?? "").localeCompare(b.team ?? "")
            : a[sortKey] - b[sortKey];
      return compare * direction || b.net - a.net;
    });
  }, [shaped, sortKey, order]);

  const points = useMemo<LandscapeDatum[]>(
    () => shaped.filter((row) => row.poss >= 300),
    [shaped],
  );
  const quadrant = (row: LandscapeDatum) =>
    row.offense >= 0
      ? row.defense >= 0
        ? "Two-way"
        : "Offense first"
      : row.defense >= 0
        ? "Defense first"
        : "Below average";
  const bound = symmetricBound(
    shaped.map((row) => row.net),
    3,
  );
  function sortBy(key: SortKey) {
    if (key === sortKey) setOrder(order === "desc" ? "asc" : "desc");
    else {
      setSortKey(key);
      setOrder(key === "name" || key === "team" ? "asc" : "desc");
    }
  }
  const ariaSort = (key: SortKey) =>
    sortKey === key
      ? order === "asc"
        ? "ascending"
        : "descending"
      : undefined;

  const columns: { key: SortKey; label: string; left?: boolean }[] = [
    { key: "name", label: "Player", left: true },
    { key: "team", label: "Team", left: true },
    { key: "offense", label: "Off" },
    { key: "defense", label: "Def" },
    { key: "net", label: active.id === "pulse" ? "PULSE" : "Net" },
    ...(showParts && active.id === "pulse"
      ? ([{ key: "prior", label: "Prior" }, { key: "update", label: "Update" }] as const)
      : []),
  ];

  return (
    <section className="ratings-workbench" aria-labelledby="ratings-heading">
      <header className="ratings-titlebar">
        <div>
          <p className="kicker">{season - 1}–{String(season).slice(2)}</p>
          <h1 id="ratings-heading">Player impact</h1>
        </div>
      </header>

      <div className="ratings-toolbar">
        <div className="filters" aria-label="Ratings filters">
          <SeasonField seasons={catalog.catalog.seasons} value={season} onChange={onSeason} />
          <ModelField rows={rows} value={model} onChange={onModel} />
          <TeamField teams={teams} value={team} onChange={setTeam} />
          <MinPossField value={minPoss} onChange={onMinPoss} />
          <label className="field">
            <span>Role side</span>
            <select value={roleSide} onChange={(event) => setRoleSide(event.target.value as RoleSide)}>
              <option value="offense">Offense</option>
              <option value="defense">Defense</option>
            </select>
          </label>
          <label className="field">
            <span>Role</span>
            <select value={activeRoleFilter} onChange={(event) => setRoleFilter(event.target.value)} disabled={!roleNames.length}>
              <option value="All">All roles</option>
              {roleNames.map((role) => <option key={role} value={role}>{role}</option>)}
            </select>
          </label>
        </div>
        <div className="segmented" aria-label="Ratings view">
          {active.id === "pulse" && (
            <button type="button" aria-pressed={showParts} onClick={() => setShowParts(!showParts)}>Prior + update</button>
          )}
          <button type="button" aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
          <button type="button" aria-pressed={view === "chart"} onClick={() => setView("chart")}>Map</button>
        </div>
      </div>

      {view === "table" ? (
        <section className="leaderboard-panel" aria-labelledby="leaderboard-heading">
          <header className="board-head">
            <div>
              <span>RANK / PLAYER</span>
              <h2 id="leaderboard-heading">{active.label} leaderboard</h2>
            </div>
            <p>Select any column to sort.</p>
          </header>
          <p className="scroll-hint">Swipe for impact columns →</p>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">
                {active.label} ratings for {season}, points per 100 possessions
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="left">
                    <button type="button" disabled style={{ cursor: "default" }}>#</button>
                  </th>
                  {columns.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={column.left ? "left" : undefined}
                      aria-sort={ariaSort(column.key)}
                    >
                      <button type="button" onClick={() => sortBy(column.key)}>
                        {column.label}
                        {sortKey === column.key && <span className="arrow">{order === "asc" ? "▲" : "▼"}</span>}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, position) => (
                  <tr
                    key={row.id}
                    className="player-row"
                    onClick={(event) => {
                      if (!(event.target as HTMLElement).closest("a")) onPlayer(row.id);
                    }}
                  >
                    <td className="rank">{String(position + 1).padStart(2, "0")}</td>
                    <th scope="row" className="left name">
                      <a className="player-link" href={`#player/${row.id}`}>{row.name}</a>
                    </th>
                    <td className="left team">{row.team ?? "—"}</td>
                    <td className={`metric-cell ${row.offense >= 0 ? "positive" : "negative"}`}>{fmtRating(row.offense)}</td>
                    <td className={`metric-cell ${row.defense >= 0 ? "positive" : "negative"}`}>{fmtRating(row.defense)}</td>
                    <td className="headline">
                      <div className="cellbar">
                        <span className="track" aria-hidden="true">
                          <i
                            className={`fill ${row.net >= 0 ? "pos" : "neg"}`}
                            style={{ width: `${Math.min(50, (Math.abs(row.net) / bound) * 50)}%` }}
                          />
                        </span>
                        <b>{fmtRating(row.net)}</b>
                      </div>
                    </td>
                    {showParts && active.id === "pulse" && <td>{fmtRating(row.prior)}</td>}
                    {showParts && active.id === "pulse" && <td>{fmtRating(row.update)}</td>}
                  </tr>
                ))}
                {!sorted.length && (
                  <tr><td colSpan={columns.length + 1} className="empty">No players match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : (
        <div className="ratings-map">
          <Figure
            kicker={`${season - 1}–${String(season).slice(2)} · ${active.label}`}
            title="Offense against defense"
            legend={<ScaleLegend caption="Net" low={`−${netBoundFor(points).toFixed(1)}`} high={`+${netBoundFor(points).toFixed(1)}`} />}
            table={
              <table className="mini">
                <thead><tr><th scope="col">Player</th><th scope="col">Group</th><th scope="col">Off</th><th scope="col">Def</th><th scope="col">Net</th></tr></thead>
                <tbody>
                  {[...points].sort((a, b) => b.net - a.net).map((row) => (
                    <tr key={row.id}>
                      <th scope="row"><a className="player-link" href={`#player/${row.id}`}>{row.name}</a></th>
                      <td>{quadrant(row)}</td><td>{fmtRating(row.offense)}</td><td>{fmtRating(row.defense)}</td><td>{fmtRating(row.net)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <Landscape rows={points} onSelect={(row) => onPlayer(row.id)} />
          </Figure>
        </div>
      )}
    </section>
  );
}
