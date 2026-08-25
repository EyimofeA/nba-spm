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
import { fmtInt, fmtRating, symmetricBound } from "../lib/viz";
import { MinPossField, ModelField, SeasonField, TeamField } from "./controls";

type SortKey = "net" | "offense" | "defense" | "poss" | "name" | "team";

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
            : sortKey === "poss"
              ? a.poss - b.poss
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
  const quadrantCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const row of points) {
      const group = quadrant(row);
      counts.set(group, (counts.get(group) ?? 0) + 1);
    }
    return counts;
  }, [points]);

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
    { key: "net", label: "Net" },
    { key: "poss", label: "Poss" },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Ratings</p>
          <h1>
            {season - 1}–{String(season).slice(2)} player impact
          </h1>
        </div>
        <div className="segmented" aria-label="Ratings view">
          <button type="button" aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
          <button type="button" aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
        </div>
      </div>

      {/* One filter row, above everything it scopes. */}
      <div className="filters">
        <SeasonField
          seasons={catalog.catalog.seasons}
          value={season}
          onChange={onSeason}
        />
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

      {view === "table" ? <section>
          <div className="section-head" style={{ marginTop: 8 }}>
            <div>
              <p className="kicker">Full board</p>
              <h2>{active.label}</h2>
            </div>
            <span className="meta">
              Sort any column · select a row for the player page
            </span>
          </div>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">
                {active.label} ratings for {season}, points per 100 possessions
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="left">
                    <button
                      type="button"
                      disabled
                      style={{ cursor: "default" }}
                    >
                      #
                    </button>
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
                        {sortKey === column.key && (
                          <span className="arrow">
                            {order === "asc" ? "▲" : "▼"}
                          </span>
                        )}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, position) => (
                  <tr
                    key={row.id}
                    tabIndex={0}
                    onClick={() => onPlayer(row.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onPlayer(row.id);
                      }
                    }}
                  >
                    <td className="rank">{position + 1}</td>
                    <td className="left name">{row.name}</td>
                    <td className="left team">{row.team ?? "—"}</td>
                    <td>{fmtRating(row.offense)}</td>
                    <td>{fmtRating(row.defense)}</td>
                    <td className="headline">
                      <div className="cellbar">
                        <span className="track" aria-hidden="true">
                          <i
                            className={`fill ${row.net >= 0 ? "pos" : "neg"}`}
                            style={{
                              width: `${Math.min(50, (Math.abs(row.net) / bound) * 50)}%`,
                            }}
                          />
                        </span>
                        <b>{fmtRating(row.net)}</b>
                      </div>
                    </td>
                    <td>{fmtInt(row.poss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
      </section> : (
        <>
          <div className="kpi-row" style={{ marginBottom: 14 }}>
            {["Two-way", "Offense first", "Defense first", "Below average"].map((label) => (
              <div className="tile" key={label}>
                <div className="tile-label">{label}</div>
                <div className="tile-value">{quadrantCounts.get(label) ?? 0}</div>
                <div className="tile-sub">
                  {points.length ? Math.round(((quadrantCounts.get(label) ?? 0) / points.length) * 100) : 0}% of field
                </div>
              </div>
            ))}
          </div>
          <Figure
            kicker={active.label}
            title="Offense against defense"
            legend={<ScaleLegend caption="Net" low={`−${netBoundFor(points).toFixed(1)}`} high={`+${netBoundFor(points).toFixed(1)}`} />}
            table={
              <table className="mini">
                <thead><tr><th scope="col">Player</th><th scope="col">Group</th><th scope="col">Off</th><th scope="col">Def</th><th scope="col">Net</th></tr></thead>
                <tbody>
                  {[...points].sort((a, b) => b.net - a.net).map((row) => (
                    <tr key={row.id} onClick={() => onPlayer(row.id)}>
                      <td>{row.name}</td><td>{quadrant(row)}</td><td>{fmtRating(row.offense)}</td><td>{fmtRating(row.defense)}</td><td>{fmtRating(row.net)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <Landscape rows={points} onSelect={(row) => onPlayer(row.id)} />
          </Figure>
        </>
      )}
    </>
  );
}
