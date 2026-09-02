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
  availableModels,
  loadRoleMap,
  loadSeason,
  rating,
  resolveModel,
} from "../lib/data";
import { fmtRating, ordinalSuffix } from "../lib/viz";
import { MinPossField, TeamField } from "./controls";

type SortKey = "net" | "offense" | "defense" | "prior" | "update" | "name" | "team" | "season";

type ShapedRow = {
  key: string;
  id: number;
  name: string;
  team: string | null;
  season: number;
  offense: number;
  defense: number;
  net: number;
  poss: number;
  prior: number;
  update: number;
};

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
  const [playerQuery, setPlayerQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("net");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [view, setView] = useState<"table" | "chart">("table");
  const [rowMode, setRowMode] = useState<"season" | "average">("season");
  const [showParts, setShowParts] = useState(false);
  const [selectedSeasons, setSelectedSeasons] = useState<number[]>([season]);
  const [loadedRows, setLoadedRows] = useState<LeaderboardRow[]>(rows);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [role, setRole] = useState("All");
  const [rolePoints, setRolePoints] = useState<RolePoint[]>([]);

  useEffect(() => {
    let live = true;
    Promise.all(selectedSeasons.map(loadSeason)).then((parts) => {
      if (live) setLoadedRows(parts.flat());
    }).catch(() => {
      if (live) setLoadedRows(rows);
    });
    return () => { live = false; };
  }, [rows, selectedSeasons]);

  const roleSeason = Math.max(...selectedSeasons);
  useEffect(() => {
    let live = true;
    Promise.all(selectedSeasons.map((year) => loadRoleMap(roleSide, year))).then((parts) => {
      if (!live) return;
      setRolePoints(parts.flat());
      setRole("All");
    }).catch(() => {
      if (live) setRolePoints([]);
    });
    return () => { live = false; };
  }, [roleSide, selectedSeasons]);

  const displayModels = availableModels(loadedRows).filter((item) =>
    item.available && selectedSeasons.every((year) =>
      loadedRows.some((row) => row.Season === year && typeof row[`${item.prefix}net`] === "number"),
    ),
  );
  const active = displayModels.find((item) => item.id === model)
    ?? resolveModel(loadedRows, model);
  const activePrefix = active.prefix;

  const teams = useMemo(
    () =>
      [
        ...new Set(
          loadedRows
            .map((row) => row.TEAM_ABBREVIATION)
            .filter((v): v is string => Boolean(v)),
        ),
      ].sort(),
    [loadedRows],
  );

  const roleNames = useMemo(
    () => [...new Set(rolePoints.map((point) => point.raw_role))].sort(),
    [rolePoints],
  );
  const roleByPlayerSeason = useMemo(
    () => new Map(rolePoints.map((point) => [`${point.Season}:${point.PLAYER_ID}`, point.raw_role])),
    [rolePoints],
  );

  const seasonRows: ShapedRow[] = loadedRows.map((row) => {
    const offense = rating(row, activePrefix, "offense") ?? 0;
    const defense = rating(row, activePrefix, "defense") ?? 0;
    return {
      key: `${row.Season}:${row.PLAYER_ID}`,
      id: row.PLAYER_ID,
      name: row.PLAYER_NAME,
      team: row.TEAM_ABBREVIATION,
      season: row.Season,
      offense,
      defense,
      net: offense + defense,
      poss: Math.min(Math.max(0, Number(row.Poss_Off)), Math.max(0, Number(row.Poss_Def))),
      prior: Number(row.pulse_prior_net ?? 0),
      update: Number(row.lineup_update_net ?? 0),
    };
  });

  const averageRows: ShapedRow[] = [...loadedRows.reduce((byPlayer, row) => {
        const offWeight = Math.max(0, Number(row.Poss_Off));
        const defWeight = Math.max(0, Number(row.Poss_Def));
        const weight = Math.min(offWeight, defWeight);
        const current = byPlayer.get(row.PLAYER_ID) ?? {
          id: row.PLAYER_ID, name: row.PLAYER_NAME, team: row.TEAM_ABBREVIATION,
          season: row.Season, offSum: 0, defSum: 0, priorSum: 0, updateSum: 0,
          offWeight: 0, defWeight: 0, weight: 0,
        };
        current.offSum += (rating(row, activePrefix, "offense") ?? 0) * offWeight;
        current.defSum += (rating(row, activePrefix, "defense") ?? 0) * defWeight;
        current.priorSum += Number(row.pulse_prior_net ?? 0) * weight;
        current.updateSum += Number(row.lineup_update_net ?? 0) * weight;
        current.offWeight += offWeight;
        current.defWeight += defWeight;
        current.weight += weight;
        if (row.Season >= current.season) {
          current.season = row.Season;
          current.team = row.TEAM_ABBREVIATION;
        }
        byPlayer.set(row.PLAYER_ID, current);
        return byPlayer;
      }, new Map<number, {
        id: number; name: string; team: string | null; season: number;
        offSum: number; defSum: number; priorSum: number; updateSum: number;
        offWeight: number; defWeight: number; weight: number;
      }>()).values()]
        .map((row) => {
          const offense = row.offWeight ? row.offSum / row.offWeight : 0;
          const defense = row.defWeight ? row.defSum / row.defWeight : 0;
          return {
            key: `average:${row.id}`,
            id: row.id, name: row.name, team: row.team, season: row.season,
            offense, defense, net: offense + defense, poss: row.weight,
            prior: row.weight ? row.priorSum / row.weight : 0,
            update: row.weight ? row.updateSum / row.weight : 0,
          };
        });

  const shaped = (rowMode === "season" ? seasonRows : averageRows)
        .filter((row) => row.poss >= minPoss)
        .filter((row) => team === "All" || row.team === team)
        .filter((row) => role === "All" || (
          roleByPlayerSeason.get(`${row.season}:${row.id}`)
          ?? roleByPlayerSeason.get(`${roleSeason}:${row.id}`)
        ) === role);

  const sorted = (() => {
    const direction = order === "asc" ? 1 : -1;
    return [...shaped].sort((a, b) => {
      const compare =
        sortKey === "name"
          ? a.name.localeCompare(b.name)
          : sortKey === "team"
            ? (a.team ?? "").localeCompare(b.team ?? "")
            : sortKey === "season"
              ? a.season - b.season
            : a[sortKey] - b[sortKey];
      return compare * direction || b.net - a.net;
    });
  })();
  const playerNeedle = playerQuery.trim().normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase();
  const tableRows = playerNeedle
    ? sorted.filter((row) => row.name.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase().includes(playerNeedle))
    : sorted;
  const percentiles = (() => {
    const byKey = (key: "offense" | "defense" | "net" | "prior" | "update") => {
      const groups = rowMode === "season"
        ? shaped.reduce((result, row) => {
          result.set(row.season, [...(result.get(row.season) ?? []), row]);
          return result;
        }, new Map<number, ShapedRow[]>())
        : new Map([[0, shaped]]);
      return new Map([...groups.values()].flatMap((group) => {
        const ordered = [...group].sort((a, b) => a[key] - b[key]);
        return ordered.map((row, index) => [row.key, ordered.length <= 1 ? 100 : Math.round((index / (ordered.length - 1)) * 99 + 1)] as const);
      }));
    };
    return {
      offense: byKey("offense"), defense: byKey("defense"), net: byKey("net"),
      prior: byKey("prior"), update: byKey("update"),
    };
  })();

  const points: LandscapeDatum[] = shaped.filter((row) => row.poss >= 300);
  const quadrant = (row: LandscapeDatum) =>
    row.offense >= 0
      ? row.defense >= 0
        ? "Two-way"
        : "Offense first"
      : row.defense >= 0
        ? "Defense first"
        : "Below average";
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
    ...(rowMode === "season" ? [{ key: "season" as const, label: "Season" }] : []),
    { key: "offense", label: "Off" },
    { key: "defense", label: "Def" },
    { key: "net", label: active.id === "pulse" ? "PULSE" : "Net" },
    ...(showParts && active.id === "pulse"
      ? ([{ key: "prior", label: "Prior" }, { key: "update", label: "Update" }] as const)
      : []),
  ];
  const seasonLabel = selectedSeasons
    .map((year) => `${year - 1}–${String(year).slice(2)}`)
    .join(", ");

  return (
    <section className="ratings-workbench" aria-labelledby="ratings-heading">
      <header className="ratings-titlebar">
        <div>
          <p className="kicker">CourtSignal ratings</p>
          <h1 id="ratings-heading">Player impact</h1>
        </div>
      </header>

      <nav className="season-strip ratings-season-strip" aria-label="Rating season">
        {[...catalog.catalog.seasons].reverse().map((year) => (
          <button key={year} type="button" aria-pressed={selectedSeasons.includes(year)} onClick={() => {
            const next = selectedSeasons.includes(year)
              ? selectedSeasons.length === 1 ? selectedSeasons : selectedSeasons.filter((item) => item !== year)
              : [...selectedSeasons, year].sort((a, b) => a - b);
            setSelectedSeasons(next);
            onSeason(Math.max(...next));
          }}>
            {year - 1}–{String(year).slice(2)}
          </button>
        ))}
      </nav>

      <div className="model-tabs" role="tablist" aria-label="Rating model">
        {displayModels.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active.id === item.id}
            onClick={() => onModel(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="ratings-toolbar">
        <div className="filters" aria-label="Ratings filters">
          <TeamField teams={teams} value={team} onChange={setTeam} />
          <MinPossField value={minPoss} onChange={onMinPoss} />
          {rolePoints.length > 0 && <>
            <label className="field"><span>Role side</span><select value={roleSide} onChange={(event) => setRoleSide(event.target.value as RoleSide)}><option value="offense">Offense</option><option value="defense">Defense</option></select></label>
            <label className="field"><span>Role</span><select value={role} onChange={(event) => setRole(event.target.value)}><option value="All">All roles</option>{roleNames.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          </>}
        </div>
        <div className="ratings-actions">
          <button
            type="button"
            className="check-toggle"
            aria-pressed={rowMode === "average"}
            onClick={() => {
              const next = rowMode === "average" ? "season" : "average";
              setRowMode(next);
              if (next === "season") setView("table");
            }}
          ><span aria-hidden="true">{rowMode === "average" ? "✓" : ""}</span>Average</button>
          <div className="segmented" aria-label="Ratings view">
            {active.id === "pulse" && (
              <button type="button" aria-pressed={showParts} onClick={() => setShowParts(!showParts)}>Prior + update</button>
            )}
            <button type="button" aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
            <button type="button" aria-pressed={view === "chart"} onClick={() => { if (selectedSeasons.length > 1) setRowMode("average"); setView("chart"); }}>Map</button>
          </div>
        </div>
      </div>

      {view === "table" ? (
        <section className="leaderboard-panel" aria-labelledby="leaderboard-heading">
          <header className="board-head">
            <div>
              <span>{seasonLabel} / {active.label}</span>
              <h2 id="leaderboard-heading">Leaderboard</h2>
            </div>
            <label className="field table-search">
              <span>Search player</span>
              <input
                type="search"
                value={playerQuery}
                onChange={(event) => setPlayerQuery(event.target.value)}
                placeholder="Player name"
              />
            </label>
          </header>
          <p className="scroll-hint">Swipe for impact columns →</p>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">
                {active.label} ratings for {seasonLabel}, points per 100 possessions
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="left">#</th>
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
                {tableRows.map((row, position) => (
                  <tr
                    key={row.key}
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
                    {rowMode === "season" && <td>{row.season - 1}–{String(row.season).slice(2)}</td>}
                    <td className={`metric-cell ${row.offense >= 0 ? "positive" : "negative"}`}><b>{fmtRating(row.offense)}</b><small>{ordinalSuffix(percentiles.offense.get(row.key) ?? 0)}</small></td>
                    <td className={`metric-cell ${row.defense >= 0 ? "positive" : "negative"}`}><b>{fmtRating(row.defense)}</b><small>{ordinalSuffix(percentiles.defense.get(row.key) ?? 0)}</small></td>
                    <td className="headline">
                      <div className="metric-cell">
                        <b>{fmtRating(row.net)}</b><small>{ordinalSuffix(percentiles.net.get(row.key) ?? 0)}</small>
                      </div>
                    </td>
                    {showParts && active.id === "pulse" && <td className="metric-cell"><b>{fmtRating(row.prior)}</b><small>{ordinalSuffix(percentiles.prior.get(row.key) ?? 0)}</small></td>}
                    {showParts && active.id === "pulse" && <td className="metric-cell"><b>{fmtRating(row.update)}</b><small>{ordinalSuffix(percentiles.update.get(row.key) ?? 0)}</small></td>}
                  </tr>
                ))}
                {!tableRows.length && (
                  <tr><td colSpan={columns.length + 1} className="empty">No players match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : (
        <div className="ratings-map">
          <Figure
            kicker={`${seasonLabel} · ${active.label}${selectedSeasons.length > 1 ? " average" : ""}`}
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
