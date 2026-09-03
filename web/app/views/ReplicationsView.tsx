"use client";

import { useEffect, useMemo, useState } from "react";
import {
  RapmLabPayload,
  ReplicationLeaderboard,
  ReplicationRecord,
} from "../lib/data";
import { fmtInt, fmtRating, ordinalSuffix } from "../lib/viz";

type PublicBoard = Omit<ReplicationLeaderboard, "rows"> & {
  rows: number;
  url: string;
};
type PublicCatalog = {
  schema: "courtsignal_reconstructions_v1";
  replications: Pick<
    ReplicationRecord,
    "metric" | "matched_rows" | "pearson" | "r_squared" | "run_id"
  >[];
  boards: PublicBoard[];
};

export function ReplicationsView({ lab }: { lab: RapmLabPayload | null }) {
  const [publicCatalog, setPublicCatalog] = useState<PublicCatalog | null>(null);
  const [remote, setRemote] = useState<{
    id: string;
    rows: ReplicationLeaderboard["rows"];
    error?: boolean;
  } | null>(null);

  useEffect(() => {
    if (lab) return;
    fetch("/data/reconstructions/catalog.json")
      .then((response) => {
        if (!response.ok) throw new Error("Reconstruction catalog unavailable");
        return response.json() as Promise<PublicCatalog>;
      })
      .then((data) => {
        if (data.schema !== "courtsignal_reconstructions_v1") {
          throw new Error("Unexpected reconstruction catalog");
        }
        setPublicCatalog(data);
      })
      .catch(() => setPublicCatalog(null));
  }, [lab]);

  const boards = useMemo(
    () => lab?.replication_leaderboards ?? publicCatalog?.boards ?? [],
    [lab, publicCatalog],
  );
  const metrics = useMemo(() => [...new Set(boards.map((board) => board.metric))], [boards]);
  const [metric, setMetric] = useState(metrics[0] ?? "");
  const effectiveMetric = metrics.includes(metric) ? metric : metrics[0] ?? "";
  const seasons = useMemo(
    () => boards.filter((board) => board.metric === effectiveMetric).map((board) => board.season).sort((a, b) => b - a),
    [boards, effectiveMetric],
  );
  const [season, setSeason] = useState(seasons[0] ?? 2026);
  const effectiveSeason = seasons.includes(season) ? season : seasons[0] ?? 2026;
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("net");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");

  const selectedMeta = boards.find((board) => board.metric === effectiveMetric && board.season === effectiveSeason)
    ?? boards.find((board) => board.metric === effectiveMetric)
    ?? boards[0];
  const publicBoard = selectedMeta && "url" in selectedMeta ? selectedMeta : null;

  useEffect(() => {
    if (!publicBoard) return;
    let live = true;
    fetch(publicBoard.url)
      .then((response) => {
        if (!response.ok) throw new Error("Reconstruction leaderboard unavailable");
        return response.json() as Promise<ReplicationLeaderboard["rows"]>;
      })
      .then((rows) => {
        if (live) setRemote({ id: publicBoard.id, rows });
      })
      .catch(() => {
        if (live) setRemote({ id: publicBoard.id, rows: [], error: true });
      });
    return () => { live = false; };
  }, [publicBoard]);

  const selected = useMemo(() => selectedMeta && !("url" in selectedMeta)
    ? selectedMeta
    : selectedMeta && remote?.id === selectedMeta.id
      ? { ...selectedMeta, rows: remote.rows }
      : null, [remote, selectedMeta]);
  const replication = selectedMeta
    ? (lab?.replications ?? publicCatalog?.replications ?? []).find((row) => row.metric === selectedMeta.metric)
    : undefined;
  const isRaptor = selectedMeta?.metric.includes("RAPTOR") ?? false;
  const rows = useMemo(() => {
    if (!selected) return [];
    const needle = query.trim().toLocaleLowerCase();
    const sign = direction === "asc" ? 1 : -1;
    return [...selected.rows]
      .filter((row) => !needle || row.player.toLocaleLowerCase().includes(needle))
      .sort((a, b) => {
        if (sortKey === "player") return a.player.localeCompare(b.player) * sign;
        const left = a[sortKey];
        const right = b[sortKey];
        if (typeof left === "string" || typeof right === "string") {
          return String(left ?? "").localeCompare(String(right ?? "")) * sign;
        }
        return ((Number(left) || 0) - (Number(right) || 0)) * sign;
      });
  }, [direction, query, selected, sortKey]);
  const percentiles = useMemo(() => {
    const maps: Record<string, Map<Record<string, string | number | null>, number>> = {};
    for (const column of selected?.columns ?? []) {
      if (!isImpactColumn(column.key)) continue;
      const ordered = [...(selected?.rows ?? [])].sort((left, right) => Number(left[column.key] ?? 0) - Number(right[column.key] ?? 0));
      maps[column.key] = new Map(ordered.map((row, index) => [row, ordered.length <= 1 ? 100 : Math.round((index / (ordered.length - 1)) * 99 + 1)]));
    }
    return maps;
  }, [selected]);

  if (!selectedMeta) return <div className="empty">Reconstruction data unavailable.</div>;

  function sort(next: string) {
    if (next === sortKey) setDirection((value) => value === "desc" ? "asc" : "desc");
    else {
      setSortKey(next);
      setDirection(next === "player" ? "asc" : "desc");
    }
  }

  function formatValue(key: string, value: string | number | null | undefined) {
    if (value == null) return "—";
    if (key === "player" || key === "team" || key === "source") return String(value);
    if (key === "minutes" || key === "exposure") return fmtInt(Number(value));
    return fmtRating(Number(value));
  }

  function shortMetric(value: string) {
    return value.replace(/^CourtSignal /, "").replace(/ reconstruction$/, "");
  }

  return <section className="replications-page">
    <header className="page-head">
      <div><p className="kicker">CourtSignal outputs only</p><h1>Reconstructions</h1></div>
    </header>

    <div className="replication-model-tabs" role="tablist" aria-label="Replication model">
      {metrics.map((item) => <button key={item} type="button" role="tab" aria-selected={item === selectedMeta.metric} onClick={() => setMetric(item)}>{shortMetric(item)}</button>)}
    </div>

    <div className="replication-controls">
      <div className="season-strip" aria-label="Season">
        {seasons.map((year) => <button type="button" key={year} aria-pressed={year === selectedMeta.season} onClick={() => setSeason(year)}>{year - 1}–{String(year).slice(2)}</button>)}
      </div>
      <label className="replication-search"><span className="visually-hidden">Search player</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search player" /></label>
    </div>

    <div className="replication-method-line">
      <div><strong>{selectedMeta.title}</strong><span>{selectedMeta.source ?? "CourtSignal reconstruction"}</span></div>
      <div><strong>{replication?.pearson == null ? "—" : replication.pearson.toFixed(3)}</strong><span>Pearson · {fmtInt(replication?.matched_rows ?? 0)} matched</span></div>
      <div><strong>{replication?.r_squared == null ? "—" : replication.r_squared.toFixed(3)}</strong><span>R² against source</span></div>
    </div>

    {!selected && <p className="empty" role="status">Loading reconstruction…</p>}
    {remote?.id === selectedMeta.id && remote.error && <p className="empty" role="alert">Reconstruction unavailable.</p>}
    {selected && <div className="table-wrap replication-table-wrap"><table className="data replication-table">
      <thead>
        {isRaptor && <tr className="reconstruction-groups"><th className="left">#</th><th colSpan={2} className="left">Player</th><th colSpan={3}>Box</th><th colSpan={3}>On/off</th><th colSpan={3}>RAPTOR</th><th>Sample</th></tr>}
        <tr><th className="left">#</th>{selected.columns.map((column, index) => <th className={index < 2 ? "left" : undefined} key={column.key} aria-sort={sortKey === column.key ? (direction === "asc" ? "ascending" : "descending") : "none"}><button type="button" onClick={() => sort(column.key)}>{column.label}</button></th>)}</tr>
      </thead>
      <tbody>{rows.map((row, index) => <tr key={`${row.player}-${index}`}>
        <td className="rank">{String(index + 1).padStart(2, "0")}</td>
        {selected.columns.map((column, columnIndex) => {
          const value = row[column.key];
          const impact = isImpactColumn(column.key) && typeof value === "number";
          return <td className={`${columnIndex < 2 ? "left" : ""} ${column.key === "player" ? "name" : ""} ${impact ? `metric-cell ${value >= 0 ? "positive" : "negative"}` : ""}`} key={column.key}>
            {impact ? <><b>{formatValue(column.key, value)}</b><small>{ordinalSuffix(percentiles[column.key]?.get(row) ?? 0)}</small></> : formatValue(column.key, value)}
          </td>;
        })}
      </tr>)}</tbody>
    </table></div>}
  </section>;
}

function isImpactColumn(key: string) {
  return /(offense|defense|net|raptor|bpm|pipm|box|onoff)/i.test(key);
}
