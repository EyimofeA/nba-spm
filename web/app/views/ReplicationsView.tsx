"use client";

import { useMemo, useState } from "react";
import { RapmLabPayload } from "../lib/data";
import { fmtInt, fmtRating } from "../lib/viz";

type SortKey = string;

export function ReplicationsView({ lab }: { lab: RapmLabPayload | null }) {
  const boards = lab?.replication_leaderboards ?? [];
  const [boardId, setBoardId] = useState(boards[0]?.id ?? "");
  const [sortKey, setSortKey] = useState<SortKey>("net");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const selected = boards.find((board) => board.id === boardId) ?? boards[0];
  const replication = selected
    ? lab?.replications.find((row) => row.metric === selected.metric)
    : undefined;
  const rows = useMemo(() => {
    if (!selected) return [];
    const sign = direction === "asc" ? 1 : -1;
    return [...selected.rows].sort((a, b) => {
      if (sortKey === "player") return a.player.localeCompare(b.player) * sign;
      const left = a[sortKey];
      const right = b[sortKey];
      if (typeof left === "string" || typeof right === "string") {
        return String(left ?? "").localeCompare(String(right ?? "")) * sign;
      }
      return ((Number(left) || 0) - (Number(right) || 0)) * sign;
    });
  }, [direction, selected, sortKey]);

  function sort(next: SortKey) {
    if (next === sortKey) setDirection((value) => value === "desc" ? "asc" : "desc");
    else {
      setSortKey(next);
      setDirection(next === "player" ? "asc" : "desc");
    }
  }

  if (!lab || !selected) return <div className="empty">Replication data unavailable.</div>;

  function formatValue(key: string, value: string | number | null | undefined) {
    if (value == null) return "—";
    if (key === "player" || key === "team") return String(value);
    if (key === "minutes" || key === "exposure") return fmtInt(Number(value));
    return fmtRating(Number(value));
  }

  return <>
    <div className="page-head">
      <div><p className="kicker">Independent reference checks</p><h1>Replications</h1></div>
      <span className="meta">Exact outputs and labeled reconstructions</span>
    </div>

    <div className="replication-summary" aria-label="Replication correlations">
      {lab.replications.map((row) => <article className="tile" key={`${row.metric}-${row.build}`}>
        <div className="tile-label">{row.metric}</div>
        <div className="tile-value">{row.pearson == null ? "—" : row.pearson.toFixed(3)}</div>
        <div className="tile-sub">Pearson · {fmtInt(row.matched_rows)} matched</div>
        <span className={`lab-status ${row.status === "exact_public_output" ? "won" : "estimate"}`}>{row.status.replaceAll("_", " ")}</span>
      </article>)}
    </div>

    <div className="filters">
      <label className="field"><span>Model and season</span><select value={selected.id} onChange={(event) => setBoardId(event.target.value)}>
        {boards.map((board) => <option value={board.id} key={board.id}>{board.title}</option>)}
      </select></label>
    </div>

    <section>
      <div className="section-head"><div><p className="kicker">Leaderboard</p><h2>{selected.title}</h2></div>
        <span className="meta">{replication?.pearson == null ? "Reference correlation unavailable" : `Pearson ${replication.pearson.toFixed(3)}`}</span>
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr>{selected.columns.map((column, index) => <th className={index < 2 ? "left" : undefined} key={column.key} aria-sort={sortKey === column.key ? (direction === "asc" ? "ascending" : "descending") : "none"}><button type="button" onClick={() => sort(column.key)}>{column.label}</button></th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={`${row.player}-${index}`}>{selected.columns.map((column, columnIndex) => <td className={`${columnIndex < 2 ? "left" : ""} ${column.key === "player" ? "name" : ""} ${column.key === "net" || column.key === "raptor" ? "headline" : ""}`} key={column.key}>{formatValue(column.key, row[column.key])}</td>)}</tr>)}</tbody>
      </table></div>
    </section>
  </>;
}
