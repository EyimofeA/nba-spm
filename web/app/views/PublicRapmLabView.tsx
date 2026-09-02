"use client";

import { useEffect, useMemo, useState } from "react";
import { fmtRating } from "../lib/viz";

type Period = { id: string; label: string; url: string; rows: number };
type Estimand = { id: string; title: string; unit: string; note: string; periods: Period[] };
type Catalog = { schema: string; estimands: Estimand[] };
type Row = Record<string, string | number | null> & { PLAYER_NAME?: string };
type Sort = "net" | "offense" | "defense" | "name";

const value = (row: Row, side: "offense" | "defense" | "net") => {
  const direct = row[side];
  const annual = row[`rapm_${side}`];
  return typeof direct === "number" ? direct : typeof annual === "number" ? annual : 0;
};

export function PublicRapmLabView() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [estimandId, setEstimandId] = useState("annual");
  const [periodId, setPeriodId] = useState("2026");
  const [rows, setRows] = useState<Row[]>([]);
  const [sort, setSort] = useState<Sort>("net");
  const [ascending, setAscending] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    fetch("/data/rapm/catalog.json")
      .then((response) => {
        if (!response.ok) throw new Error("RAPM catalog unavailable");
        return response.json() as Promise<Catalog>;
      })
      .then(setCatalog)
      .catch(() => setCatalog(null));
  }, []);

  const estimand = catalog?.estimands.find((item) => item.id === estimandId) ?? catalog?.estimands[0];
  const period = estimand?.periods.find((item) => item.id === periodId) ?? estimand?.periods.at(-1);

  useEffect(() => {
    if (!period) return;
    fetch(period.url)
      .then((response) => {
        if (!response.ok) throw new Error("RAPM leaderboard unavailable");
        return response.json() as Promise<Row[]>;
      })
      .then(setRows)
      .catch(() => setRows([]));
  }, [period]);

  const visible = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const direction = ascending ? 1 : -1;
    return rows
      .filter((row) => !needle || String(row.PLAYER_NAME ?? "").toLocaleLowerCase().includes(needle))
      .sort((left, right) => {
        const delta = sort === "name"
          ? String(left.PLAYER_NAME ?? "").localeCompare(String(right.PLAYER_NAME ?? ""))
          : value(left, sort) - value(right, sort);
        return delta * direction;
      })
      .slice(0, 200);
  }, [ascending, query, rows, sort]);

  const setSorting = (next: Sort) => {
    if (next === sort) setAscending(!ascending);
    else { setSort(next); setAscending(next === "name"); }
  };

  if (!catalog || !estimand || !period) return <p className="empty">RAPM Lab unavailable.</p>;

  return (
    <section aria-labelledby="rapm-lab-heading">
      <header className="page-head">
        <div><p className="kicker">Research</p><h1 id="rapm-lab-heading">RAPM Lab</h1></div>
      </header>
      <div className="filters">
        <label className="field"><span>Estimand</span><select value={estimand.id} onChange={(event) => {
          const next = catalog.estimands.find((item) => item.id === event.target.value);
          setEstimandId(event.target.value);
          if (next?.periods.length) setPeriodId(next.periods.at(-1)!.id);
        }}>{catalog.estimands.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
        <label className="field"><span>Window</span><select value={period.id} onChange={(event) => setPeriodId(event.target.value)}>{estimand.periods.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label className="field"><span>Find</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Player or unit" /></label>
      </div>
      <section className="card leaderboard-panel">
        <div className="card-head"><div><p className="kicker">{period.label}</p><h2>{estimand.title}</h2></div><span className="meta">{estimand.unit}</span></div>
        {estimand.note && <p className="note">{estimand.note}</p>}
        <div className="table-wrap"><table className="data">
          <thead><tr><th className="left"><button type="button" onClick={() => setSorting("name")}>Player</button></th><th><button type="button" onClick={() => setSorting("offense")}>Off</button></th><th><button type="button" onClick={() => setSorting("defense")}>Def</button></th><th><button type="button" onClick={() => setSorting("net")}>Net</button></th></tr></thead>
          <tbody>{visible.map((row, index) => <tr key={`${row.PLAYER_NAME}-${index}`}><th className="left">{row.PLAYER_NAME ?? "—"}</th><td>{fmtRating(value(row, "offense"))}</td><td>{fmtRating(value(row, "defense"))}</td><td>{fmtRating(value(row, "net"))}</td></tr>)}</tbody>
        </table></div>
      </section>
    </section>
  );
}
