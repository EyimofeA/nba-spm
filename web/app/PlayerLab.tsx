"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Component = "net" | "offense" | "defense";
type ModelId = "aio" | "spm" | "rapm";
type Tab = "impact" | "roles" | "aging";
type Scope = "1y" | "3y" | "5y" | "2022-24" | "2025";
type AnnualRating = { Season: number; [key: string]: number };
type RollingRating = { window_end: number; window_seasons: number; net: number; offense: number; defense: number };
type Membership = { role_id: string; label: string; affinity: number };
type Role = { primary_role: string; confidence: number; memberships: Membership[] };
type RoleSeason = { Season: number; offense?: Role; defense?: Role };
type Interval = { low: number; high: number };
type UncertaintyComponent = { estimate: number; interval_80: Interval; interval_95: Interval; standard_error: number };
type UncertaintyScope = { seasons: number[]; rating: { uncertainty: { components: Record<Component, UncertaintyComponent> } } } | null;
type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  rolling: RollingRating[];
  roles: RoleSeason[];
  normal_rapm_uncertainty: Record<string, UncertaintyScope>;
};
type PlayerIndex = { id: number; name: string; shard: number };
type AgingRow = { Age: number; f_total: number; f_off: number; f_def: number; n: number | null };
type Catalog = {
  created_at: string;
  shards: number;
  catalog: {
    models: { id: ModelId; label: string; scopes: Scope[] }[];
    interval_scopes: Record<string, string>;
  };
  lineage: Record<string, string | Record<string, string> | null>;
  aging: { rows: AgingRow[] };
};
type Point = { label: string; value: number };

const formatRating = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const componentLabel: Record<Component, string> = { net: "Net", offense: "Off", defense: "Def" };

function ImpactChart({ points, interval }: { points: Point[]; interval?: UncertaintyComponent }) {
  if (!points.length) return <div className="empty">No rating.</div>;
  const width = 840;
  const height = 280;
  const pad = { left: 42, right: 24, top: 22, bottom: 38 };
  const values = points.map((point) => point.value);
  if (interval) values.push(interval.interval_95.low, interval.interval_95.high);
  const bound = Math.max(2, Math.ceil(Math.max(...values.map(Math.abs))));
  const x = (index: number) => points.length === 1
    ? width / 2
    : pad.left + index * (width - pad.left - pad.right) / (points.length - 1);
  const y = (value: number) => pad.top + (bound - value) * (height - pad.top - pad.bottom) / (bound * 2);
  const labelStep = Math.max(1, Math.ceil(points.length / 9));
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Player impact chart">
    {[-bound, 0, bound].map((tick) => <g key={tick}>
      <line className={tick === 0 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} />
      <text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick > 0 ? `+${tick}` : tick}</text>
    </g>)}
    {!interval && <polyline className="impact-line" points={points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ")} />}
    {points.map((point, index) => <g key={`${point.label}-${index}`}>
      {interval && <>
        <line className="error" x1={x(index)} x2={x(index)} y1={y(interval.interval_95.low)} y2={y(interval.interval_95.high)} />
        <line className="error faint" x1={x(index)} x2={x(index)} y1={y(interval.interval_80.low)} y2={y(interval.interval_80.high)} />
        <line className="cap" x1={x(index) - 10} x2={x(index) + 10} y1={y(interval.interval_95.low)} y2={y(interval.interval_95.low)} />
        <line className="cap" x1={x(index) - 10} x2={x(index) + 10} y1={y(interval.interval_95.high)} y2={y(interval.interval_95.high)} />
      </>}
      <circle className="impact-dot" cx={x(index)} cy={y(point.value)} r="4" />
      {(index % labelStep === 0 || index === points.length - 1) && <text className="axis" x={x(index)} y={height - 12} textAnchor="middle">{point.label}</text>}
    </g>)}
  </svg></div>;
}

function AgingChart({ rows }: { rows: AgingRow[] }) {
  const width = 840;
  const height = 280;
  const pad = { left: 42, right: 24, top: 22, bottom: 38 };
  const low = -10;
  const high = 1;
  const x = (age: number) => pad.left + (age - 19) * (width - pad.left - pad.right) / 21;
  const y = (value: number) => pad.top + (high - value) * (height - pad.top - pad.bottom) / (high - low);
  const line = (key: "f_total" | "f_off" | "f_def") => rows.map((row) => `${x(row.Age)},${y(row[key])}`).join(" ");
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Observed NBA aging curve">
    {[-10, -5, 0].map((tick) => <g key={tick}><line className={tick === 0 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick}</text></g>)}
    <polyline className="aging total" points={line("f_total")} />
    <polyline className="aging off" points={line("f_off")} />
    <polyline className="aging def" points={line("f_def")} />
    {[19, 22, 25, 28, 31, 34, 37, 40].map((age) => <text className="axis" key={age} x={x(age)} y={height - 12} textAnchor="middle">{age}</text>)}
  </svg></div>;
}

function RolePanel({ title, role }: { title: string; role?: Role }) {
  if (!role) return <section className="role-panel"><h3>{title}</h3><p className="muted">No role data.</p></section>;
  return <section className="role-panel">
    <div className="role-title"><span>{title}</span><strong>{role.primary_role}</strong></div>
    <div className="memberships">{role.memberships.map((item) => <div className="membership" key={item.role_id}>
      <span>{item.label}</span><div><i style={{ width: `${item.affinity * 100}%` }} /></div><b>{Math.round(item.affinity * 100)}%</b>
    </div>)}</div>
  </section>;
}

export function PlayerLab() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [index, setIndex] = useState<PlayerIndex[]>([]);
  const [player, setPlayer] = useState<Player | null>(null);
  const [query, setQuery] = useState("");
  const [model, setModel] = useState<ModelId>("aio");
  const [scope, setScope] = useState<Scope>("1y");
  const [component, setComponent] = useState<Component>("net");
  const [tab, setTab] = useState<Tab>("impact");
  const [roleSeason, setRoleSeason] = useState(2024);
  const [status, setStatus] = useState("Loading…");
  const shardCache = useRef(new Map<number, Record<string, Player>>());

  async function loadPlayer(id: number, shard = id % 32) {
    setStatus("Loading…");
    try {
      let data = shardCache.current.get(shard);
      if (!data) {
        const response = await fetch(`/data/ratings-${String(shard).padStart(2, "0")}.json`);
        if (!response.ok) throw new Error("rating shard unavailable");
        data = await response.json();
        shardCache.current.set(shard, data!);
      }
      const next = data[String(id)];
      if (!next) throw new Error("player unavailable");
      setPlayer(next);
      setRoleSeason(next.roles.at(-1)?.Season ?? 2024);
      setQuery("");
      setStatus("");
    } catch {
      setStatus("Data unavailable.");
    }
  }

  useEffect(() => {
    Promise.all([
      fetch("/data/catalog.json").then((response) => response.json()),
      fetch("/data/players.json").then((response) => response.json()),
    ]).then(([nextCatalog, nextIndex]: [Catalog, PlayerIndex[]]) => {
      setCatalog(nextCatalog);
      setIndex(nextIndex);
      const fallback = nextIndex.find((item) => item.id === 2544) ?? nextIndex[0];
      if (fallback) loadPlayer(fallback.id, fallback.shard);
    }).catch(() => setStatus("Data unavailable."));
  }, []);

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return [];
    return index.filter((item) => item.name.toLocaleLowerCase().includes(needle)).slice(0, 7);
  }, [index, query]);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    if (matches[0]) loadPlayer(matches[0].id, matches[0].shard);
  }

  const modelConfig = catalog?.catalog.models.find((item) => item.id === model);
  const scopes = useMemo(() => modelConfig?.scopes ?? ["1y"], [modelConfig]);

  const interval = useMemo(() => {
    if (!player || !catalog || model !== "rapm") return undefined;
    const key = catalog.catalog.interval_scopes[scope];
    return key ? player.normal_rapm_uncertainty[key]?.rating.uncertainty.components[component] : undefined;
  }, [player, catalog, model, scope, component]);

  const points = useMemo<Point[]>(() => {
    if (!player) return [];
    if (interval) return [{ label: scope, value: interval.estimate }];
    if (scope === "1y") {
      const prefix = model === "aio" ? "aio" : model === "spm" ? "spm_raw" : "normal_rapm";
      return player.annual.map((row) => ({ label: String(row.Season), value: Number(row[`${prefix}_${component}`]) }));
    }
    const years = Number(scope[0]);
    return player.rolling.filter((row) => row.window_seasons === years).map((row) => ({ label: String(row.window_end), value: row[component] }));
  }, [player, model, scope, component, interval]);

  const latest = points.at(-1);
  const latestAnnual = player?.annual.at(-1);
  const roles = player?.roles.find((row) => row.Season === roleSeason);

  return <main>
    <header>
      <a className="brand" href="#top">NBA Impact</a>
      <form className="search" onSubmit={submitSearch}>
        <input aria-label="Find player" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find player" />
        {matches.length > 0 && <div className="results">{matches.map((item) => <button type="button" key={item.id} onClick={() => loadPlayer(item.id, item.shard)}>{item.name}</button>)}</div>}
      </form>
    </header>
    <nav className="tabs" aria-label="View">
      {(["impact", "roles", "aging"] as Tab[]).map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}
    </nav>
    {status && <p className="status">{status}</p>}
    {player && <div className="content" id="top">
      {tab === "impact" && <>
        <section className="title-row"><div><p className="kicker">PLAYER</p><h1>{player.PLAYER_NAME}</h1></div><div className="headline"><span>{latest?.label ?? "—"}</span><strong>{latest ? formatRating(latest.value) : "—"}</strong></div></section>
        <section className="controls" aria-label="Rating controls">
          <label>Model<select value={model} onChange={(event) => {
            const next = event.target.value as ModelId;
            const nextScopes = catalog?.catalog.models.find((item) => item.id === next)?.scopes ?? ["1y"];
            setModel(next);
            if (!nextScopes.includes(scope)) setScope(nextScopes[0]);
          }}>{catalog?.catalog.models.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          <label>Window<select value={scope} onChange={(event) => setScope(event.target.value as Scope)}>{scopes.map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}</select></label>
          <label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label>
        </section>
        <ImpactChart points={points} interval={interval} />
        {interval ? <section className="intervals"><div><span>80%</span><strong>{formatRating(interval.interval_80.low)} – {formatRating(interval.interval_80.high)}</strong></div><div><span>95%</span><strong>{formatRating(interval.interval_95.low)} – {formatRating(interval.interval_95.high)}</strong></div></section> : <p className="muted compact">Interval not estimated for this metric.</p>}
        {model === "aio" && latestAnnual && <section className="equation" aria-label="AIO decomposition"><div><span>SPM</span><b>{formatRating(latestAnnual[`spm_center_${component}`])}</b></div><i>+</i><div><span>RAPM update</span><b>{formatRating(latestAnnual[`rapm_update_${component}`])}</b></div><i>=</i><div><span>AIO</span><b>{formatRating(latestAnnual[`aio_${component}`])}</b></div></section>}
      </>}
      {tab === "roles" && <>
        <section className="title-row"><div><p className="kicker">ROLES</p><h1>{player.PLAYER_NAME}</h1></div><label className="year">Season<select value={roleSeason} onChange={(event) => setRoleSeason(Number(event.target.value))}>{player.roles.map((row) => <option key={row.Season}>{row.Season}</option>)}</select></label></section>
        <p className="muted compact">Soft membership. Position excluded.</p>
        <div className="role-grid"><RolePanel title="Offense" role={roles?.offense} /><RolePanel title="Defense" role={roles?.defense} /></div>
      </>}
      {tab === "aging" && catalog && <>
        <section className="title-row"><div><p className="kicker">AGING</p><h1>Observed curve</h1></div><div className="mini-legend"><span className="total-key">Net</span><span className="off-key">Off</span><span className="def-key">Def</span></div></section>
        <AgingChart rows={catalog.aging.rows} />
        <p className="muted compact">1997–2024 annual RAPM panel. Descriptive, not a player forecast.</p>
      </>}
      <details><summary>Details</summary><div className="details"><p>SPM trains on player-seasons from 2014–24. Historical AIO centers use leave-one-season-out SPM. One-season RAPM uses fixed 3000 / 3000 / 300 penalties.</p><p>SPM and AIO intervals are not calibrated. RAPM intervals appear only for exact 2022–24 and 2025 scopes.</p><code>{String(catalog?.lineage.annual_run_id ?? "")}</code></div></details>
    </div>}
  </main>;
}
