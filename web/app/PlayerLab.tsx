"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Component = "net" | "offense" | "defense";
type ModelId = "aio" | "spm" | "rapm";
type Tab = "about" | "ratings" | "player" | "roles" | "research";
type RoleSide = "offense" | "defense";
type RoleMode = "stable" | "raw";
type Membership = { role_id: string; label: string; affinity: number };
type Role = {
  primary_role: string;
  confidence: number;
  memberships: Membership[];
  stabilized_primary_role?: string;
  stabilized_confidence?: number;
  stabilized_memberships?: Membership[];
};
type RoleSeason = { Season: number; offense?: Role; defense?: Role };
type AnnualRating = { Season: number; [key: string]: number | boolean | null };
type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  roles: RoleSeason[];
};
type PlayerIndex = { id: number; name: string; shard: number };
type LeaderboardRow = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  Poss_Off: number;
  Poss_Def: number;
  offense_stable_role: string | null;
  defense_stable_role: string | null;
  [key: string]: string | number | null;
};
type RolePoint = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  x: number;
  y: number;
  raw_role: string;
  stable_role: string;
};
type AgingRow = {
  Age: number;
  n: number | null;
  f_total?: number;
  f_off?: number;
  f_def?: number;
  change_net?: number;
  change_offense?: number;
  change_defense?: number;
  [key: string]: number | null | undefined;
};
type Calibration = {
  intercept: number;
  slope: number;
  display_range_01_99: [number, number];
};
type Stability = {
  selected_current_weight: number;
  later_raw_same_role_rate: number;
  later_stable_same_role_rate: number;
  later_stable_raw_disagreement_rate: number;
};
type Catalog = {
  shards: number;
  catalog: {
    seasons: number[];
    role_seasons: Record<RoleSide, number[]>;
    models: { id: ModelId; label: string }[];
  };
  methods: {
    aio_equation: string;
    rapm_update_note: string;
    spm_calibration: Record<Component, Calibration>;
    role_stabilization: Record<RoleSide, Stability>;
  };
  aging: Record<"rapm" | "aio", { coverage: string; rows: AgingRow[] }>;
  lineage: Record<string, string | null>;
};
type Point = { label: string; value: number };

const formatRating = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const componentLabel: Record<Component, string> = { net: "Net", offense: "Off", defense: "Def" };
const roleColors = ["#e85d32", "#2467d5", "#22a06b", "#8c56c2", "#c99a12", "#9c4c68"];

function LineChart({ points, label }: { points: Point[]; label: string }) {
  if (!points.length) return <div className="empty">No rating.</div>;
  const width = 900, height = 270;
  const pad = { left: 46, right: 24, top: 24, bottom: 38 };
  const bound = Math.max(1, Math.ceil(Math.max(...points.map((point) => Math.abs(point.value)))));
  const x = (index: number) => points.length === 1 ? width / 2 : pad.left + index * (width - pad.left - pad.right) / (points.length - 1);
  const y = (value: number) => pad.top + (bound - value) * (height - pad.top - pad.bottom) / (bound * 2);
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
    {[-bound, 0, bound].map((tick) => <g key={tick}><line className={tick === 0 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick > 0 ? `+${tick}` : tick}</text></g>)}
    <polyline className="impact-line" points={points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ")} />
    {points.map((point, index) => <g key={`${point.label}-${index}`}><circle className="impact-dot" cx={x(index)} cy={y(point.value)} r="4" /><text className="axis" x={x(index)} y={height - 12} textAnchor="middle">{point.label}</text></g>)}
  </svg></div>;
}

function AgingChart({ rows, valueKey, label }: { rows: AgingRow[]; valueKey: string; label: string }) {
  const points = rows.flatMap((row) => typeof row[valueKey] === "number" ? [{ age: row.Age, value: Number(row[valueKey]) }] : []);
  if (!points.length) return <div className="empty">No curve.</div>;
  const width = 900, height = 270;
  const pad = { left: 48, right: 24, top: 24, bottom: 38 };
  const minAge = Math.min(...points.map((point) => point.age));
  const maxAge = Math.max(...points.map((point) => point.age));
  const rawLow = Math.min(0, ...points.map((point) => point.value));
  const rawHigh = Math.max(0, ...points.map((point) => point.value));
  const margin = Math.max(0.2, (rawHigh - rawLow) * 0.08);
  const low = rawLow - margin, high = rawHigh + margin;
  const x = (age: number) => pad.left + (age - minAge) * (width - pad.left - pad.right) / Math.max(1, maxAge - minAge);
  const y = (value: number) => pad.top + (high - value) * (height - pad.top - pad.bottom) / (high - low);
  const ticks = [low, 0, high];
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
    {ticks.map((tick, index) => <g key={index}><line className={Math.abs(tick) < 1e-9 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick.toFixed(1)}</text></g>)}
    <polyline className="aging total" points={points.map((point) => `${x(point.age)},${y(point.value)}`).join(" ")} />
    {points.map((point) => <circle className="aging-dot" key={point.age} cx={x(point.age)} cy={y(point.value)} r="3"><title>{`Age ${point.age}: ${formatRating(point.value)}`}</title></circle>)}
    {points.filter((_, index) => index % 3 === 0).map((point) => <text className="axis" key={point.age} x={x(point.age)} y={height - 12} textAnchor="middle">{point.age}</text>)}
  </svg></div>;
}

function shownRole(role: Role | undefined, mode: RoleMode): Role | undefined {
  if (!role || mode === "raw" || !role.stabilized_memberships) return role;
  return {
    primary_role: role.stabilized_primary_role ?? role.primary_role,
    confidence: role.stabilized_confidence ?? role.confidence,
    memberships: role.stabilized_memberships,
  };
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

function RoleScatter({ points, mode, onPlayer }: { points: RolePoint[]; mode: RoleMode; onPlayer: (id: number) => void }) {
  if (!points.length) return <div className="empty">No role map.</div>;
  const width = 900, height = 520, pad = 34;
  const xs = points.map((point) => point.x), ys = points.map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value: number) => pad + (value - minX) * (width - pad * 2) / Math.max(0.001, maxX - minX);
  const y = (value: number) => height - pad - (value - minY) * (height - pad * 2) / Math.max(0.001, maxY - minY);
  const roles = [...new Set(points.map((point) => mode === "stable" ? point.stable_role : point.raw_role))].sort();
  const color = new Map(roles.map((role, index) => [role, roleColors[index % roleColors.length]]));
  return <><div className="role-legend">{roles.map((role) => <span key={role}><i style={{ background: color.get(role) }} />{role}</span>)}</div><div className="chart scatter"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Player role cluster map">
    <line className="grid" x1={width / 2} x2={width / 2} y1={pad} y2={height - pad} /><line className="grid" x1={pad} x2={width - pad} y1={height / 2} y2={height / 2} />
    {points.map((point) => { const role = mode === "stable" ? point.stable_role : point.raw_role; return <circle key={point.PLAYER_ID} className="role-dot" cx={x(point.x)} cy={y(point.y)} r="5" fill={color.get(role)} tabIndex={0} onClick={() => onPlayer(point.PLAYER_ID)}><title>{`${point.PLAYER_NAME} · ${point.TEAM_ABBREVIATION ?? "—"} · ${role}`}</title></circle>; })}
    <text className="axis" x={width - pad} y={height - 8} textAnchor="end">Role axis 1</text><text className="axis" x={pad} y={18}>Role axis 2</text>
  </svg></div></>;
}

export function PlayerLab() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [index, setIndex] = useState<PlayerIndex[]>([]);
  const [tab, setTab] = useState<Tab>("about");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("Loading…");
  const [player, setPlayer] = useState<Player | null>(null);
  const [model, setModel] = useState<ModelId>("aio");
  const [component, setComponent] = useState<Component>("net");
  const [season, setSeason] = useState("2024");
  const [team, setTeam] = useState("All");
  const [order, setOrder] = useState<"desc" | "asc">("desc");
  const [sortKey, setSortKey] = useState("rating");
  const [tableView, setTableView] = useState<"compact" | "full">("compact");
  const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [roleMode, setRoleMode] = useState<RoleMode>("stable");
  const [roleYear, setRoleYear] = useState(2024);
  const [rolePoints, setRolePoints] = useState<RolePoint[]>([]);
  const [playerRoleMode, setPlayerRoleMode] = useState<RoleMode>("stable");
  const [roleSeason, setRoleSeason] = useState(2024);
  const [agingSource, setAgingSource] = useState<"rapm" | "aio">("rapm");
  const [agingView, setAgingView] = useState<"level" | "change">("change");
  const shardCache = useRef(new Map<number, Record<string, Player>>());
  const seasonCache = useRef(new Map<number, LeaderboardRow[]>());
  const roleCache = useRef(new Map<string, RolePoint[]>());

  async function loadSeason(nextSeason: string, availableSeasons?: number[]) {
    const allSeasons = availableSeasons ?? catalog?.catalog.seasons;
    if (!allSeasons) return;
    setStatus("Loading…");
    try {
      const years = nextSeason === "All" ? allSeasons : [Number(nextSeason)];
      const frames = await Promise.all(years.map(async (year) => {
        const cached = seasonCache.current.get(year);
        if (cached) return cached;
        const response = await fetch(`/data/leaderboard-${year}.json`);
        if (!response.ok) throw new Error("season unavailable");
        const rows: LeaderboardRow[] = await response.json();
        seasonCache.current.set(year, rows);
        return rows;
      }));
      setLeaderboard(frames.flat()); setTeam("All"); setStatus("");
    } catch { setStatus("Data unavailable."); }
  }

  async function loadPlayer(id: number, open = true) {
    setStatus("Loading…");
    try {
      const shard = index.find((item) => item.id === id)?.shard ?? id % (catalog?.shards ?? 32);
      let data = shardCache.current.get(shard);
      if (!data) {
        const response = await fetch(`/data/ratings-${String(shard).padStart(2, "0")}.json`);
        if (!response.ok) throw new Error("player unavailable");
        data = await response.json(); shardCache.current.set(shard, data!);
      }
      const next = data[String(id)];
      if (!next) throw new Error("player unavailable");
      setPlayer(next); setRoleSeason(next.roles.at(-1)?.Season ?? 2024); setQuery(""); setStatus("");
      if (open) setTab("player");
    } catch { setStatus("Data unavailable."); }
  }

  async function loadRoleMap(side: RoleSide, year: number) {
    const key = `${side}-${year}`; setStatus("Loading…");
    try {
      let rows = roleCache.current.get(key);
      if (!rows) {
        const response = await fetch(`/data/roles-${side}-${year}.json`);
        if (!response.ok) throw new Error("role map unavailable");
        rows = await response.json(); roleCache.current.set(key, rows!);
      }
      setRolePoints(rows); setStatus("");
    } catch { setStatus("Data unavailable."); }
  }

  useEffect(() => {
    Promise.all([
      fetch("/data/catalog.json").then((response) => response.json()),
      fetch("/data/players.json").then((response) => response.json()),
    ]).then(([nextCatalog, nextIndex]: [Catalog, PlayerIndex[]]) => {
      const latest = Math.max(...nextCatalog.catalog.seasons);
      setCatalog(nextCatalog); setIndex(nextIndex); setSeason(String(latest)); setRoleYear(latest); setStatus("");
    }).catch(() => setStatus("Data unavailable."));
  }, []);

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return needle ? index.filter((item) => item.name.toLocaleLowerCase().includes(needle)).slice(0, 7) : [];
  }, [index, query]);

  function submitSearch(event: FormEvent) { event.preventDefault(); if (matches[0]) loadPlayer(matches[0].id); }
  const valuePrefix = model === "spm" ? "spm_raw" : model === "rapm" ? "normal_rapm" : "aio";
  const ratingKey = `${valuePrefix}_${component}`;
  const teams = useMemo(() => [...new Set(leaderboard.map((row) => row.TEAM_ABBREVIATION).filter((value): value is string => Boolean(value)))].sort(), [leaderboard]);
  const sortedRows = useMemo(() => {
    const rows = team === "All" ? [...leaderboard] : leaderboard.filter((row) => row.TEAM_ABBREVIATION === team);
    const key = sortKey === "rating" ? ratingKey : sortKey;
    rows.sort((a, b) => {
      const av = a[key], bv = b[key];
      const comparison = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
      return (order === "asc" ? comparison : -comparison) || b.Season - a.Season || a.PLAYER_NAME.localeCompare(b.PLAYER_NAME);
    });
    return rows;
  }, [leaderboard, team, sortKey, ratingKey, order]);
  const points = useMemo<Point[]>(() => player?.annual.map((row) => ({ label: String(row.Season), value: Number(row[ratingKey]) })) ?? [], [player, ratingKey]);
  const latestAnnual = player?.annual.at(-1);
  const currentRoles = player?.roles.find((row) => row.Season === roleSeason);
  const agingRows = catalog?.aging[agingSource].rows ?? [];
  const agingKey = agingView === "change" ? `change_${component}` : component === "net" ? "f_total" : component === "offense" ? "f_off" : "f_def";

  function sortBy(key: string) { if (sortKey === key) setOrder(order === "desc" ? "asc" : "desc"); else { setSortKey(key); setOrder("desc"); } }

  return <main id="top">
    <header><a className="brand" href="#top" onClick={() => setTab("about")}>NBA Impact</a><form className="search" onSubmit={submitSearch}><input aria-label="Find player" placeholder="Find player" value={query} onChange={(event) => setQuery(event.target.value)} />{matches.length > 0 && <div className="results">{matches.map((item) => <button type="button" key={item.id} onClick={() => loadPlayer(item.id)}>{item.name}</button>)}</div>}</form></header>
    <nav className="tabs" aria-label="Sections">{(["about", "ratings", "player", "roles", "research"] as Tab[]).map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => { setTab(item); if (item === "ratings" && leaderboard.length === 0) loadSeason(season); if (item === "roles") loadRoleMap(roleSide, roleYear); }}>{item}</button>)}</nav>
    {status && <p className="status">{status}</p>}
    <div className="content">
      {tab === "about" && <>
        <section className="intro-hero"><p className="kicker">NBA IMPACT LAB</p><h1>Player impact,<br />from stats to lineups.</h1><p>Compare three views of a player’s offense, defense and total value. Then inspect how the estimate was built.</p><div className="intro-actions"><button onClick={() => { setTab("ratings"); loadSeason(season); }}>Explore ratings</button><button className="secondary" onClick={() => { setTab("roles"); loadRoleMap(roleSide, roleYear); }}>Explore roles</button></div><div className="release-line"><span>2017–24</span><span>4,341 player-seasons</span><span>Points per 100 possessions</span></div></section>
        <section className="model-cards"><article><span>01</span><h2>SPM</h2><p>What a player’s box, tracking, playtype and matchup profile predicts.</p></article><article><span>02</span><h2>RAPM</h2><p>What changed on the scoreboard after controlling for all ten players.</p></article><article><span>03</span><h2>AIO</h2><p>SPM starts the estimate. Possession evidence updates it in one joint fit.</p></article></section>
        <section className="method-section"><div className="method-title"><p className="kicker">PROCESS</p><h2>How it works</h2></div><ol className="method-steps"><li><b>Build one row per player-season.</b><span>Regular-season box, shooting, tracking, hustle, playtype, zTS and scorer-adjusted defensive matchup features.</span></li><li><b>Estimate normal RAPM.</b><span>Each possession records the ten terminal-lineup players and points scored. Ridge regression separates offense and defense.</span></li><li><b>Train SPM on RAPM.</b><span>The offense model learns nonlinear effects. The defense model stays linear. A rated season is excluded from its own SPM training fit.</span></li><li><b>Refit RAPM around SPM.</b><span>The SPM prediction becomes the coefficient center. The lineup evidence can move the rating away from that center.</span></li></ol></section>
        <section className="rebuild"><div><p className="kicker">REBUILD SPEC</p><h2>Enough detail to reproduce it</h2></div><div className="spec-grid"><article><h3>Target</h3><p>Single-season regular-season normal RAPM. Terminal lineup. Zero prior. Penalties: offense 3000, defense 3000, home 300.</p></article><article><h3>Rows + weights</h3><p>Training seasons 2014–24. Minimum 1,000 possessions per side. Player-season weight is the square root of the smaller side’s possessions.</p></article><article><h3>Offense</h3><p>127 features. Median imputation. Histogram GBM: 250 trees, learning rate .03, 7 leaves, 30-row leaf minimum, L2 = 1.</p></article><article><h3>Defense</h3><p>68 features. Median imputation plus missing flags, standardization, ridge alpha 3000. Includes eight scorer-adjusted matchup factors.</p></article><article><h3>Validation</h3><p>Leave one season out for every public year from 2017–24. Final standalone SPM refits all labeled seasons. No age, height, position, minutes, games or on/off input.</p></article><article><h3>AIO equation</h3><p><code>argmin ||y − Xβ||² + 3000||βoff − μoff||² + 3000||βdef − μdef||²</code><br />μ is cross-fitted SPM. AIO is β. RAPM update is β − μ.</p></article></div></section>
      </>}
      {tab === "ratings" && <>
        <section className="section-head"><div><p className="kicker">RATINGS</p><h1>Player impact</h1></div><span>{sortedRows.length.toLocaleString()} rows</span></section>
        <section className="controls table-controls">
          <label>Year<select value={season} onChange={(event) => { setSeason(event.target.value); loadSeason(event.target.value); }}><option>All</option>{[...(catalog?.catalog.seasons ?? [])].reverse().map((year) => <option key={year}>{year}</option>)}</select></label>
          <label>Model<select value={model} onChange={(event) => setModel(event.target.value as ModelId)}>{catalog?.catalog.models.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          <label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label>
          <label>Team<select value={team} onChange={(event) => setTeam(event.target.value)}><option>All</option>{teams.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label>Order<select value={order} onChange={(event) => setOrder(event.target.value as "desc" | "asc")}><option value="desc">High first</option><option value="asc">Low first</option></select></label>
          <label>View<select value={tableView} onChange={(event) => setTableView(event.target.value as "compact" | "full")}><option value="compact">Compact</option><option value="full">Full</option></select></label>
        </section>
        <div className="ratings-table-wrap"><table className="ratings-table"><thead><tr><th>#</th><th><button onClick={() => sortBy("PLAYER_NAME")}>Player</button></th><th><button onClick={() => sortBy("TEAM_ABBREVIATION")}>Team</button></th><th><button onClick={() => sortBy("Season")}>Year</button></th>{tableView === "compact" ? <><th><button onClick={() => sortBy(`${valuePrefix}_offense`)}>Off</button></th><th><button onClick={() => sortBy(`${valuePrefix}_defense`)}>Def</button></th><th><button onClick={() => sortBy(`${valuePrefix}_net`)}>Net</button></th></> : <><th><button onClick={() => sortBy("spm_raw_net")}>SPM</button></th><th><button onClick={() => sortBy("normal_rapm_net")}>RAPM</button></th><th><button onClick={() => sortBy("rapm_update_net")}>Update</button></th><th><button onClick={() => sortBy("aio_net")}>AIO</button></th><th>Off role</th><th>Def role</th></>}<th><button onClick={() => sortBy("Poss_Off")}>Poss</button></th></tr></thead><tbody>{sortedRows.map((row, indexRow) => <tr key={`${row.PLAYER_ID}-${row.Season}`} onClick={() => loadPlayer(row.PLAYER_ID)}><td>{indexRow + 1}</td><td className="player-cell">{row.PLAYER_NAME}</td><td>{row.TEAM_ABBREVIATION ?? "—"}</td><td>{row.Season}</td>{tableView === "compact" ? <><td>{formatRating(Number(row[`${valuePrefix}_offense`]))}</td><td>{formatRating(Number(row[`${valuePrefix}_defense`]))}</td><td className="rating-cell">{formatRating(Number(row[`${valuePrefix}_net`]))}</td></> : <><td>{formatRating(Number(row.spm_raw_net))}</td><td>{formatRating(Number(row.normal_rapm_net))}</td><td>{formatRating(Number(row.rapm_update_net))}</td><td className="rating-cell">{formatRating(Number(row.aio_net))}</td><td>{row.offense_stable_role ?? "—"}</td><td>{row.defense_stable_role ?? "—"}</td></>}<td>{Math.round(Math.min(row.Poss_Off, row.Poss_Def)).toLocaleString()}</td></tr>)}</tbody></table></div>
      </>}
      {tab === "player" && (player ? <>
        <section className="section-head"><div><p className="kicker">PLAYER</p><h1>{player.PLAYER_NAME}</h1></div><strong className="big-rating">{points.at(-1) ? formatRating(points.at(-1)!.value) : "—"}</strong></section>
        <section className="controls"><label>Model<select value={model} onChange={(event) => setModel(event.target.value as ModelId)}>{catalog?.catalog.models.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label></section>
        <LineChart points={points} label={`${player.PLAYER_NAME} annual impact`} />
        {model === "aio" && latestAnnual && <section className="equation"><div><span>SPM center</span><b>{formatRating(Number(latestAnnual[`spm_center_${component}`]))}</b></div><i>+</i><div><span>RAPM update</span><b>{formatRating(Number(latestAnnual[`rapm_update_${component}`]))}</b></div><i>=</i><div><span>AIO</span><b>{formatRating(Number(latestAnnual[`aio_${component}`]))}</b></div></section>}
        <section className="subhead"><h2>Roles</h2><div className="inline-controls"><select aria-label="Role season" value={roleSeason} onChange={(event) => setRoleSeason(Number(event.target.value))}>{player.roles.map((row) => <option key={row.Season}>{row.Season}</option>)}</select><select aria-label="Role mode" value={playerRoleMode} onChange={(event) => setPlayerRoleMode(event.target.value as RoleMode)}><option value="stable">Stable</option><option value="raw">Raw</option></select></div></section>
        <div className="role-grid"><RolePanel title="Offense" role={shownRole(currentRoles?.offense, playerRoleMode)} /><RolePanel title="Defense" role={shownRole(currentRoles?.defense, playerRoleMode)} /></div>
      </> : <div className="empty">Choose a player.</div>)}
      {tab === "roles" && catalog && <>
        <section className="section-head"><div><p className="kicker">ROLES</p><h1>Role map</h1></div><span>{rolePoints.length.toLocaleString()} players</span></section>
        <section className="controls"><label>Side<select value={roleSide} onChange={(event) => { const next = event.target.value as RoleSide; const year = Math.max(...catalog.catalog.role_seasons[next]); setRoleSide(next); setRoleYear(year); loadRoleMap(next, year); }}><option value="offense">Offense</option><option value="defense">Defense</option></select></label><label>Year<select value={roleYear} onChange={(event) => { const year = Number(event.target.value); setRoleYear(year); loadRoleMap(roleSide, year); }}>{[...catalog.catalog.role_seasons[roleSide]].reverse().map((year) => <option key={year}>{year}</option>)}</select></label><label>Roles<select value={roleMode} onChange={(event) => setRoleMode(event.target.value as RoleMode)}><option value="stable">Stable</option><option value="raw">Raw</option></select></label></section>
        <RoleScatter points={rolePoints} mode={roleMode} onPlayer={(id) => loadPlayer(id)} />
      </>}
      {tab === "research" && catalog && <>
        <section className="section-head"><div><p className="kicker">RESEARCH</p><h1>Aging</h1></div></section>
          <section className="controls"><label>Metric<select value={agingSource} onChange={(event) => { const next = event.target.value as "rapm" | "aio"; setAgingSource(next); if (next === "aio") setAgingView("change"); }}><option value="rapm">RAPM</option><option value="aio">AIO</option></select></label><label>View<select value={agingView} onChange={(event) => setAgingView(event.target.value as "level" | "change")}><option value="change">Year-over-year</option>{agingSource === "rapm" && <option value="level">Level</option>}</select></label><label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label></section>
          <AgingChart rows={agingRows} valueKey={agingKey} label={`${agingSource.toUpperCase()} aging ${agingView}`} />
      </>}
    </div>
  </main>;
}
