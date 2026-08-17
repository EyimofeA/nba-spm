"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Component = "net" | "offense" | "defense";
type ModelId = "aio" | "spm" | "rapm";
type Tab = "about" | "ratings" | "player" | "roles" | "projections" | "research";
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
type AnnualRating = { Season: number; TEAM_ABBREVIATION?: string | null; [key: string]: string | number | boolean | null | undefined };
type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  roles: RoleSeason[];
  profiles: SkillProfile[];
};
type SkillProfile = { Season: number; [key: string]: number };
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
  validation: {
    walk_forward: ValidationRow[];
    walk_backward: ValidationRow[];
    aging_projection: {
      selected_method: string;
      selection: AgingMethodRow[];
      diagnostic: AgingMethodRow[];
      subgroups: Record<string, string | number>[];
    };
    win_probability: {
      checkpoints: WinProbabilityRow[];
      espn_game_start?: { brier: number; auc: number; rows: number };
      paired?: { delta_ci95: [number, number]; mean_game_brier_delta_candidate_minus_baseline: number };
    };
  };
  lineage: Record<string, string | null>;
};
type Point = { label: string; value: number };
type ValidationRow = { component: string; direction?: string; seasons?: string; folds?: number; rows?: number; rmse: number; correlation: number; r2: number };
type AgingMethodRow = { method: string; mean_rmse: number; mean_correlation: number; mean_r2: number; folds: number };
type WinProbabilityRow = { checkpoint: string; rows: number; brier: number; auc: number };
type TeamProjection = { TEAM_ABBREVIATION: string; projection_season: number; players: number; projected_net_rating: number; projected_win_pace: number };
type PlayerProjection = { PLAYER_ID: number; PLAYER_NAME: string; TEAM_ABBREVIATION: string; AGE: number; projected_offense: number; projected_defense: number; projected_net: number };

const formatRating = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const componentLabel: Record<Component, string> = { net: "Net", offense: "Off", defense: "Def" };
const roleColors = ["#e85d32", "#2467d5", "#22a06b", "#8c56c2", "#c99a12", "#9c4c68"];
const profileAxes = [
  ["shooting", "Shooting"], ["creation", "Creation"], ["security", "Security"],
  ["rim_pressure", "Rim pressure"], ["rebounding", "Rebounding"],
  ["shot_defense", "Shot defense"], ["disruption", "Disruption"], ["suppression", "Suppression"],
] as const;

function LineChart({ points, label, selected, onSelect }: { points: Point[]; label: string; selected?: string; onSelect?: (label: string) => void }) {
  if (!points.length) return <div className="empty">No rating.</div>;
  const width = 900, height = 270;
  const pad = { left: 46, right: 24, top: 24, bottom: 38 };
  const bound = Math.max(1, Math.ceil(Math.max(...points.map((point) => Math.abs(point.value)))));
  const x = (index: number) => points.length === 1 ? width / 2 : pad.left + index * (width - pad.left - pad.right) / (points.length - 1);
  const y = (value: number) => pad.top + (bound - value) * (height - pad.top - pad.bottom) / (bound * 2);
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
    {[-bound, 0, bound].map((tick) => <g key={tick}><line className={tick === 0 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick > 0 ? `+${tick}` : tick}</text></g>)}
    <polyline className="impact-line" points={points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ")} />
    {points.map((point, index) => <g className="chart-point" key={`${point.label}-${index}`} onClick={() => onSelect?.(point.label)}><circle className={selected === point.label ? "impact-dot selected" : "impact-dot"} cx={x(index)} cy={y(point.value)} r={selected === point.label ? "7" : "4"}><title>{`${point.label}: ${formatRating(point.value)}`}</title></circle><text className={selected === point.label ? "axis selected" : "axis"} x={x(index)} y={height - 12} textAnchor="middle">{point.label}</text></g>)}
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

function ProfileRadar({ profile, playerName }: { profile?: SkillProfile; playerName: string }) {
  if (!profile) return <div className="empty profile-empty">No profile for this year.</div>;
  const size = 520, center = size / 2, radius = 170;
  const point = (index: number, value: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / profileAxes.length;
    const scaled = radius * value / 100;
    return [center + Math.cos(angle) * scaled, center + Math.sin(angle) * scaled];
  };
  const ring = (value: number) => profileAxes.map((_, index) => point(index, value).join(",")).join(" ");
  const polygon = profileAxes.map(([key], index) => point(index, Number(profile[key] ?? 0)).join(",")).join(" ");
  return <div className="chart radar"><svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${playerName} ${profile.Season} skill profile`}>
    {[25, 50, 75, 100].map((value) => <polygon key={value} className="radar-ring" points={ring(value)} />)}
    {profileAxes.map(([, label], index) => { const [x, y] = point(index, 118); return <g key={label}><line className="radar-spoke" x1={center} y1={center} x2={point(index, 100)[0]} y2={point(index, 100)[1]} /><text className="radar-label" x={x} y={y} textAnchor={x < center - 5 ? "end" : x > center + 5 ? "start" : "middle"}>{label}</text></g>; })}
    <polygon className="radar-player" points={polygon} />
    {profileAxes.map(([key], index) => { const [x, y] = point(index, Number(profile[key] ?? 0)); return <circle className="radar-dot" key={key} cx={x} cy={y} r="4"><title>{`${profileAxes[index][1]}: ${Math.round(Number(profile[key]))}th percentile`}</title></circle>; })}
    <text className="radar-note" x={center} y={size - 16} textAnchor="middle">Season percentile · descriptive skill profile</text>
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

function RoleScatter({ points, mode, selectedId, onPlayer }: { points: RolePoint[]; mode: RoleMode; selectedId?: number; onPlayer: (id: number) => void }) {
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
    {points.map((point) => { const role = mode === "stable" ? point.stable_role : point.raw_role; const selected = point.PLAYER_ID === selectedId; return <g key={point.PLAYER_ID}><circle className={selected ? "role-dot selected" : "role-dot"} cx={x(point.x)} cy={y(point.y)} r={selected ? "10" : "5"} fill={color.get(role)} tabIndex={0} onClick={() => onPlayer(point.PLAYER_ID)}><title>{`${point.PLAYER_NAME} · ${point.TEAM_ABBREVIATION ?? "—"} · ${role}`}</title></circle>{selected && <text className="role-selected-name" x={x(point.x)} y={y(point.y) - 15} textAnchor="middle">{point.PLAYER_NAME}</text>}</g>; })}
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
  const [component, setComponent] = useState<Component>("net");
  const [season, setSeason] = useState("2024");
  const [team, setTeam] = useState("All");
  const [order, setOrder] = useState<"desc" | "asc">("desc");
  const [minimumPossessions, setMinimumPossessions] = useState(1000);
  const [sortKey, setSortKey] = useState("rating");
  const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [roleMode, setRoleMode] = useState<RoleMode>("stable");
  const [roleYear, setRoleYear] = useState(2024);
  const [rolePoints, setRolePoints] = useState<RolePoint[]>([]);
  const [selectedRoleId, setSelectedRoleId] = useState<number>();
  const [playerRoleMode, setPlayerRoleMode] = useState<RoleMode>("stable");
  const [roleSeason, setRoleSeason] = useState(2024);
  const [selectedPlayerSeason, setSelectedPlayerSeason] = useState(2024);
  const [agingSource, setAgingSource] = useState<"rapm" | "aio">("rapm");
  const [agingView, setAgingView] = useState<"level" | "change">("change");
  const [teamProjections, setTeamProjections] = useState<TeamProjection[]>([]);
  const [playerProjections, setPlayerProjections] = useState<PlayerProjection[]>([]);
  const [projectionView, setProjectionView] = useState<"teams" | "players">("teams");
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

  async function loadPlayer(id: number, open = true, requestedSeason?: number) {
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
      const latest = next.annual.at(-1)?.Season ?? 2024;
      const selected = next.annual.some((row) => row.Season === requestedSeason) ? requestedSeason! : latest;
      setPlayer(next); setRoleSeason(selected); setSelectedPlayerSeason(selected); setQuery(""); setStatus("");
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

  async function selectRolePlayer(id: number) {
    setSelectedRoleId(id);
    await loadPlayer(id, false);
    setTab("roles");
  }

  async function loadProjections() {
    if (teamProjections.length && playerProjections.length) return;
    setStatus("Loading…");
    try {
      const [teamsResponse, playersResponse] = await Promise.all([
        fetch("/data/projection-teams.json"), fetch("/data/projection-players.json"),
      ]);
      if (!teamsResponse.ok || !playersResponse.ok) throw new Error("projections unavailable");
      setTeamProjections(await teamsResponse.json()); setPlayerProjections(await playersResponse.json()); setStatus("");
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

  useEffect(() => {
    const tabs: Tab[] = ["about", "ratings", "player", "roles", "projections", "research"];
    const applyHash = () => {
      const requested = window.location.hash.slice(1) as Tab;
      if (tabs.includes(requested)) setTab(requested);
    };
    const initial = window.setTimeout(applyHash, 0);
    window.addEventListener("hashchange", applyHash);
    return () => { window.clearTimeout(initial); window.removeEventListener("hashchange", applyHash); };
  }, []);

  useEffect(() => {
    if (tab === "ratings" && catalog && leaderboard.length === 0) {
      const retry = window.setTimeout(() => void loadSeason(season, catalog.catalog.seasons), 0);
      return () => window.clearTimeout(retry);
    }
    // loadSeason is intentionally event-like; the state values below define the retry gate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, catalog, season, leaderboard.length]);

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return needle ? index.filter((item) => item.name.toLocaleLowerCase().includes(needle)).slice(0, 7) : [];
  }, [index, query]);

  function submitSearch(event: FormEvent) { event.preventDefault(); if (matches[0]) loadPlayer(matches[0].id); }
  const ratingKey = `aio_${component}`;
  const teams = useMemo(() => [...new Set(leaderboard.map((row) => row.TEAM_ABBREVIATION).filter((value): value is string => Boolean(value)))].sort(), [leaderboard]);
  const sortedRows = useMemo(() => {
    const eligible = leaderboard.filter((row) => Math.min(row.Poss_Off, row.Poss_Def) >= minimumPossessions);
    const rows = team === "All" ? eligible : eligible.filter((row) => row.TEAM_ABBREVIATION === team);
    const key = sortKey === "rating" ? ratingKey : sortKey;
    rows.sort((a, b) => {
      const av = a[key], bv = b[key];
      const comparison = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
      return (order === "asc" ? comparison : -comparison) || b.Season - a.Season || a.PLAYER_NAME.localeCompare(b.PLAYER_NAME);
    });
    return rows;
  }, [leaderboard, team, minimumPossessions, sortKey, ratingKey, order]);
  const points = useMemo<Point[]>(() => player?.annual.map((row) => ({ label: String(row.Season), value: Number(row[ratingKey]) })) ?? [], [player, ratingKey]);
  const selectedAnnual = player?.annual.find((row) => row.Season === selectedPlayerSeason) ?? player?.annual.at(-1);
  const currentRoles = player?.roles.find((row) => row.Season === roleSeason);
  const agingRows = catalog?.aging[agingSource].rows ?? [];
  const agingKey = agingView === "change" ? `change_${component}` : component === "net" ? "f_total" : component === "offense" ? "f_off" : "f_def";
  const currentProfile = player?.profiles.find((row) => row.Season === selectedPlayerSeason) ?? player?.profiles.at(-1);
  const selectedRolePoint = rolePoints.find((point) => point.PLAYER_ID === selectedRoleId);
  const similarRolePlayers = useMemo(() => {
    if (!selectedRolePoint) return [];
    return rolePoints.filter((point) => point.PLAYER_ID !== selectedRolePoint.PLAYER_ID).map((point) => ({
      ...point, distance: Math.hypot(point.x - selectedRolePoint.x, point.y - selectedRolePoint.y),
    })).sort((a, b) => a.distance - b.distance).slice(0, 8);
  }, [rolePoints, selectedRolePoint]);

  function sortBy(key: string) { if (sortKey === key) setOrder(order === "desc" ? "asc" : "desc"); else { setSortKey(key); setOrder("desc"); } }

  return <main id="top">
    <header><a className="brand" href="#top" onClick={() => setTab("about")}>NBA Impact</a><form className="search" onSubmit={submitSearch}><input aria-label="Find player" placeholder="Find player" value={query} onChange={(event) => setQuery(event.target.value)} />{matches.length > 0 && <div className="results">{matches.map((item) => <button type="button" key={item.id} onClick={() => loadPlayer(item.id)}>{item.name}</button>)}</div>}</form></header>
    <nav className="tabs" aria-label="Sections">{(["about", "ratings", "player", "roles", "projections", "research"] as Tab[]).map((item) => <a href={`#${item}`} key={item} className={tab === item ? "active" : ""} onClick={() => { setTab(item); if (item === "ratings" && leaderboard.length === 0) loadSeason(season); if (item === "roles") loadRoleMap(roleSide, roleYear); if (item === "projections") loadProjections(); }}>{item}</a>)}</nav>
    {status && <p className="status">{status}</p>}
    <div className="content">
      {tab === "about" && <>
        <section className="intro-hero"><p className="kicker">NBA IMPACT LAB</p><h1>One number.<br />More context.</h1><p>Explore player impact, skills and roles.</p><div className="intro-actions"><a href="#ratings" onClick={() => { setTab("ratings"); loadSeason(season); }}>View ratings</a><a href="#roles" className="secondary" onClick={() => { setTab("roles"); loadRoleMap(roleSide, roleYear); }}>View roles</a></div><div className="release-line"><span>2017–24</span><span>Points per 100 possessions</span></div></section>
        <section className="model-cards definitions"><article><span>SPM</span><h2>Stats</h2><p>A model that predicts impact from box, tracking, playtype and matchup data.</p></article><article><span>RAPM</span><h2>Lineups</h2><p>A model that separates each player’s scoreboard effect while controlling for the other nine players.</p></article><article><span>AIO</span><h2>Combined</h2><p>SPM starts the estimate. RAPM updates it with possession evidence.</p></article></section>
      </>}
      {tab === "ratings" && <>
        <section className="section-head"><div><p className="kicker">AIO RATINGS</p><h1>Player impact</h1></div><span>Points per 100 possessions</span></section>
        <section className="controls table-controls">
          <label>Year<select value={season} onChange={(event) => { setSeason(event.target.value); loadSeason(event.target.value); }}><option>All</option>{[...(catalog?.catalog.seasons ?? [])].reverse().map((year) => <option key={year}>{year}</option>)}</select></label>
          <label>Team<select value={team} onChange={(event) => setTeam(event.target.value)}><option>All</option>{teams.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label>Min poss<select value={minimumPossessions} onChange={(event) => setMinimumPossessions(Number(event.target.value))}><option value="0">All</option><option value="500">500</option><option value="1000">1,000</option><option value="2000">2,000</option></select></label>
          <label>Order<select value={order} onChange={(event) => setOrder(event.target.value as "desc" | "asc")}><option value="desc">High first</option><option value="asc">Low first</option></select></label>
        </section>
        <div className="ratings-table-wrap"><table className="ratings-table"><thead><tr><th>#</th><th><button onClick={() => sortBy("PLAYER_NAME")}>Player</button></th><th><button onClick={() => sortBy("TEAM_ABBREVIATION")}>Team</button></th><th><button onClick={() => sortBy("Season")}>Year</button></th><th><button onClick={() => sortBy("aio_offense")}>Off</button></th><th><button onClick={() => sortBy("aio_defense")}>Def</button></th><th><button onClick={() => sortBy("aio_net")}>Net</button></th><th><button onClick={() => sortBy("Poss_Off")}>Poss</button></th></tr></thead><tbody>{sortedRows.map((row, indexRow) => <tr key={`${row.PLAYER_ID}-${row.Season}`} tabIndex={0} onClick={() => loadPlayer(row.PLAYER_ID, true, row.Season)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") loadPlayer(row.PLAYER_ID, true, row.Season); }}><td>{indexRow + 1}</td><td className="player-cell">{row.PLAYER_NAME}</td><td>{row.TEAM_ABBREVIATION ?? "—"}</td><td>{row.Season}</td><td>{formatRating(Number(row.aio_offense))}</td><td>{formatRating(Number(row.aio_defense))}</td><td className="rating-cell">{formatRating(Number(row.aio_net))}</td><td>{Math.round(Math.min(row.Poss_Off, row.Poss_Def)).toLocaleString()}</td></tr>)}</tbody></table></div>
      </>}
      {tab === "player" && (player ? <>
        <section className="section-head"><div><p className="kicker">{selectedAnnual?.TEAM_ABBREVIATION ?? "PLAYER"} · {selectedPlayerSeason} · {selectedAnnual ? Math.round(Math.min(Number(selectedAnnual.Poss_Off), Number(selectedAnnual.Poss_Def))).toLocaleString() : "—"} POSS</p><h1>{player.PLAYER_NAME}</h1></div><strong className="big-rating">{selectedAnnual ? formatRating(Number(selectedAnnual[ratingKey])) : "—"}</strong></section>
        <section className="controls"><label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label></section>
        <LineChart points={points} label={`${player.PLAYER_NAME} annual impact`} selected={String(selectedPlayerSeason)} onSelect={(label) => { const year = Number(label); setSelectedPlayerSeason(year); setRoleSeason(year); }} />
        {component === "net" && selectedAnnual && <section className="equation"><div><span>Offense</span><b>{formatRating(Number(selectedAnnual.aio_offense))}</b></div><i>+</i><div><span>Defense</span><b>{formatRating(Number(selectedAnnual.aio_defense))}</b></div><i>=</i><div><span>Net</span><b>{formatRating(Number(selectedAnnual.aio_net))}</b></div></section>}
        <section className="subhead"><div><p className="kicker">SKILL PROFILE</p><h2>{currentProfile?.Season ?? selectedPlayerSeason}</h2></div><p className="muted compact">Season percentile.</p></section>
        <ProfileRadar profile={currentProfile} playerName={player.PLAYER_NAME} />
        <section className="subhead"><h2>Roles</h2><div className="inline-controls"><select aria-label="Player season" value={selectedPlayerSeason} onChange={(event) => { const year = Number(event.target.value); setSelectedPlayerSeason(year); setRoleSeason(year); }}>{player.annual.map((row) => <option key={row.Season}>{row.Season}</option>)}</select><select aria-label="Role mode" value={playerRoleMode} onChange={(event) => setPlayerRoleMode(event.target.value as RoleMode)}><option value="stable">Stable</option><option value="raw">Raw</option></select></div></section>
        <div className="role-grid"><RolePanel title="Offense" role={shownRole(currentRoles?.offense, playerRoleMode)} /><RolePanel title="Defense" role={shownRole(currentRoles?.defense, playerRoleMode)} /></div>
      </> : <div className="empty">Choose a player.</div>)}
      {tab === "roles" && catalog && <>
        <section className="section-head"><div><p className="kicker">ROLES</p><h1>Role map</h1></div></section>
        <section className="controls"><label>Side<select value={roleSide} onChange={(event) => { const next = event.target.value as RoleSide; const year = Math.max(...catalog.catalog.role_seasons[next]); setRoleSide(next); setRoleYear(year); setSelectedRoleId(undefined); loadRoleMap(next, year); }}><option value="offense">Offense</option><option value="defense">Defense</option></select></label><label>Year<select value={roleYear} onChange={(event) => { const year = Number(event.target.value); setRoleYear(year); setSelectedRoleId(undefined); loadRoleMap(roleSide, year); }}>{[...catalog.catalog.role_seasons[roleSide]].reverse().map((year) => <option key={year}>{year}</option>)}</select></label><label>Roles<select value={roleMode} onChange={(event) => setRoleMode(event.target.value as RoleMode)}><option value="stable">Stable</option><option value="raw">Raw</option></select></label></section>
        <p className="role-explainer">Raw uses this season only. Stable blends recent seasons to reduce noisy role flips.</p>
        <RoleScatter points={rolePoints} mode={roleMode} selectedId={selectedRoleId} onPlayer={selectRolePlayer} />
        {selectedRolePoint && player && <section className="role-selection"><div><p className="kicker">SELECTED</p><h2>{selectedRolePoint.PLAYER_NAME}</h2><p>{roleMode === "stable" ? selectedRolePoint.stable_role : selectedRolePoint.raw_role} · {selectedRolePoint.TEAM_ABBREVIATION ?? "—"}</p><button onClick={() => setTab("player")}>Open player</button></div><RolePanel title={roleSide === "offense" ? "Offense role mix" : "Defense role mix"} role={shownRole(player.roles.find((row) => row.Season === roleYear)?.[roleSide], roleMode)} /><div className="similar"><p className="kicker">MOST SIMILAR</p>{similarRolePlayers.map((item) => <button key={item.PLAYER_ID} onClick={() => selectRolePlayer(item.PLAYER_ID)}><b>{item.PLAYER_NAME}</b><span>{roleMode === "stable" ? item.stable_role : item.raw_role}</span></button>)}</div></section>}
      </>}
      {tab === "projections" && <>
        <section className="section-head"><div><p className="kicker">2027 BASELINE</p><h1>Projections</h1></div><span>Returning 2026 minutes</span></section>
        <section className="controls"><label>View<select value={projectionView} onChange={(event) => setProjectionView(event.target.value as "teams" | "players")}><option value="teams">Teams</option><option value="players">Players</option></select></label></section>
        {projectionView === "teams" ? <div className="ratings-table-wrap projection-table"><table className="ratings-table"><thead><tr><th>#</th><th>Team</th><th>Net</th><th>Win pace</th><th>Players</th></tr></thead><tbody>{[...teamProjections].sort((a, b) => b.projected_win_pace - a.projected_win_pace).map((row, indexRow) => <tr key={row.TEAM_ABBREVIATION}><td>{indexRow + 1}</td><td className="player-cell">{row.TEAM_ABBREVIATION}</td><td>{formatRating(row.projected_net_rating)}</td><td className="rating-cell">{row.projected_win_pace.toFixed(1)}</td><td>{row.players}</td></tr>)}</tbody></table></div> : <div className="ratings-table-wrap projection-table"><table className="ratings-table"><thead><tr><th>#</th><th>Player</th><th>Team</th><th>Age</th><th>Off</th><th>Def</th><th>Net</th></tr></thead><tbody>{[...playerProjections].sort((a, b) => b.projected_net - a.projected_net).slice(0, 150).map((row, indexRow) => <tr key={row.PLAYER_ID} onClick={() => loadPlayer(row.PLAYER_ID)}><td>{indexRow + 1}</td><td className="player-cell">{row.PLAYER_NAME}</td><td>{row.TEAM_ABBREVIATION}</td><td>{row.AGE.toFixed(0)}</td><td>{formatRating(row.projected_offense)}</td><td>{formatRating(row.projected_defense)}</td><td className="rating-cell">{formatRating(row.projected_net)}</td></tr>)}</tbody></table></div>}
        <p className="projection-note">Research baseline only. It ages the 2026 latent rating and holds each team’s 2026 players and minutes fixed. Trades, rookies, injuries and schedule are absent.</p>
      </>}
      {tab === "research" && catalog && <>
        <section className="section-head"><div><p className="kicker">RESEARCH</p><h1>What held up</h1></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">WALK FORWARD</p><h2>Earlier-season SPM vs RAPM</h2><p>Each test season’s SPM uses only older training seasons, then predicts that season’s RAPM target. This tests the statistical model, not the final AIO.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th>Side</th><th>Years</th><th>RMSE</th><th>r</th><th>R²</th></tr></thead><tbody>{catalog.validation.walk_forward.map((row) => <tr key={row.component}><td>{componentLabel[row.component as Component]}</td><td>{row.seasons}</td><td>{row.rmse.toFixed(3)}</td><td>{row.correlation.toFixed(3)}</td><td>{row.r2.toFixed(3)}</td></tr>)}</tbody></table></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">REVERSE CHECK</p><h2>Walk backward</h2><p>Training on later seasons and predicting earlier ones checks whether the mapping is merely riding the arrow of time. Similar results are useful; this is not forecast evidence.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th>Direction</th><th>Side</th><th>RMSE</th><th>r</th><th>R²</th></tr></thead><tbody>{catalog.validation.walk_backward.filter((row) => row.component === "net").map((row) => <tr key={row.direction}><td>{row.direction}</td><td>Net</td><td>{row.rmse.toFixed(3)}</td><td>{row.correlation.toFixed(3)}</td><td>{row.r2.toFixed(3)}</td></tr>)}</tbody></table></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">WIN PROBABILITY</p><h2>Score, clock and pregame context</h2><p>Team context improves the Elo baseline. At tipoff, ESPN remains better.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th>Checkpoint</th><th>Games</th><th>Brier ↓</th><th>AUC ↑</th></tr></thead><tbody>{catalog.validation.win_probability.checkpoints.map((row) => <tr key={row.checkpoint}><td>{row.checkpoint.replaceAll("_", " ")}</td><td>{row.rows}</td><td>{row.brier.toFixed(3)}</td><td>{row.auc.toFixed(3)}</td></tr>)}{catalog.validation.win_probability.espn_game_start && <tr><td>ESPN · game start</td><td>{catalog.validation.win_probability.espn_game_start.rows}</td><td>{catalog.validation.win_probability.espn_game_start.brier.toFixed(3)}</td><td>{catalog.validation.win_probability.espn_game_start.auc.toFixed(3)}</td></tr>}</tbody></table></div></section>
        <section className="research-block aging-block"><div className="research-copy"><p className="kicker">AGING</p><h2>Year-over-year change</h2><p>Descriptive curve below. For projection, age plus prior minutes beat the AR(1) baseline by 0.016 RMSE on 2022–23, a small gain.</p></div><div><section className="controls"><label>Metric<select value={agingSource} onChange={(event) => { const next = event.target.value as "rapm" | "aio"; setAgingSource(next); if (next === "aio") setAgingView("change"); }}><option value="rapm">RAPM</option><option value="aio">AIO</option></select></label><label>View<select value={agingView} onChange={(event) => setAgingView(event.target.value as "level" | "change")}><option value="change">Year-over-year</option>{agingSource === "rapm" && <option value="level">Level</option>}</select></label><label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label></section><AgingChart rows={agingRows} valueKey={agingKey} label={`${agingSource.toUpperCase()} aging ${agingView}`} /></div></section>
      </>}
    </div>
  </main>;
}
