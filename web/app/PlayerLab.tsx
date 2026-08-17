"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Component = "net" | "offense" | "defense";
type ModelId = "aio" | "rapm" | "spm";
type Tab = "home" | "ratings" | "player" | "roles" | "projections" | "research";
type RoleSide = "offense" | "defense";
type Membership = { role_id: string; label: string; affinity: number };
type Role = { primary_role: string; confidence: number; memberships: Membership[] };
type RoleSeason = { Season: number; offense?: Role; defense?: Role };
type RatingRow = Record<string, string | number | boolean | null | undefined>;
type AnnualRating = RatingRow & { Season: number; TEAM_ABBREVIATION?: string | null };
type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  roles: RoleSeason[];
  profiles: SkillProfile[];
};
type SkillProfile = { Season: number; [key: string]: number };
type PlayerIndex = { id: number; name: string; shard: number };
type LeaderboardRow = RatingRow & {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  Poss_Off: number;
  Poss_Def: number;
};
type RolePoint = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  x: number;
  y: number;
  raw_role: string;
};
type AgingRow = { Age: number; n: number | null; [key: string]: number | null | undefined };
type Catalog = {
  shards: number;
  catalog: { seasons: number[]; role_seasons: Record<RoleSide, number[]> };
  methods: { aio_equation: string; rapm_update_note: string };
  aging: Record<"rapm" | "aio", { coverage: string; rows: AgingRow[] }>;
  validation: {
    walk_forward: ValidationRow[];
    walk_backward: ValidationRow[];
    aging_projection: {
      selected_method: string;
      selection: AgingMethodRow[];
      diagnostic: AgingMethodRow[];
    };
    external_benchmark: {
      note: string;
      pinned_model_note: string;
      rows: ExternalRow[];
    };
  };
};
type Point = { label: string; value: number };
type ValidationRow = { component: string; direction?: string; seasons?: string; folds?: number; rows?: number; rmse: number; correlation: number; r2: number };
type AgingMethodRow = { method: string; mean_rmse: number; mean_correlation: number; folds: number };
type ExternalRow = { scope: string; exposure: string; players: number | null; component: string; bpm: number | null; xrapm: number | null };
type TeamProjection = { TEAM_ABBREVIATION: string; projection_season: number; players: number; projected_net_rating: number; projected_win_pace: number };
type PlayerProjection = { PLAYER_ID: number; PLAYER_NAME: string; TEAM_ABBREVIATION: string; AGE: number; projected_offense: number; projected_defense: number; projected_net: number };

// Mirrors MODEL_CATALOG in src/nba_impact/api/web_snapshot.py.
const MODELS: { id: ModelId; label: string; prefix: string; note: string }[] = [
  { id: "aio", label: "AIO", prefix: "aio_", note: "SPM center plus the RAPM update." },
  { id: "rapm", label: "Normal RAPM", prefix: "normal_rapm_", note: "Zero-prior one-season ridge on possessions." },
  { id: "spm", label: "SPM", prefix: "spm_", note: "Held-out statistical prediction that centers the RAPM fit." },
];
const TABS: Tab[] = ["home", "ratings", "player", "roles", "projections", "research"];
const formatRating = (value: number | undefined) => value === undefined || Number.isNaN(value) ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const componentLabel: Record<Component, string> = { net: "Net", offense: "Off", defense: "Def" };
const roleColors = ["#e85d32", "#2467d5", "#22a06b", "#8c56c2", "#c99a12", "#9c4c68"];
const profileAxes: Record<RoleSide, readonly (readonly [string, string])[]> = {
  offense: [["shooting", "Shooting"], ["spacing", "Spacing"], ["creation", "Creation"], ["security", "Security"], ["rim_pressure", "Rim pressure"], ["rebounding", "Rebounding"]],
  defense: [["shot_defense", "Shot defense"], ["disruption", "Disruption"], ["suppression", "Suppression"], ["rebounding", "Rebounding"]],
};

/** A model is selectable only when the loaded snapshot carries its values. */
function availableModels(rows: RatingRow[]) {
  return MODELS.map((model) => ({
    ...model,
    available: rows.length === 0 || rows.some((row) => typeof row[`${model.prefix}net`] === "number"),
  }));
}

const rating = (row: RatingRow | undefined, prefix: string, component: Component) => {
  const value = row?.[`${prefix}${component}`];
  return typeof value === "number" ? value : undefined;
};

/** One honest sentence about models the loaded snapshot cannot show. */
function missingModelNote(rows: RatingRow[]) {
  const missing = availableModels(rows).filter((item) => !item.available).map((item) => item.label);
  return missing.length ? `${missing.join(" and ")} need a rebuilt data snapshot from the pinned runs.` : "";
}

function ModelControl({ rows, model, onChange }: { rows: RatingRow[]; model: ModelId; onChange: (id: ModelId) => void }) {
  const models = availableModels(rows);
  return <label>Model<select value={model} onChange={(event) => onChange(event.target.value as ModelId)}>
    {models.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.available ? item.label : `${item.label} · not in snapshot`}</option>)}
  </select></label>;
}

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
  return <div className="chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
    {[low, 0, high].map((tick, index) => <g key={index}><line className={Math.abs(tick) < 1e-9 ? "zero" : "grid"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} /><text className="axis" x={pad.left - 9} y={y(tick) + 4} textAnchor="end">{tick.toFixed(1)}</text></g>)}
    <polyline className="aging total" points={points.map((point) => `${x(point.age)},${y(point.value)}`).join(" ")} />
    {points.map((point) => <circle className="aging-dot" key={point.age} cx={x(point.age)} cy={y(point.value)} r="3"><title>{`Age ${point.age}: ${formatRating(point.value)}`}</title></circle>)}
    {points.filter((_, index) => index % 3 === 0).map((point) => <text className="axis" key={point.age} x={x(point.age)} y={height - 12} textAnchor="middle">{point.age}</text>)}
  </svg></div>;
}

function ProfileRadar({ profile, playerName, side }: { profile?: SkillProfile; playerName: string; side: RoleSide }) {
  if (!profile) return <div className="empty profile-empty">No profile for this year.</div>;
  const axes = profileAxes[side].filter(([key]) => typeof profile[key] === "number");
  if (axes.length < 3) return <div className="empty profile-empty">No {side} profile for this year.</div>;
  const size = 460, center = size / 2, radius = 132;
  const point = (index: number, value: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / axes.length;
    const scaled = radius * value / 100;
    return [center + Math.cos(angle) * scaled, center + Math.sin(angle) * scaled];
  };
  const ring = (value: number) => axes.map((_, index) => point(index, value).join(",")).join(" ");
  const polygon = axes.map(([key], index) => point(index, Number(profile[key])).join(",")).join(" ");
  return <section className="profile-card"><h3>{side}</h3><div className="chart radar"><svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${playerName} ${profile.Season} ${side} skill profile`}>
    {[25, 50, 75, 100].map((value) => <polygon key={value} className="radar-ring" points={ring(value)} />)}
    {axes.map(([, label], index) => { const [x, y] = point(index, 124); return <g key={label}><line className="radar-spoke" x1={center} y1={center} x2={point(index, 100)[0]} y2={point(index, 100)[1]} /><text className="radar-label" x={x} y={y} textAnchor={x < center - 5 ? "end" : x > center + 5 ? "start" : "middle"}>{label}</text></g>; })}
    <polygon className="radar-player" points={polygon} />
    {axes.map(([key, label], index) => { const [x, y] = point(index, Number(profile[key])); return <circle className="radar-dot" key={key} cx={x} cy={y} r="4"><title>{`${label}: ${Math.round(Number(profile[key]))} percentile`}</title></circle>; })}
  </svg></div></section>;
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

function RoleScatter({ points, selectedId, onPlayer }: { points: RolePoint[]; selectedId?: number; onPlayer: (id: number) => void }) {
  if (!points.length) return <div className="empty">No role map.</div>;
  const width = 900, height = 520, pad = 34;
  const xs = points.map((point) => point.x), ys = points.map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value: number) => pad + (value - minX) * (width - pad * 2) / Math.max(0.001, maxX - minX);
  const y = (value: number) => height - pad - (value - minY) * (height - pad * 2) / Math.max(0.001, maxY - minY);
  const roles = [...new Set(points.map((point) => point.raw_role))].sort();
  const color = new Map(roles.map((role, index) => [role, roleColors[index % roleColors.length]]));
  return <><div className="role-legend">{roles.map((role) => <span key={role}><i style={{ background: color.get(role) }} />{role}</span>)}</div><div className="chart scatter"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Player role cluster map">
    <line className="grid" x1={width / 2} x2={width / 2} y1={pad} y2={height - pad} /><line className="grid" x1={pad} x2={width - pad} y1={height / 2} y2={height / 2} />
    {points.map((point) => { const selected = point.PLAYER_ID === selectedId; return <g key={point.PLAYER_ID}><circle className={selected ? "role-dot selected" : "role-dot"} cx={x(point.x)} cy={y(point.y)} r={selected ? "10" : "5"} fill={color.get(point.raw_role)} tabIndex={0} onClick={() => onPlayer(point.PLAYER_ID)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onPlayer(point.PLAYER_ID); }}><title>{`${point.PLAYER_NAME} · ${point.TEAM_ABBREVIATION ?? "—"} · ${point.raw_role}`}</title></circle>{selected && <text className="role-selected-name" x={x(point.x)} y={y(point.y) - 15} textAnchor="middle">{point.PLAYER_NAME}</text>}</g>; })}
    <text className="axis" x={width - pad} y={height - 8} textAnchor="end">Role axis 1</text><text className="axis" x={pad} y={18}>Role axis 2</text>
  </svg></div></>;
}

export function PlayerLab() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [index, setIndex] = useState<PlayerIndex[]>([]);
  const [tab, setTab] = useState<Tab>("home");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("Loading…");
  const [player, setPlayer] = useState<Player | null>(null);
  const [component, setComponent] = useState<Component>("net");
  const [model, setModel] = useState<ModelId>("aio");
  const [season, setSeason] = useState("2024");
  const [team, setTeam] = useState("All");
  const [order, setOrder] = useState<"desc" | "asc">("desc");
  const [minimumPossessions, setMinimumPossessions] = useState(0);
  const [sortKey, setSortKey] = useState("net");
  const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [roleYear, setRoleYear] = useState(2024);
  const [rolePoints, setRolePoints] = useState<RolePoint[]>([]);
  const [selectedRoleId, setSelectedRoleId] = useState<number>();
  const [roleSeason, setRoleSeason] = useState(2024);
  const [selectedPlayerSeason, setSelectedPlayerSeason] = useState(2024);
  const [agingSource, setAgingSource] = useState<"rapm" | "aio">("rapm");
  const [agingView, setAgingView] = useState<"level" | "change">("change");
  const [teamProjections, setTeamProjections] = useState<TeamProjection[]>([]);
  const [playerProjections, setPlayerProjections] = useState<PlayerProjection[]>([]);
  const [projectionView, setProjectionView] = useState<"teams" | "players">("teams");
  const [projectionSeason, setProjectionSeason] = useState<number>();
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
    const applyHash = () => {
      const requested = window.location.hash.slice(1) as Tab;
      if (TABS.includes(requested)) setTab(requested);
    };
    const initial = window.setTimeout(applyHash, 0);
    window.addEventListener("hashchange", applyHash);
    return () => { window.clearTimeout(initial); window.removeEventListener("hashchange", applyHash); };
  }, []);

  // Deep links (#ratings, #roles, #projections) must load their own data.
  useEffect(() => {
    if (!catalog) return;
    const load = tab === "ratings" && leaderboard.length === 0 ? () => loadSeason(season, catalog.catalog.seasons)
      : tab === "roles" && rolePoints.length === 0 ? () => loadRoleMap(roleSide, roleYear)
      : tab === "projections" && teamProjections.length === 0 ? () => loadProjections()
      : undefined;
    if (!load) return;
    const retry = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(retry);
    // The loaders are event-like; the state values below define the retry gate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, catalog, season, leaderboard.length, rolePoints.length, roleSide, roleYear, teamProjections.length]);

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return needle ? index.filter((item) => item.name.toLocaleLowerCase().includes(needle)).slice(0, 7) : [];
  }, [index, query]);

  function submitSearch(event: FormEvent) { event.preventDefault(); if (matches[0]) loadPlayer(matches[0].id); }

  const ratingsModel = availableModels(leaderboard).find((item) => item.id === model && item.available) ?? MODELS[0];
  const playerModel = availableModels(player?.annual ?? []).find((item) => item.id === model && item.available) ?? MODELS[0];
  const teams = useMemo(() => [...new Set(leaderboard.map((row) => row.TEAM_ABBREVIATION).filter((value): value is string => Boolean(value)))].sort(), [leaderboard]);
  const sortedRows = useMemo(() => {
    const eligible = leaderboard.filter((row) => Math.min(row.Poss_Off, row.Poss_Def) >= minimumPossessions);
    const rows = team === "All" ? eligible : eligible.filter((row) => row.TEAM_ABBREVIATION === team);
    const value = (row: LeaderboardRow) => sortKey === "Poss" ? Math.min(row.Poss_Off, row.Poss_Def)
      : ["net", "offense", "defense"].includes(sortKey) ? row[`${ratingsModel.prefix}${sortKey}`]
      : row[sortKey];
    rows.sort((a, b) => {
      const av = value(a), bv = value(b);
      const comparison = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
      return (order === "asc" ? comparison : -comparison) || b.Season - a.Season || a.PLAYER_NAME.localeCompare(b.PLAYER_NAME);
    });
    return rows;
  }, [leaderboard, team, minimumPossessions, sortKey, ratingsModel.prefix, order]);
  const points = useMemo<Point[]>(() => player?.annual.flatMap((row) => {
    const value = rating(row, playerModel.prefix, component);
    return value === undefined ? [] : [{ label: String(row.Season), value }];
  }) ?? [], [player, playerModel.prefix, component]);
  const selectedAnnual = player?.annual.find((row) => row.Season === selectedPlayerSeason) ?? player?.annual.at(-1);
  const currentRoles = player?.roles.find((row) => row.Season === roleSeason);
  const agingRows = catalog?.aging[agingSource].rows ?? [];
  const agingKey = agingView === "change" ? `change_${component}` : component === "net" ? "f_total" : component === "offense" ? "f_off" : "f_def";
  const currentProfile = player?.profiles.find((row) => row.Season === selectedPlayerSeason) ?? player?.profiles.at(-1);
  const selectedRolePoint = rolePoints.find((point) => point.PLAYER_ID === selectedRoleId);
  const projectionSeasons = useMemo(() => {
    const playerSeasons = new Set(playerProjections.map((row) => row.projection_season));
    return [...new Set(teamProjections.map((row) => row.projection_season))]
      .filter((year) => playerSeasons.has(year)).sort((a, b) => a - b);
  }, [teamProjections, playerProjections]);
  const activeProjectionSeason = projectionSeason ?? projectionSeasons.at(-1);
  const visibleTeamProjections = useMemo(
    () => teamProjections.filter((row) => row.projection_season === activeProjectionSeason),
    [teamProjections, activeProjectionSeason],
  );
  const visiblePlayerProjections = useMemo(
    () => playerProjections.filter((row) => row.projection_season === activeProjectionSeason),
    [playerProjections, activeProjectionSeason],
  );
  const spmCenter = rating(selectedAnnual, "spm_", component);
  const aioValue = rating(selectedAnnual, "aio_", component);
  const seasons = catalog?.catalog.seasons ?? [];
  const similarRolePlayers = useMemo(() => {
    if (!selectedRolePoint) return [];
    return rolePoints.filter((point) => point.PLAYER_ID !== selectedRolePoint.PLAYER_ID).map((point) => ({
      ...point, distance: Math.hypot(point.x - selectedRolePoint.x, point.y - selectedRolePoint.y),
    })).sort((a, b) => a.distance - b.distance).slice(0, 8);
  }, [rolePoints, selectedRolePoint]);

  function sortBy(key: string) { if (sortKey === key) setOrder(order === "desc" ? "asc" : "desc"); else { setSortKey(key); setOrder("desc"); } }
  const sortState = (key: string) => sortKey === key ? (order === "asc" ? "ascending" : "descending") : "none";
  const columns: [string, string][] = [["PLAYER_NAME", "Player"], ["TEAM_ABBREVIATION", "Team"], ["Season", "Year"], ["offense", "Off"], ["defense", "Def"], ["net", "Net"], ["Poss", "Poss"]];

  return <main id="top">
    <header><a className="brand" href="#home" onClick={() => setTab("home")}>NBA Impact</a><form className="search" onSubmit={submitSearch}><input aria-label="Find player" placeholder="Find player" value={query} onChange={(event) => setQuery(event.target.value)} />{matches.length > 0 && <div className="results">{matches.map((item) => <button type="button" key={item.id} onClick={() => loadPlayer(item.id)}>{item.name}</button>)}</div>}</form></header>
    <nav className="tabs" aria-label="Sections">{TABS.map((item) => <a href={`#${item}`} key={item} className={tab === item ? "active" : ""} aria-current={tab === item ? "page" : undefined} onClick={() => { setTab(item); if (item === "ratings" && leaderboard.length === 0) loadSeason(season); if (item === "roles") loadRoleMap(roleSide, roleYear); if (item === "projections") loadProjections(); }}>{item}</a>)}</nav>
    {status && <p className="status" role="status">{status}</p>}
    <div className="content">
      {tab === "home" && <>
        <section className="intro-hero"><p className="kicker">NBA IMPACT LAB</p><h1>One number.<br />More context.</h1><p>Every rating is points per 100 possessions. Offense and defense are separate, and both are positive when the player helps.</p><div className="intro-actions"><a href="#ratings" onClick={() => { setTab("ratings"); loadSeason(season); }}>View ratings</a><a href="#research" className="secondary" onClick={() => setTab("research")}>Read the evidence</a></div><div className="release-line"><span>{seasons.length ? `${seasons[0]}–${seasons[seasons.length - 1]}` : "2017–24"}</span><span>{seasons.length} seasons</span><span>Points per 100 possessions</span><span>Retrospective research</span></div></section>
        <section className="model-cards definitions">
          <article><span>SPM</span><h2>Stats</h2><p>A statistical model. It predicts one season of impact from box, tracking, playtype, and matchup features. The rated season is never in its own training labels.</p></article>
          <article><span>RAPM</span><h2>Lineups</h2><p>A ridge regression on possessions. It separates each player from the other nine on the floor. Normal RAPM uses no prior.</p></article>
          <article><span>AIO</span><h2>Combined</h2><p>The all-in-one rating. SPM sets the center, then one centered ridge fit updates it with that season’s possessions.</p></article>
        </section>
        <section className="method-section"><div className="method-title"><p className="kicker">HOW TO READ</p><h2>Four rules</h2></div><ol className="method-steps">
          <li><b>Scale</b><span>A rating of +3.0 means the model estimates about three more points per 100 possessions for the team with that player on the floor, after it separates teammates and opponents.</span></li>
          <li><b>Sides</b><span>Offense and defense add to net. A positive defensive rating means the player prevents points.</span></li>
          <li><b>Sample</b><span>The possession column shows the smaller of offensive and defensive exposure. Small samples move toward the model’s center, so read low-possession rows with care.</span></li>
          <li><b>Scope</b><span>These ratings describe past seasons. They are not forecasts, and they carry no published interval yet.</span></li>
        </ol></section>
      </>}
      {tab === "ratings" && <>
        <section className="section-head"><div><p className="kicker">RATINGS</p><h1>Player impact</h1></div><span>Points per 100 possessions</span></section>
        <section className="controls table-controls">
          <ModelControl rows={leaderboard} model={model} onChange={setModel} />
          <label>Year<select value={season} onChange={(event) => { setSeason(event.target.value); loadSeason(event.target.value); }}><option>All</option>{[...seasons].reverse().map((year) => <option key={year}>{year}</option>)}</select></label>
          <label>Team<select value={team} onChange={(event) => setTeam(event.target.value)}><option>All</option>{teams.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label>Min poss<select value={minimumPossessions} onChange={(event) => setMinimumPossessions(Number(event.target.value))}><option value="0">All</option><option value="500">500</option><option value="1000">1,000</option><option value="2000">2,000</option></select></label>
        </section>
        <p className="control-note">{ratingsModel.label}: {ratingsModel.note} Sort by any column header. {missingModelNote(leaderboard)}</p>
        <div className="ratings-table-wrap"><table className="ratings-table"><caption className="visually-hidden">{`${ratingsModel.label} ratings, points per 100 possessions`}</caption><thead><tr><th scope="col">#</th>{columns.map(([key, label]) => <th key={key} scope="col" aria-sort={sortState(key)}><button onClick={() => sortBy(key)}>{label}</button></th>)}</tr></thead><tbody>{sortedRows.map((row, indexRow) => <tr key={`${row.PLAYER_ID}-${row.Season}`} tabIndex={0} onClick={() => loadPlayer(row.PLAYER_ID, true, row.Season)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") loadPlayer(row.PLAYER_ID, true, row.Season); }}><td>{indexRow + 1}</td><td className="player-cell">{row.PLAYER_NAME}</td><td>{row.TEAM_ABBREVIATION ?? "—"}</td><td>{row.Season}</td><td>{formatRating(rating(row, ratingsModel.prefix, "offense"))}</td><td>{formatRating(rating(row, ratingsModel.prefix, "defense"))}</td><td className="rating-cell">{formatRating(rating(row, ratingsModel.prefix, "net"))}</td><td>{Math.round(Math.min(row.Poss_Off, row.Poss_Def)).toLocaleString()}</td></tr>)}</tbody></table></div>
      </>}
      {tab === "player" && (player ? <>
        <section className="section-head"><div><p className="kicker">{selectedAnnual?.TEAM_ABBREVIATION ?? "PLAYER"} · {selectedPlayerSeason} · {selectedAnnual ? Math.round(Math.min(Number(selectedAnnual.Poss_Off), Number(selectedAnnual.Poss_Def))).toLocaleString() : "—"} POSS</p><h1>{player.PLAYER_NAME}</h1></div><strong className="big-rating">{formatRating(rating(selectedAnnual, playerModel.prefix, component))}</strong></section>
        <section className="controls">
          <ModelControl rows={player.annual} model={model} onChange={setModel} />
          <label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label>
        </section>
        <p className="control-note">{playerModel.label}: {playerModel.note} {missingModelNote(player.annual)}</p>
        <LineChart points={points} label={`${player.PLAYER_NAME} annual ${playerModel.label} ${component}`} selected={String(selectedPlayerSeason)} onSelect={(label) => { const year = Number(label); setSelectedPlayerSeason(year); setRoleSeason(year); }} />
        {component === "net" && selectedAnnual && <section className="equation"><div><span>Offense</span><b>{formatRating(rating(selectedAnnual, playerModel.prefix, "offense"))}</b></div><i>+</i><div><span>Defense</span><b>{formatRating(rating(selectedAnnual, playerModel.prefix, "defense"))}</b></div><i>=</i><div><span>{playerModel.label} net</span><b>{formatRating(rating(selectedAnnual, playerModel.prefix, "net"))}</b></div></section>}
        {playerModel.id === "aio" && spmCenter !== undefined && aioValue !== undefined && <><section className="equation"><div><span>SPM center</span><b>{formatRating(spmCenter)}</b></div><i>+</i><div><span>RAPM update</span><b>{formatRating(aioValue - spmCenter)}</b></div><i>=</i><div><span>AIO {componentLabel[component].toLowerCase()}</span><b>{formatRating(aioValue)}</b></div></section><p className="control-note">{catalog?.methods.rapm_update_note}</p></>}
        <section className="subhead"><div><p className="kicker">SKILL PROFILE</p><h2>{currentProfile?.Season ?? selectedPlayerSeason}</h2></div><p className="muted compact">Season-relative percentile.</p></section>
        <div className={`profile-grid ${component}`}>
          {component !== "defense" && <ProfileRadar profile={currentProfile} playerName={player.PLAYER_NAME} side="offense" />}
          {component !== "offense" && <ProfileRadar profile={currentProfile} playerName={player.PLAYER_NAME} side="defense" />}
        </div>
        <section className="subhead"><h2>Roles</h2><div className="inline-controls"><select aria-label="Player season" value={selectedPlayerSeason} onChange={(event) => { const year = Number(event.target.value); setSelectedPlayerSeason(year); setRoleSeason(year); }}>{player.annual.map((row) => <option key={row.Season}>{row.Season}</option>)}</select></div></section>
        <div className={`role-grid ${component}`}>
          {component !== "defense" && <RolePanel title="Offense" role={currentRoles?.offense} />}
          {component !== "offense" && <RolePanel title="Defense" role={currentRoles?.defense} />}
        </div>
        <p className="control-note">Role mix for this season. Roles are not rating inputs.</p>
      </> : <div className="empty">Choose a player.</div>)}
      {tab === "roles" && catalog && <>
        <section className="section-head"><div><p className="kicker">ROLES</p><h1>Role map</h1></div></section>
        <section className="controls"><label>Side<select value={roleSide} onChange={(event) => { const next = event.target.value as RoleSide; const year = Math.max(...catalog.catalog.role_seasons[next]); setRoleSide(next); setRoleYear(year); setSelectedRoleId(undefined); loadRoleMap(next, year); }}><option value="offense">Offense</option><option value="defense">Defense</option></select></label><label>Year<select value={roleYear} onChange={(event) => { const year = Number(event.target.value); setRoleYear(year); setSelectedRoleId(undefined); loadRoleMap(roleSide, year); }}>{[...catalog.catalog.role_seasons[roleSide]].reverse().map((year) => <option key={year}>{year}</option>)}</select></label></section>
        <p className="role-explainer">Each point is one player-season. Clusters come from behavior only, one season at a time. Height and listed position are excluded. Roles describe how a player is used; they do not measure impact.</p>
        <RoleScatter points={rolePoints} selectedId={selectedRoleId} onPlayer={selectRolePlayer} />
        {selectedRolePoint && player && <section className="role-selection"><div><p className="kicker">SELECTED</p><h2>{selectedRolePoint.PLAYER_NAME}</h2><p>{selectedRolePoint.raw_role} · {selectedRolePoint.TEAM_ABBREVIATION ?? "—"}</p><button onClick={() => setTab("player")}>Open player</button></div><RolePanel title={roleSide === "offense" ? "Offense role mix" : "Defense role mix"} role={player.roles.find((row) => row.Season === roleYear)?.[roleSide]} /><div className="similar"><p className="kicker">MOST SIMILAR</p>{similarRolePlayers.map((item) => <button key={item.PLAYER_ID} onClick={() => selectRolePlayer(item.PLAYER_ID)}><b>{item.PLAYER_NAME}</b><span>{item.raw_role}</span></button>)}</div></section>}
      </>}
      {tab === "projections" && <>
        <section className="section-head"><div><p className="kicker">{activeProjectionSeason ?? "—"} FORECAST</p><h1>Projections</h1></div><span>Research baseline</span></section>
        <section className="controls"><label>Season<select value={activeProjectionSeason ?? ""} onChange={(event) => setProjectionSeason(Number(event.target.value))}>{projectionSeasons.map((year) => <option key={year} value={year}>{year}</option>)}</select></label><label>View<select value={projectionView} onChange={(event) => setProjectionView(event.target.value as "teams" | "players")}><option value="teams">Teams</option><option value="players">Players</option></select></label></section>
        {projectionView === "teams" ? <div className="ratings-table-wrap projection-table"><table className="ratings-table"><thead><tr><th scope="col">#</th><th scope="col">Team</th><th scope="col">Net</th><th scope="col">Win pace</th><th scope="col">Players</th></tr></thead><tbody>{[...visibleTeamProjections].sort((a, b) => b.projected_win_pace - a.projected_win_pace).map((row, indexRow) => <tr key={row.TEAM_ABBREVIATION}><td>{indexRow + 1}</td><td className="player-cell">{row.TEAM_ABBREVIATION}</td><td>{formatRating(row.projected_net_rating)}</td><td className="rating-cell">{row.projected_win_pace.toFixed(1)}</td><td>{row.players}</td></tr>)}</tbody></table></div> : <div className="ratings-table-wrap projection-table"><table className="ratings-table"><thead><tr><th scope="col">#</th><th scope="col">Player</th><th scope="col">Team</th><th scope="col">Age</th><th scope="col">Off</th><th scope="col">Def</th><th scope="col">Net</th></tr></thead><tbody>{[...visiblePlayerProjections].sort((a, b) => b.projected_net - a.projected_net).slice(0, 150).map((row, indexRow) => <tr key={row.PLAYER_ID} onClick={() => loadPlayer(row.PLAYER_ID)}><td>{indexRow + 1}</td><td className="player-cell">{row.PLAYER_NAME}</td><td>{row.TEAM_ABBREVIATION}</td><td>{row.AGE.toFixed(0)}</td><td>{formatRating(row.projected_offense)}</td><td>{formatRating(row.projected_defense)}</td><td className="rating-cell">{formatRating(row.projected_net)}</td></tr>)}</tbody></table></div>}
        <p className="projection-note">Only {projectionSeasons.join(", ") || "no"} forecast vintage{projectionSeasons.length === 1 ? " is" : "s are"} published. This view will add 2018–2026 when their player and team forecast artifacts are exported. {activeProjectionSeason === 2027 ? "2027 is a forecast, not observed training or confirmation data." : ""}</p>
      </>}
      {tab === "research" && catalog && <>
        <section className="section-head"><div><p className="kicker">RESEARCH</p><h1>What held up</h1></div><span>Frozen runs only</span></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">BUILD ORDER</p><h2>From SPM to AIO</h2><p>SPM predicts one season of impact from box, tracking, playtype, and matchup features. One ridge RAPM fit is then centered on that prediction. Normal RAPM is the separate zero-prior fit. It is the reference model, not a part of AIO.</p></div><div><p className="formula">{catalog.methods.aio_equation}</p><div className="factor-grid"><span>Rated season excluded from its own SPM labels</span><span>Ridge penalties fixed at 3000 / 3000 / 300</span><span>Offense and defense fitted separately, then added</span><span>Identity holds to floating-point precision</span></div><p className="control-note">{catalog.methods.rapm_update_note}</p></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">FORWARD TEST</p><h2>Earlier seasons only</h2><p>Each season’s SPM is trained on earlier seasons only, then scored against that season’s normal RAPM. Errors are weighted by the square root of the smaller possession count. The published ratings use leave-one-season-out centers, which may also see later seasons, so this stricter table is the honest accuracy check.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th scope="col">Side</th><th scope="col">Years</th><th scope="col">Folds</th><th scope="col">RMSE</th><th scope="col">r</th><th scope="col">R²</th></tr></thead><tbody>{catalog.validation.walk_forward.map((row) => <tr key={row.component}><td>{componentLabel[row.component as Component]}</td><td>{row.seasons}</td><td>{row.folds}</td><td>{row.rmse.toFixed(3)}</td><td>{row.correlation.toFixed(3)}</td><td>{row.r2.toFixed(3)}</td></tr>)}</tbody></table></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">DIRECTION CHECK</p><h2>Next year and last year</h2><p>The frozen held-out SPM at one season is compared with normal RAPM in the next season and in the previous season. Each direction uses the same 1,768 matched transitions. Net correlation is almost equal in both directions, so the agreement is not explained by player development alone. This is a diagnostic, not forecast evidence.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th scope="col">Direction</th><th scope="col">Side</th><th scope="col">Rows</th><th scope="col">RMSE</th><th scope="col">r</th></tr></thead><tbody>{catalog.validation.walk_backward.map((row) => <tr key={`${row.direction}-${row.component}`}><td>{row.direction === "forward" ? "Next season" : "Previous season"}</td><td>{componentLabel[row.component as Component]}</td><td>{row.rows?.toLocaleString()}</td><td>{row.rmse.toFixed(3)}</td><td>{row.correlation.toFixed(3)}</td></tr>)}</tbody></table></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">EXTERNAL AGREEMENT</p><h2>Against public metrics</h2><p>{catalog.validation.external_benchmark.note} Correlations use matched, minutes-weighted rows above the stated exposure floor.</p><p>{catalog.validation.external_benchmark.pinned_model_note}</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th scope="col">Comparison</th><th scope="col">Side</th><th scope="col">Players</th><th scope="col">r with BPM</th><th scope="col">r with xRAPM</th></tr></thead><tbody>{catalog.validation.external_benchmark.rows.map((row, rowIndex) => <tr key={rowIndex}><td>{row.scope}<span className="cell-note">{row.exposure}</span></td><td>{componentLabel[row.component as Component]}</td><td>{row.players?.toLocaleString() ?? "—"}</td><td>{row.bpm?.toFixed(3) ?? "—"}</td><td>{row.xrapm?.toFixed(3) ?? "—"}</td></tr>)}</tbody></table></div></section>
        <section className="research-block aging-block"><div className="research-copy"><p className="kicker">AGING</p><h2>Observed change by age</h2><p>The curve shows measured year-over-year change for players who kept playing. Survivors stay in the sample, so it is descriptive, not a causal aging effect.</p></div><div><section className="controls"><label>Metric<select value={agingSource} onChange={(event) => { const next = event.target.value as "rapm" | "aio"; setAgingSource(next); if (next === "aio") setAgingView("change"); }}><option value="rapm">RAPM</option><option value="aio">AIO</option></select></label><label>View<select value={agingView} onChange={(event) => setAgingView(event.target.value as "level" | "change")}><option value="change">Year-over-year</option>{agingSource === "rapm" && <option value="level">Level</option>}</select></label><label>Side<select value={component} onChange={(event) => setComponent(event.target.value as Component)}>{(["net", "offense", "defense"] as Component[]).map((item) => <option key={item} value={item}>{componentLabel[item]}</option>)}</select></label></section><AgingChart rows={agingRows} valueKey={agingKey} label={`${agingSource.toUpperCase()} aging ${agingView}`} /><p className="control-note">Coverage: {catalog.aging[agingSource].coverage}.</p></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">PROJECTION METHOD</p><h2>Age helps a little</h2><p>Candidate one-year projection methods were selected on four earlier origin seasons and then scored on two later ones. Age plus prior minutes was selected. Its gain over the plain AR(1) baseline is small.</p></div><div className="ratings-table-wrap"><table className="mini-table"><thead><tr><th scope="col">Method</th><th scope="col">Selection RMSE</th><th scope="col">Diagnostic RMSE</th></tr></thead><tbody>{catalog.validation.aging_projection.selection.map((row) => <tr key={row.method} className={row.method === catalog.validation.aging_projection.selected_method ? "selected-row" : undefined}><td>{row.method.replaceAll("_", " ")}</td><td>{row.mean_rmse.toFixed(3)}</td><td>{catalog.validation.aging_projection.diagnostic.find((item) => item.method === row.method)?.mean_rmse.toFixed(3) ?? "—"}</td></tr>)}</tbody></table></div></section>
        <section className="research-block"><div className="research-copy"><p className="kicker">LIMITS</p><h2>What is not proven</h2><p>These are recorded results, including the failures. Read them before you quote a rating.</p></div><ul className="limit-list">
          <li><b>Defense is the weak side.</b><span>Held-out defensive accuracy stays well below offense in every test above.</span></li>
          <li><b>The frozen model missed its untouched confirmation season.</b><span>Offense, defense, and net RMSE were 1.102, 1.154, and 1.610, and defense correlation fell to 0.331. The public ratings therefore stay pinned to 2017–24.</span></li>
          <li><b>Production RAPM keeps a zero prior.</b><span>The matched-game test of a prior-centered fit had a paired interval that crossed zero. A later annual prior challenger looks better, but it stays research only.</span></li>
          <li><b>No published intervals.</b><span>SPM and AIO uncertainty is not calibrated yet, so no rating here shows a range.</span></li>
          <li><b>Roles and skill profiles are descriptive.</b><span>They are not impact estimates and not model inputs.</span></li>
          <li><b>Season 2027 is untouched.</b><span>It is reserved for annual confirmation and is never used to develop or select a model.</span></li>
        </ul></section>
      </>}
    </div>
  </main>;
}
