/** Types and cached loaders for the published web snapshot. */

export type Component = "net" | "offense" | "defense";
export type ModelId = "aio" | "rapm" | "spm";
export type RoleSide = "offense" | "defense";

export const COMPONENTS: Component[] = ["net", "offense", "defense"];
export const COMPONENT_LABEL: Record<Component, string> = {
  net: "Net",
  offense: "Offense",
  defense: "Defense",
};

/** Mirrors MODEL_CATALOG in src/nba_impact/api/web_snapshot.py. */
export const MODELS: {
  id: ModelId;
  label: string;
  prefix: string;
  note: string;
}[] = [
  {
    id: "aio",
    label: "AIO",
    prefix: "aio_",
    note: "SPM center plus the RAPM update.",
  },
  {
    id: "rapm",
    label: "RAPM",
    prefix: "normal_rapm_",
    note: "Zero-prior one-season ridge on possessions.",
  },
  {
    id: "spm",
    label: "SPM",
    prefix: "spm_",
    note: "Held-out statistical prediction that centers the RAPM fit.",
  },
];

export type RatingRow = Record<
  string,
  string | number | boolean | null | undefined
>;
export type AnnualRating = RatingRow & {
  Season: number;
  TEAM_ABBREVIATION?: string | null;
};
export type SkillProfile = { Season: number; [key: string]: number | null };
export type Membership = { role_id: string; label: string; affinity: number };
export type Role = {
  primary_role: string;
  confidence: number;
  memberships: Membership[];
};
export type RoleSeason = { Season: number; offense?: Role; defense?: Role };

export type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  roles: RoleSeason[];
  profiles: SkillProfile[];
};

export type PlayerIndex = { id: number; name: string; shard: number };

export type LeaderboardRow = RatingRow & {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  Poss_Off: number;
  Poss_Def: number;
};

export type RolePoint = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  TEAM_ABBREVIATION: string | null;
  x: number;
  y: number;
  raw_role: string;
};

export type AgingRow = {
  Age: number;
  n: number | null;
  [key: string]: number | null | undefined;
};
export type ValidationRow = {
  component: string;
  direction?: string;
  seasons?: string;
  folds?: number;
  rows?: number;
  rmse: number;
  correlation: number;
  r2: number;
};
export type AgingMethodRow = {
  method: string;
  mean_rmse: number;
  mean_correlation: number;
  folds: number;
};
export type ExternalRow = {
  scope: string;
  exposure: string;
  players: number | null;
  component: string;
  bpm: number | null;
  xrapm: number | null;
};
export type TeamProjection = {
  TEAM_ABBREVIATION: string;
  projection_season: number;
  players: number;
  projected_net_rating: number;
  projected_win_pace: number;
};
export type PlayerProjection = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  TEAM_ABBREVIATION: string;
  AGE: number;
  projected_offense: number;
  projected_defense: number;
  projected_net: number;
};

export type Catalog = {
  shards: number;
  catalog: {
    seasons: number[];
    role_seasons: Record<RoleSide, number[]>;
    role_labels: Record<RoleSide, Record<string, string>>;
  };
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

/* ------------------------------------------------------------------ read -- */

export const rating = (
  row: RatingRow | undefined,
  prefix: string,
  component: Component,
) => {
  const value = row?.[`${prefix}${component}`];
  return typeof value === "number" ? value : undefined;
};

/** The smaller of the two exposures — the honest sample size for a row. */
export const possessions = (row: { Poss_Off: number; Poss_Def: number }) =>
  Math.min(row.Poss_Off, row.Poss_Def);

/** A model is selectable only when the loaded snapshot carries its values. */
export function availableModels(rows: RatingRow[]) {
  return MODELS.map((model) => ({
    ...model,
    available:
      rows.length === 0 ||
      rows.some((row) => typeof row[`${model.prefix}net`] === "number"),
  }));
}

export function resolveModel(rows: RatingRow[], wanted: ModelId) {
  return (
    availableModels(rows).find(
      (item) => item.id === wanted && item.available,
    ) ?? MODELS[0]
  );
}

/** One honest sentence about models the loaded snapshot cannot show. */
export function missingModelNote(rows: RatingRow[]) {
  const missing = availableModels(rows)
    .filter((item) => !item.available)
    .map((item) => item.label);
  return missing.length
    ? `${missing.join(" and ")} need a rebuilt data snapshot from the pinned runs.`
    : "";
}

/* ------------------------------------------------------------------ load -- */

const cache = new Map<string, Promise<unknown>>();

function load<T>(url: string): Promise<T> {
  let pending = cache.get(url) as Promise<T> | undefined;
  if (!pending) {
    pending = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`${url} unavailable`);
      return response.json() as Promise<T>;
    });
    // A failed fetch must not poison the cache for a later retry.
    pending.catch(() => cache.delete(url));
    cache.set(url, pending);
  }
  return pending;
}

export const loadCatalog = () => load<Catalog>("/data/catalog.json");
export const loadIndex = () => load<PlayerIndex[]>("/data/players.json");
export const loadSeason = (season: number) =>
  load<LeaderboardRow[]>(`/data/leaderboard-${season}.json`);
export const loadShard = (shard: number) =>
  load<Record<string, Player>>(
    `/data/ratings-${String(shard).padStart(2, "0")}.json`,
  );
export const loadRoleMap = (side: RoleSide, season: number) =>
  load<RolePoint[]>(`/data/roles-${side}-${season}.json`);
export const loadTeamProjections = () =>
  load<TeamProjection[]>("/data/projection-teams.json");
export const loadPlayerProjections = () =>
  load<PlayerProjection[]>("/data/projection-players.json");

export async function loadPlayer(
  id: number,
  index: PlayerIndex[],
  shards: number,
) {
  const shard = index.find((item) => item.id === id)?.shard ?? id % shards;
  const data = await loadShard(shard);
  const found = data[String(id)];
  if (!found) throw new Error("player unavailable");
  return found;
}
