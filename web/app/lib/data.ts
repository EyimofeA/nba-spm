/** Types and cached loaders for the published web snapshot. */

export type Component = "net" | "offense" | "defense";
export type ModelId = "aio" | "rapm" | "spm";
export type RoleSide = "offense" | "defense";
export type MatchupRow = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  TEAM_ABBREVIATION: string | null;
  Season: number;
  offense_elo: number;
  defense_elo: number;
  net_elo: number;
  offense_matchup_possessions: number;
  defense_matchup_possessions: number;
  window_start_season: number;
  window_end_season: number;
  time_decay: number;
};
export type ShotQualityRow = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  TEAM_ABBREVIATION: string | null;
  Season: number;
  raw_net_rank: number;
  shot_quality_net_rank: number;
  rank_change: number;
  lineup_offense_shotmaking_per_100_shots: number;
  lineup_defense_contest_per_100_shots: number;
  lineup_net_residual_per_100_shots: number;
  lineup_shots: number;
  shooter_quality_above_league_per_100_shots: number | null;
  shooter_shotmaking_above_quality_per_100_shots: number | null;
};
export type SpmLabRating = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  metric: "spm" | "aio";
  selected_offense: number;
  selected_defense: number;
  selected_net: number;
  baseline_offense: number;
  baseline_defense: number;
  baseline_net: number;
  delta_offense: number;
  delta_defense: number;
  delta_net: number;
};
export type Box15LeaderboardRow = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  Season: number;
  prior_offense: number;
  prior_defense: number;
  prior_net: number;
  posterior_offense: number;
  posterior_defense: number;
  posterior_net: number;
  change_net: number;
};
export type Box15Correlation = {
  component: "offense" | "defense" | "net";
  metric: string;
  metric_label: string;
  prior_rows: number;
  posterior_rows: number;
  prior_pearson: number;
  posterior_pearson: number;
  prior_spearman: number;
  posterior_spearman: number;
};
export type SpmMetricDefinition = {
  metric: string;
  metric_label: string;
  included: boolean;
  kind: string;
  how_it_works: string;
  interpretation: string;
};
export type SpmTeamWinSummary = {
  metric: string;
  metric_label: string;
  replacement_value: number;
  folds: number;
  team_seasons: number;
  mean_pearson: number;
  mean_spearman: number;
  mean_r_squared: number;
  pooled_pearson: number;
  pooled_spearman: number;
  pooled_r_squared: number;
  minimum_minute_coverage: number;
};
export type SpmPairwiseCorrelation = {
  component: "offense" | "defense" | "net";
  left_metric: string;
  right_metric: string;
  rows: number;
  seasons: number;
  pearson: number;
  spearman: number;
};
export type SpmLabPayload = {
  run_id: string;
  scope: "localhost_only";
  seasons: number[];
  stabilization: Record<string, string | boolean>;
  selection_gate: Record<string, string | number | number[] | boolean>;
  decisions: {
    group: string;
    side: "offense" | "defense";
    feature_count: number;
    development_mean_rmse_delta: number;
    development_fold_wins: number;
    team_changer_mean_rmse_delta: number;
    team_changer_mean_correlation_delta: number;
    selected: boolean;
  }[];
  validation: {
    test_season: number;
    baseline_rmse: number;
    selected_rmse: number;
    rmse_delta: number;
    baseline_correlation: number;
    selected_correlation: number;
  }[];
  ratings: SpmLabRating[];
  box15: {
    run_id: string;
    seasons: number[];
    correlation_seasons: number[];
    minimum_minutes: number;
    leaderboard: Box15LeaderboardRow[];
    correlations: Box15Correlation[];
  };
  weighting: {
    run_id: string;
    summary: {
      variant: "sqrt_possessions" | "unweighted";
      component: "offense" | "defense" | "net";
      evaluation: "sqrt_possessions" | "equal_players";
      folds: number;
      mean_rmse: number;
      mean_correlation: number;
    }[];
    fold_metrics: Record<string, string | number>[];
    feature_catalog: {
      feature: string;
      side: "offense" | "defense" | "both";
      description: string;
      offense_input: boolean;
      defense_input: boolean;
    }[];
    quality: {
      rows: number;
      duplicate_keys: number;
      feature_catalog_rows: number;
      weighted_reference_reproduction: Record<string, Record<string, number>>;
    };
    caveats: string[];
  };
  comparison: {
    run_id: string;
    common_seasons: number[];
    minimum_metric_year_minutes: number;
    replacement_value: number;
    team_rating_formula: string;
    minutes_mode: "observed_next_season";
    projected_minutes_status: string;
    team_win_summary: SpmTeamWinSummary[];
    team_win_folds: {
      rating_season: number;
      outcome_season: number;
      metric: string;
      metric_label: string;
      replacement_value: number;
      teams: number;
      pearson: number;
      spearman: number;
      r_squared: number;
      mean_minute_coverage: number;
    }[];
    pairwise_correlations: SpmPairwiseCorrelation[];
    coverage: Record<string, string | number>[];
    definitions: SpmMetricDefinition[];
    caveats: string[];
  };
};
export type RapmLabExperiment = {
  id: string;
  title: string;
  test: string;
  result: string;
  decision: string;
  status: "won" | "lost" | "built" | "estimate";
  run_id: string;
};
export type RapmLabLeaderboard = {
  id: string;
  experiment_id: string;
  title: string;
  columns: { key: string; label: string }[];
  rows: Record<string, string | number | null>[];
};
export type RubberbandCoefficient = {
  fit_seasons: string;
  time_bucket: number;
  minutes_elapsed_start: number;
  minutes_elapsed_end: number;
  slope_points_per_100_per_margin_point: number;
  standard_error: number;
  lower_95: number;
  upper_95: number;
};
export type RubberbandCurveRow = {
  time_bucket: number;
  minutes_elapsed_start: number;
  minutes_elapsed_end: number;
  offense_margin_before: number;
  adjustment_points_per_100: number;
};
export type RubberbandComparisonCoefficient = {
  basis: "actual_clock" | "possession_progress";
  time_bucket: number;
  minutes_elapsed_start: number | null;
  minutes_elapsed_end: number | null;
  possessions_elapsed_start: number | null;
  possessions_elapsed_end: number | null;
  slope_points_per_100_per_margin_point: number;
  standard_error: number;
  lower_95: number;
  upper_95: number;
};
export type RubberbandRating = {
  player_id: number;
  player_name: string;
  off_possessions: number;
  def_possessions: number;
  normal_offense: number;
  normal_defense: number;
  normal_net: number;
  clock_offense: number;
  clock_defense: number;
  clock_net: number;
  possession_offense: number;
  possession_defense: number;
  possession_net: number;
  clock_net_change: number;
  possession_net_change: number;
};
export type RubberbandJeCurveRow = {
  margin: number;
  effect_points_per_100_vs_tie: number;
  possessions: number;
  state: "Trailing" | "Leading" | "Tied";
};
export type AgeCurveRow = {
  age: number;
  offense: number;
  defense: number;
  net: number;
};
export type RapmLabPayload = {
  generated_at: string;
  scope: "localhost_only";
  experiments: RapmLabExperiment[];
  leaderboards: RapmLabLeaderboard[];
  age: {
    run_id: string;
    curve: AgeCurveRow[];
    selection: {
      age_penalty: number;
      games: number;
      margin_rmse: number;
      margin_mae: number;
      margin_correlation: number;
      possession_rmse: number;
    }[];
    evaluation: {
      variant: "normal" | "same_age_27" | "age_conditional";
      games: number;
      margin_rmse: number;
      margin_mae: number;
      margin_correlation: number;
      possession_rmse: number;
    }[];
    bootstrap_vs_normal: Record<
      "same_age_27" | "age_conditional",
      {
        observed_rmse_delta: number;
        lower_95: number;
        upper_95: number;
        probability_improvement: number;
      }
    >;
    rating_effect: {
      net_correlation_with_normal: number;
      mean_absolute_net_change: number;
    };
    quality: {
      possessions: number;
      games: number;
      age_slot_coverage: number;
      qualified_players: number;
    };
  };
  rubberband: {
    run_id: string;
    selected_spec: string;
    margin_clip: number;
    possessions: number;
    games: number;
    expected_games: number;
    game_coverage_rate: number;
    diagnostic: {
      rows: number;
      games: number;
      baseline_rmse: number;
      candidate_rmse: number;
      mean_squared_error_improvement: number;
      residual_correlation: number;
    };
    bootstrap: {
      draws: number;
      mean_mse_improvement: number;
      lower_95: number;
      upper_95: number;
      probability_improvement: number;
    };
    minimum_season_correlation: number;
    selection_winner_vs_runner_up: {
      winner: string;
      runner_up: string;
      lower_95: number;
      upper_95: number;
      probability_improvement: number;
    };
    coefficients: RubberbandCoefficient[];
    curve: RubberbandCurveRow[];
    comparison_run_id: string;
    comparison_coefficients: RubberbandComparisonCoefficient[];
    context_effect: {
      clock: {
        baseline_rmse: number;
        candidate_rmse: number;
        mean_squared_error_improvement: number;
      };
      possession_progress: {
        baseline_rmse: number;
        candidate_rmse: number;
        mean_squared_error_improvement: number;
      };
      slope_correlation: number;
    };
    rapm_evaluation: {
      variant: "normal" | "clock" | "possession_progress";
      prediction_mode: "neutral_player_only" | "conditional_score_path";
      games: number;
      margin_rmse: number;
      margin_mae: number;
      margin_correlation: number;
    }[];
    rapm_bootstrap_vs_normal: Record<
      "clock" | "possession_progress",
      {
        observed_rmse_delta: number;
        lower_95: number;
        upper_95: number;
        probability_improvement: number;
      }
    >;
    conditional_rapm_bootstrap_vs_normal: Record<
      "clock" | "possession_progress",
      {
        observed_rmse_delta: number;
        lower_95: number;
        upper_95: number;
        probability_improvement: number;
      }
    >;
    rating_effect: {
      clock_net_correlation_with_normal: number;
      possession_net_correlation_with_normal: number;
      clock_mean_absolute_net_change: number;
      possession_mean_absolute_net_change: number;
    };
    je: {
      run_id: string;
      fit_seasons: number[];
      curve: RubberbandJeCurveRow[];
      effects: Record<string, number>;
      evaluation: {
        variant: "normal" | "je_categorical";
        prediction_mode: "neutral_player_only" | "conditional_score_path";
        possession_rmse: number;
        games: number;
        margin_rmse: number;
        margin_mae: number;
        margin_correlation: number;
      }[];
      bootstrap_vs_normal: Record<
        "neutral_player_only" | "conditional_score_path",
        {
          observed_rmse_delta: number;
          lower_95: number;
          upper_95: number;
          probability_improvement: number;
        }
      >;
      rating_effect: {
        net_correlation_with_normal: number;
        mean_absolute_net_change: number;
      };
    };
    five_point: {
      run_id: string;
      curve_run_id: string;
      curve: RubberbandJeCurveRow[];
      selection_winner: {
        margin_rmse: number;
        margin_correlation: number;
        lambda_off: number;
        lambda_def: number;
        score_control: string;
      };
      diagnostic: {
        rmse_delta_candidate_minus_baseline: number;
        paired_game_bootstrap: {
          observed_rmse_delta: number;
          lower_95: number;
          upper_95: number;
          probability_improvement: number;
        };
      };
      decision: string;
    };
    ratings: RubberbandRating[];
    test: string;
    decision: string;
  };
};

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

export type LocalSkillDefinition = {
  key: string;
  label: string;
  group: "shooting" | "creation" | "rebounding" | "defense";
  unit: string;
  higherIsBetter: boolean;
  definition: string;
};
export type LocalSkillIndex = {
  schema: "courtsignal_local_player_skills_v1";
  scope: "localhost_only";
  runId: string;
  defaultPlayerId: number;
  seasons: number[];
  players: { id: number; name: string; team: string | null; complete2026: boolean }[];
  definitions: LocalSkillDefinition[];
  profileAxes: Record<string, string[]>;
  gameSkills: string[];
  league: Record<string, [season: number, estimate: number | null][]>;
};
export type LocalSkillRow = [
  season: number,
  estimate: number | null,
  raw: number | null,
  opportunities: number | null,
  percentile: number | null,
  yearOverYear: number | null,
  standardError: number | null,
  lastUpdate: string,
];
export type LocalPlayerSkill = {
  arm: string;
  halfLife: number | null;
  prior: number | null;
  rows: LocalSkillRow[];
};
export type LocalGameSkillRow = {
  date: string;
  game: number | null;
  played: boolean;
  raw: number | null;
  estimate: number | null;
  opportunities: number | null;
};
export type LocalPlayerSkills = {
  id: number;
  name: string;
  team: string | null;
  complete2026: boolean;
  skills: Record<string, LocalPlayerSkill>;
  profiles: ({ season: number } & Record<string, number | null>)[];
  games: Record<string, LocalGameSkillRow[]>;
};

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
export const loadMatchups = (season: number) =>
  load<MatchupRow[]>(`/data/matchup-elo-${season}.json`);
export const loadShotQualityMatchups = () =>
  load<ShotQualityRow[]>("/data/shot-quality-lineup-2026.json");
export const loadRapmLab = () => load<RapmLabPayload>("/data/rapm-lab.json");
export const loadSpmLab = () => load<SpmLabPayload>("/data/spm-lab.json");
export const loadLocalSkillIndex = () =>
  load<LocalSkillIndex>("/data/skills/index.json");
export const loadLocalPlayerSkills = (id: number) =>
  load<LocalPlayerSkills>(`/data/skills/player-${id}.json`);

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
