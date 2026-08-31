"""Research-only current RAPM score-ledger correction.

This keeps one canonical terminal-lineup row per possession. It rebuilds only
the response from raw score increments: technical free throws do not consume a
possession, and nontechnical points scored by the other team inside a listed
possession are excluded. The result is a clean sensitivity, not a production
replacement or a claim that technical FTs have no basketball value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficients,
    load_current_possessions,
    ratings_table,
)


ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data/lake/bronze/nba_data_archive/cdnnba"
GAME_DIM = ROOT / "data/lake/silver/game_dim.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
DEFAULT_OUTPUT = ROOT / "research/rapm_lab/outputs/scorer_owned_current_rapm"


def _raw_actions(bronze_root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["gameId", "orderNumber", "scoreHome", "scoreAway", "actionType", "description"]
    for season_end in seasons:
        path = bronze_root / f"season={season_end - 1}" / "regular.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw regular-season actions: {path}")
        frame = pd.read_parquet(path, columns=columns).copy()
        frame["season"] = int(season_end)
        frames.append(frame)
    actions = pd.concat(frames, ignore_index=True)
    actions["game_id"] = actions["gameId"].astype("int64").astype(str).str.zfill(10)
    return actions.drop(columns="gameId")


def _score_events(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Emit one scorer-owned row per positive home or away scoreboard increment."""
    work = actions.merge(
        games[["game_id", "season", "home_team_id", "away_team_id"]],
        on=["game_id", "season"], how="inner", validate="many_to_one",
    ).sort_values(["game_id", "orderNumber"], kind="stable")
    for side in ("Home", "Away"):
        score = pd.to_numeric(work[f"score{side}"], errors="coerce")
        score = score.groupby(work["game_id"]).ffill().fillna(0.0)
        work[f"{side.lower()}_delta"] = score.groupby(work["game_id"]).diff().fillna(score)
    negative = work["home_delta"].lt(0) | work["away_delta"].lt(0)
    if negative.any():
        bad = sorted(work.loc[negative, "game_id"].unique())
        raise ValueError(f"Raw scoreboard decreases require a separate repair: {bad[:10]}")
    parts: list[pd.DataFrame] = []
    for delta, owner in (("home_delta", "home_team_id"), ("away_delta", "away_team_id")):
        part = work.loc[work[delta] > 0].copy()
        part["scoring_team_id"] = part[owner].astype("int64")
        part["points"] = part[delta].astype(float)
        parts.append(part)
    events = pd.concat(parts, ignore_index=True)
    events["technical_ft"] = (
        events["actionType"].astype(str).str.lower().eq("freethrow")
        & events["description"].astype(str).str.contains("technical", case=False, na=False)
    )
    return events.sort_values(["game_id", "orderNumber"], kind="stable").reset_index(drop=True)


def _map_events_to_canonical_possessions(events: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    """Assign each scoring action to one canonical possession by latest start."""
    lookup = possessions[[
        "possession_id", "game_id", "season_end", "offense_team_id",
        "start_order_number", "end_order_number",
    ]].rename(columns={"season_end": "season"})
    left = events.sort_values(["orderNumber", "game_id"], kind="stable")
    right = lookup.sort_values(["start_order_number", "game_id"], kind="stable")
    mapped = pd.merge_asof(
        left, right, left_on="orderNumber", right_on="start_order_number",
        by=["game_id", "season"], direction="backward", allow_exact_matches=True,
    )
    mapped["mapped"] = (
        mapped["possession_id"].notna()
        & mapped["orderNumber"].le(mapped["end_order_number"])
    )
    return mapped


def build_adjusted_targets(
    actions: pd.DataFrame, games: pd.DataFrame, possessions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one corrected response per canonical possession and a game ledger."""
    events = _map_events_to_canonical_possessions(_score_events(actions, games), possessions)
    events["technical_points"] = np.where(events["technical_ft"], events["points"], 0.0)
    events["cross_owner_points"] = np.where(
        ~events["technical_ft"] & events["mapped"]
        & events["scoring_team_id"].ne(events["offense_team_id"]),
        events["points"], 0.0,
    )
    events["retained_points"] = np.where(
        ~events["technical_ft"] & events["mapped"]
        & events["scoring_team_id"].eq(events["offense_team_id"]),
        events["points"], 0.0,
    )
    # Technical points remain in their own ledger bucket even when the event
    # cannot map to a canonical possession. The categories must be disjoint.
    events["unmapped_points"] = np.where(
        ~events["mapped"] & ~events["technical_ft"], events["points"], 0.0
    )
    event_ledger = events.groupby(["game_id", "season"], as_index=False).agg(
        raw_score_points=("points", "sum"),
        retained_own_possession_points=("retained_points", "sum"),
        technical_ft_points=("technical_points", "sum"),
        cross_owner_nontechnical_points=("cross_owner_points", "sum"),
        unmapped_points=("unmapped_points", "sum"),
        score_events=("points", "size"),
    )
    official = games[["game_id", "season", "home_score", "away_score"]].copy()
    official["official_points"] = official["home_score"] + official["away_score"]
    ledger = official.merge(event_ledger, on=["game_id", "season"], how="left").fillna(0.0)
    decomposition = (
        ledger["retained_own_possession_points"] + ledger["technical_ft_points"]
        + ledger["cross_owner_nontechnical_points"] + ledger["unmapped_points"]
    )
    ledger["raw_score_conserved"] = np.isclose(ledger["raw_score_points"], ledger["official_points"])
    ledger["decomposition_conserved"] = np.isclose(decomposition, ledger["official_points"])
    retained = events.loc[events["mapped"]].groupby("possession_id", as_index=False)["retained_points"].sum()
    target = possessions[["possession_id", "game_id", "possession_number", "season_end", "home_team_id", "away_team_id", "offense_team_id"]].copy()
    target = target.merge(retained, on="possession_id", how="left")
    target["retained_points"] = target["retained_points"].fillna(0.0)
    target["home_retained_points"] = np.where(target["offense_team_id"].eq(target["home_team_id"]), target["retained_points"], 0.0)
    target["away_retained_points"] = np.where(target["offense_team_id"].eq(target["away_team_id"]), target["retained_points"], 0.0)
    adjusted = target.groupby(["game_id", "season_end"], as_index=False).agg(
        adjusted_home_score=("home_retained_points", "sum"),
        adjusted_away_score=("away_retained_points", "sum"),
    ).rename(columns={"season_end": "season"})
    ledger = ledger.merge(adjusted, on=["game_id", "season"], how="left").fillna(0.0)
    ledger["adjusted_margin"] = ledger["adjusted_home_score"] - ledger["adjusted_away_score"]
    return target[["game_id", "possession_number", "retained_points"]], ledger.sort_values(["season", "game_id"], kind="stable").reset_index(drop=True)


def _bootstrap(predictions: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    base = (predictions["actual_margin"] - predictions["baseline_prediction"]).to_numpy() ** 2
    corrected = (predictions["actual_margin"] - predictions["corrected_prediction"]).to_numpy() ** 2
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        index = rng.integers(0, len(base), len(base))
        values[draw] = float(base[index].mean() - corrected[index].mean())
    return pd.DataFrame({"draw": np.arange(draws), "baseline_minus_corrected_mse": values}), {
        "draws": draws, "seed": seed, "mean_mse_improvement": float(values.mean()),
        "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975)),
        "probability_mse_improvement": float((values > 0).mean()),
    }


def run(*, seasons: tuple[int, ...], output_root: Path, draws: int, seed: int, smoke: bool) -> dict:
    games = pd.read_parquet(GAME_DIM)
    games = games.loc[(games["season_type"] == "regular") & games["season_end"].isin(seasons)].copy()
    games["game_id"] = games["game_id"].astype(str)
    games = games.rename(columns={"season_end": "season"})
    possessions = pd.read_parquet(POSSESSIONS)
    possessions["game_id"] = possessions["game_id"].astype(str)
    possessions = possessions.loc[possessions["game_id"].isin(games["game_id"])].copy()
    targets, ledger = build_adjusted_targets(_raw_actions(BRONZE, seasons), games, possessions)
    # These are source-coverage failures, not ordinary technical/cross-owner
    # events. Exclude them symmetrically rather than assigning missing points
    # to a zero response.
    eligible_games = set(ledger.loc[
        ledger["unmapped_points"].eq(0)
        & ledger["raw_score_conserved"]
        & ledger["decomposition_conserved"],
        "game_id",
    ])
    if not eligible_games:
        raise ValueError("No games have complete canonical score-action coverage.")
    identity_payload = {"script_hash": sha256_file(Path(__file__)), "seasons": seasons, "draws": draws, "smoke": smoke}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, default=list).encode()).hexdigest()[:10]
    output = output_root / f"score_ledger_current_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output / "adjusted_possession_targets.parquet", index=False)
    ledger.to_parquet(output / "game_score_ledger.parquet", index=False)
    run_record: dict = {
        "run_id": output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_smoke" if smoke else "research_challenger",
        "scope": "2024-26 canonical possession X with technical/cross-owner score exclusion",
        "script_hash": identity_payload["script_hash"], "seasons": list(seasons),
        "quality": {"games": int(len(ledger)), "eligible_games": int(len(eligible_games)), "technical_ft_points": float(ledger["technical_ft_points"].sum()), "cross_owner_nontechnical_points": float(ledger["cross_owner_nontechnical_points"].sum()), "unmapped_points": float(ledger["unmapped_points"].sum())},
        "sources": {"game_dim": sha256_file(GAME_DIM), "possessions": sha256_file(POSSESSIONS), "segments": sha256_file(SEGMENTS), **{f"raw_{season}": sha256_file(BRONZE / f"season={season - 1}" / "regular.parquet") for season in seasons}},
        "artifacts": {"adjusted_targets": "adjusted_possession_targets.parquet", "score_ledger": "game_score_ledger.parquet"},
        "forbidden_interpretation": "Research-only response sensitivity. It excludes technical free throws because they do not consume a possession and excludes unresolved cross-owner nontechnical scores.",
    }
    if smoke:
        write_json_atomic(run_record, output / "run.json")
        return run_record
    if seasons != (2024, 2025, 2026):
        raise ValueError("The frozen full comparison requires exactly 2024, 2025, and 2026.")
    baseline = load_current_possessions(POSSESSIONS, SEGMENTS, lineup_policy="terminal")
    baseline = baseline.loc[
        baseline["season"].isin(seasons) & baseline["gameid"].isin(eligible_games)
    ].copy()
    corrected = baseline.merge(
        targets, left_on=["gameid", "num"], right_on=["game_id", "possession_number"],
        how="left", validate="one_to_one",
    )
    if corrected["retained_points"].isna().any():
        raise ValueError("Every canonical possession must receive an adjusted response.")
    corrected["pts"] = corrected["retained_points"]
    corrected = corrected.drop(columns=["game_id", "possession_number", "retained_points"])
    baseline_design, corrected_design = build_design(baseline), build_design(corrected)
    if baseline_design.X.shape != corrected_design.X.shape or (baseline_design.X != corrected_design.X).nnz:
        raise ValueError("The corrected response must preserve the canonical RAPM design matrix.")
    config = RapmConfig(seasons=(2024, 2025), lambda_off=3000.0, lambda_def=3000.0, lambda_home=300.0)
    train, test = baseline_design.seasons < 2026, baseline_design.seasons == 2026
    base_beta, base_intercept = fit_coefficients(baseline_design, config, train)
    corr_beta, corr_intercept = fit_coefficients(corrected_design, config, train)
    base_games = _game_margin_frame(baseline_design, base_beta, base_intercept, test, train).rename(columns={"predicted_margin": "baseline_prediction"})
    corr_games = _game_margin_frame(corrected_design, corr_beta, corr_intercept, test, train).rename(columns={"predicted_margin": "corrected_prediction"})
    actual = ledger.loc[
        ledger["season"].eq(2026) & ledger["game_id"].isin(eligible_games),
        ["game_id", "adjusted_margin"],
    ]
    predictions = base_games[["game_id", "baseline_prediction"]].merge(corr_games[["game_id", "corrected_prediction"]], on="game_id", validate="one_to_one").merge(actual, on="game_id", validate="one_to_one").rename(columns={"adjusted_margin": "actual_margin"})
    draws_frame, bootstrap = _bootstrap(predictions, draws, seed)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    draws_frame.to_parquet(output / "paired_bootstrap_draws.parquet", index=False)
    ratings_table(baseline_design, base_beta).to_parquet(output / "baseline_ratings.parquet", index=False)
    ratings_table(corrected_design, corr_beta).to_parquet(output / "corrected_ratings.parquet", index=False)
    run_record["comparison"] = {"train_seasons": [2024, 2025], "test_season": 2026, "fixed_lambdas": {"offense": 3000, "defense": 3000, "home": 300}, "identical_2026_games": int(len(predictions)), "design_matrix_identical": True, "paired_whole_game_bootstrap": bootstrap}
    run_record["artifacts"].update({"predictions": "game_predictions.parquet", "bootstrap": "paired_bootstrap_draws.parquet", "baseline_ratings": "baseline_ratings.parquet", "corrected_ratings": "corrected_ratings.parquet"})
    write_json_atomic(run_record, output / "run.json")
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seasons = (2024,) if args.smoke else (2024, 2025, 2026)
    print(json.dumps(run(seasons=seasons, output_root=args.output_root, draws=args.draws, seed=args.seed, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
