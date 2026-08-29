"""Transparent shot-quality, shooting-threat, and lineup-spacing research."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.observable_play_channels import _shot_finish
from nba_impact.models.expected_shot_quality import (
    _player_aggregates,
    fit_and_predict_expected_shots,
)
from nba_impact.models.possession_outcome_rapm import assign_events_to_possessions


MODEL_VERSION = "shot_model_suite_v1"
DIMENSIONS = {
    "release": (("CATCH_SHOOT_FG3A", "CATCH_SHOOT_FG3M"), ("PULL_UP_FG3A", "PULL_UP_FG3M")),
    "contest": tuple(
        (f"{label}_FG3A", f"{label}_FG3M")
        for label in ("very_tight", "tight", "open", "wide_open")
    ),
    "location": (("Corner3FGA", "Corner3FGM"), ("Arc3FGA", "Arc3FGM")),
}


def _text(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(item) for item in value)
    return "" if pd.isna(value) else str(value)


def enrich_shots(
    shots: pd.DataFrame,
    possessions: pd.DataFrame,
    raw_paths: tuple[Path, ...],
) -> pd.DataFrame:
    """Attach outcome-safe possession and event context to current shot rows."""
    raw = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "gameId",
                    "actionNumber",
                    "description",
                    "qualifiers",
                ],
            )
            for path in raw_paths
        ],
        ignore_index=True,
    )
    raw["game_id"] = raw["gameId"].map(lambda value: f"{int(value):010d}")
    raw["actionNumber"] = pd.to_numeric(raw["actionNumber"], errors="raise").astype("int64")
    raw["qualifier_text"] = raw["qualifiers"].map(_text).str.casefold()
    raw = raw.drop_duplicates(["game_id", "actionNumber"], keep="last")
    mapped = assign_events_to_possessions(
        possessions,
        shots[
            [
                "game_id",
                "actionNumber",
                "period",
                "seconds_remaining_period",
            ]
        ].drop_duplicates().copy(),
    )
    mapped = mapped.merge(
        possessions[
            [
                "possession_id",
                "start_seconds_elapsed",
            ]
        ],
        on="possession_id",
        validate="many_to_one",
    )
    period = mapped["period"].to_numpy(dtype=int)
    remaining = mapped["seconds_remaining_period"].to_numpy(dtype=float)
    shot_elapsed = np.where(
        period <= 4,
        (period - 1) * 720.0 + (720.0 - remaining),
        2880.0 + (period - 5) * 300.0 + (300.0 - remaining),
    )
    mapped["seconds_since_possession_start"] = (
        shot_elapsed - mapped["start_seconds_elapsed"].to_numpy(dtype=float)
    ).clip(min=0.0)
    context = raw[
        ["game_id", "actionNumber", "description", "qualifier_text"]
    ].merge(
        mapped[
            ["game_id", "actionNumber", "seconds_since_possession_start"]
        ],
        on=["game_id", "actionNumber"],
        how="left",
        validate="one_to_one",
    )
    context["description"] = context["description"].astype("string")
    context["shot_finish"] = _shot_finish(
        context["description"], context["qualifier_text"].astype("string")
    )
    context["is_transition"] = (
        context["shot_finish"].eq("transition")
        | context["qualifier_text"].str.contains("fastbreak", na=False)
    )
    context["is_putback"] = context["shot_finish"].eq("putback")
    context["is_second_chance"] = context["qualifier_text"].str.contains(
        "2ndchance|secondchance", regex=True, na=False
    )
    context["is_from_turnover"] = context["qualifier_text"].str.contains(
        "fromturnover", na=False
    )
    output = shots.merge(
        context.drop(columns="description"),
        on=["game_id", "actionNumber"],
        how="left",
        validate="one_to_one",
    )
    output["seconds_since_possession_start"] = output[
        "seconds_since_possession_start"
    ].fillna(0.0)
    output["shot_finish"] = output["shot_finish"].fillna("other")
    for field in (
        "is_transition",
        "is_putback",
        "is_second_chance",
        "is_from_turnover",
    ):
        output[field] = output[field].fillna(False)
    return output


def _shot_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, candidate: str, split: str
) -> dict[str, object]:
    outcome = frame["shot_made"].to_numpy(dtype=int)
    return {
        "candidate": candidate,
        "split": split,
        "shots": len(frame),
        "brier": float(brier_score_loss(outcome, prediction)),
        "log_loss": float(log_loss(outcome, prediction, labels=[0, 1])),
    }


def fit_context_comparison(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel.loc[panel["season_end"].eq(2024)].copy()
    calibration = panel.loc[panel["season_end"].eq(2025)].copy()
    test = panel.loc[panel["season_end"].eq(2026)].copy()
    rows = []
    predictions = test[
        ["shot_id", "game_id", "season_end", "shooter_id", "shot_zone", "shot_value", "shot_made"]
    ].copy()
    for candidate, include_context in (("location", False), ("possession_context", True)):
        _, calibrated = fit_and_predict_expected_shots(
            train,
            calibration,
            test,
            include_possession_context=include_context,
        )
        predictions[candidate] = calibrated
        for split, mask in (
            ("all", np.ones(len(test), dtype=bool)),
            ("rim", test["shot_zone"].eq("rim").to_numpy()),
            ("non_rim", test["shot_zone"].ne("rim").to_numpy()),
        ):
            rows.append(_shot_metrics(test.loc[mask], calibrated[mask], candidate, split))
    players = _player_aggregates(test, predictions["possession_context"].to_numpy())
    return predictions, pd.DataFrame(rows), players


def paired_shot_bootstrap(
    predictions: pd.DataFrame, *, draws: int = 5000, seed: int = 20260829
) -> pd.DataFrame:
    """Compare shot models by resampling whole games, not individual shots."""
    actual = predictions["shot_made"].to_numpy(dtype=float)
    clipped = predictions[["location", "possession_context"]].clip(1e-9, 1 - 1e-9)
    rows = []
    for metric in ("brier", "log_loss"):
        if metric == "brier":
            losses = (clipped.sub(actual, axis=0)) ** 2
        else:
            losses = pd.DataFrame(
                -(
                    actual[:, None] * np.log(clipped.to_numpy())
                    + (1 - actual[:, None]) * np.log(1 - clipped.to_numpy())
                ),
                columns=clipped.columns,
            )
        game = losses.assign(game_id=predictions["game_id"].to_numpy()).groupby(
            "game_id", as_index=False
        ).mean()
        delta = game["possession_context"].to_numpy() - game["location"].to_numpy()
        rng = np.random.default_rng(seed)
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = delta[rng.integers(0, len(delta), len(delta))].mean()
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "games": len(delta),
                "context_minus_location": float(delta.mean()),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_context_better": float(np.mean(samples < 0)),
                "bootstrap_draws": draws,
                "resampling_unit": "whole_game",
            }
        )
    return pd.DataFrame(rows)


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame.get(field, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)


def annual_shooting_threat(player_sheets: tuple[Path, ...], prior_attempts: float = 100.0) -> pd.DataFrame:
    """Estimate context-aware 3-point ability and its points-per-100 threat."""
    frames = []
    for path in player_sheets:
        frame = pd.read_parquet(path)
        season = int(path.stem)
        needed = {"PLAYER_ID", "PLAYER_NAME", "FG3A", "FG3M", "OffPoss"}
        if missing := sorted(needed - set(frame.columns)):
            raise ValueError(f"{path} lacks shooting-threat fields: {missing}")
        keep = frame[["PLAYER_ID", "PLAYER_NAME"]].copy()
        keep["Season"] = season
        keep["FG3A"] = _numeric(frame, "FG3A")
        keep["FG3M"] = _numeric(frame, "FG3M")
        keep["OffPoss"] = _numeric(frame, "OffPoss")
        for dimension, bins in DIMENSIONS.items():
            for attempts, makes in bins:
                keep[attempts] = _numeric(frame, attempts)
                keep[makes] = _numeric(frame, makes)
        duplicated = keep.duplicated("PLAYER_ID", keep=False)
        if duplicated.any():
            value_columns = [field for field in keep.columns if field not in {"PLAYER_ID", "PLAYER_NAME", "Season"}]
            inconsistent = keep.loc[duplicated].groupby("PLAYER_ID")[value_columns].nunique(dropna=False).gt(1).any(axis=1)
            if inconsistent.any():
                raise ValueError(f"{path} has conflicting duplicate player totals.")
            keep = keep.drop_duplicates("PLAYER_ID", keep="first")
        frames.append(keep)
    panel = pd.concat(frames, ignore_index=True)
    output = []
    for season, frame in panel.groupby("Season", sort=True):
        frame = frame.copy()
        league_attempts = frame["FG3A"].sum()
        league_makes = frame["FG3M"].sum()
        league_rate = float(league_makes / league_attempts)
        expected_logits = []
        availability = []
        for dimension, bins in DIMENSIONS.items():
            expected_makes = np.zeros(len(frame), dtype=float)
            attempts_used = np.zeros(len(frame), dtype=float)
            for attempts, makes in bins:
                attempts_values = frame[attempts].to_numpy(dtype=float)
                makes_values = frame[makes].to_numpy(dtype=float)
                total_attempts = attempts_values.sum()
                total_makes = makes_values.sum()
                denominator = total_attempts - attempts_values
                loo = np.divide(
                    total_makes - makes_values,
                    denominator,
                    out=np.full(len(frame), league_rate),
                    where=denominator > 0,
                )
                expected_makes += attempts_values * loo
                attempts_used += attempts_values
            rate = np.divide(
                expected_makes,
                attempts_used,
                out=np.full(len(frame), league_rate),
                where=attempts_used > 0,
            )
            frame[f"{dimension}_expected_3p_pct"] = rate
            expected_logits.append(logit(np.clip(rate, 1e-5, 1 - 1e-5)))
            availability.append(attempts_used > 0)
        league_logit = logit(np.clip(league_rate, 1e-5, 1 - 1e-5))
        context_logit = league_logit + np.mean(
            np.column_stack(expected_logits) - league_logit, axis=1
        )
        context_rate = expit(context_logit)
        attempts = frame["FG3A"].to_numpy(dtype=float)
        makes = frame["FG3M"].to_numpy(dtype=float)
        ability = (makes + prior_attempts * context_rate) / (attempts + prior_attempts)
        attempts_p100 = np.divide(
            100.0 * attempts,
            frame["OffPoss"].to_numpy(dtype=float),
            out=np.zeros(len(frame)),
            where=frame["OffPoss"].to_numpy(dtype=float) > 0,
        )
        frame["league_3p_pct"] = league_rate
        frame["context_expected_3p_pct"] = context_rate
        frame["shooting_ability_3p_pct_eb"] = ability
        frame["three_pa_p100"] = attempts_p100
        frame["shooting_threat_p100"] = 3.0 * attempts_p100 * (ability - league_rate)
        frame["context_shotmaking_p100"] = 3.0 * attempts_p100 * (ability - context_rate)
        frame["context_dimensions_available"] = np.column_stack(availability).sum(axis=1)
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def five_year_shooting_threat(annual: pd.DataFrame, prior_attempts: float = 100.0) -> pd.DataFrame:
    rows = []
    for window_end in sorted(annual["Season"].unique()):
        window = annual.loc[annual["Season"].between(window_end - 4, window_end)].copy()
        for player_id, frame in window.groupby("PLAYER_ID", sort=False):
            attempts = frame["FG3A"].sum()
            off_possessions = frame["OffPoss"].sum()
            expected_makes = (frame["context_expected_3p_pct"] * frame["FG3A"]).sum()
            expected_rate = expected_makes / attempts if attempts else frame["league_3p_pct"].mean()
            ability = (frame["FG3M"].sum() + prior_attempts * expected_rate) / (attempts + prior_attempts)
            league_rate = np.average(
                frame["league_3p_pct"], weights=frame["FG3A"].clip(lower=1)
            )
            attempts_p100 = 100.0 * attempts / off_possessions if off_possessions else 0.0
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "PLAYER_NAME": frame["PLAYER_NAME"].iloc[-1],
                    "Window_End": int(window_end),
                    "shooting_threat_p100": 3.0 * attempts_p100 * (ability - league_rate),
                    "context_shotmaking_p100": 3.0 * attempts_p100 * (ability - expected_rate),
                    "shooting_ability_3p_pct_eb": ability,
                    "context_expected_3p_pct": expected_rate,
                    "three_pa_p100": attempts_p100,
                    "five_year_three_pa": attempts,
                }
            )
    return pd.DataFrame(rows)


def lineup_spacing(
    segments: pd.DataFrame,
    possessions: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    """Average the other four players' annual threat over lineup segments."""
    segment = segments.merge(
        possessions[["possession_id", "season_end"]],
        on="possession_id",
        validate="many_to_one",
    )
    segment = segment.loc[segment["season_end"].isin(annual["Season"].unique())].copy()
    weight = (segment["end_seconds_elapsed"] - segment["start_seconds_elapsed"]).clip(lower=1.0).to_numpy()
    threat = annual.set_index(["Season", "PLAYER_ID"])["shooting_threat_p100"].to_dict()
    pieces = []
    for prefix in ("home", "away"):
        columns = [f"{prefix}_player_{index}" for index in range(1, 6)]
        players = segment[columns].to_numpy(dtype=np.int64)
        values = np.column_stack(
            [
                np.fromiter(
                    (threat.get((int(season), int(player)), 0.0) for season, player in zip(segment["season_end"], players[:, slot], strict=True)),
                    dtype=float,
                    count=len(segment),
                )
                for slot in range(5)
            ]
        )
        total = values.sum(axis=1)
        for slot in range(5):
            pieces.append(
                pd.DataFrame(
                    {
                        "Season": segment["season_end"].to_numpy(dtype=int),
                        "PLAYER_ID": players[:, slot],
                        "weighted_teammate_spacing": ((total - values[:, slot]) / 4.0) * weight,
                        "segment_seconds": weight,
                    }
                )
            )
    output = pd.concat(pieces, ignore_index=True).groupby(
        ["Season", "PLAYER_ID"], as_index=False
    ).sum()
    output["teammate_shooting_threat"] = (
        output["weighted_teammate_spacing"] / output["segment_seconds"]
    )
    return output.drop(columns="weighted_teammate_spacing")


def run_suite(root: Path) -> dict:
    shot_path = root / "data/lake/silver/shot_defense_events.parquet"
    possession_path = root / "data/lake/silver/possessions.parquet"
    segment_path = root / "data/lake/silver/possession_lineup_segments.parquet"
    raw_paths = tuple(sorted((root / "data/lake/bronze/nba_data_archive/cdnnba").glob("season=*/regular.parquet")))
    player_sheets = tuple(
        root / f"data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals/{season}.parquet"
        for season in range(2014, 2027)
    )
    if len(raw_paths) != 3 or not all(path.exists() for path in player_sheets):
        raise FileNotFoundError("The pinned shot or annual player sources are incomplete.")
    shots = pd.read_parquet(shot_path)
    possessions = pd.read_parquet(possession_path)
    enriched = enrich_shots(shots, possessions, raw_paths)
    predictions, shot_metrics, player_shots = fit_context_comparison(enriched)
    shot_bootstrap = paired_shot_bootstrap(predictions)
    annual = annual_shooting_threat(player_sheets)
    five_year = five_year_shooting_threat(annual)
    spacing = lineup_spacing(pd.read_parquet(segment_path), possessions, annual)
    sources = (shot_path, possession_path, segment_path, *raw_paths, *player_sheets)
    config = {
        "model_version": MODEL_VERSION,
        "shot_split": {"train": [2024], "calibration": [2025], "test": [2026]},
        "evidence_status": "reused_2026_diagnostic",
        "shooting_threat_prior_attempts": 100.0,
        "source_hashes": {str(path.relative_to(root)): sha256_file(path) for path in sources},
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = root / "artifacts/research/shot_model_suite" / f"{MODEL_VERSION}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "shot_predictions.parquet": predictions,
        "shot_metrics.parquet": shot_metrics,
        "shot_paired_bootstrap.parquet": shot_bootstrap,
        "player_shot_quality.parquet": player_shots,
        "annual_shooting_threat.parquet": annual,
        "five_year_shooting_threat.parquet": five_year,
        "lineup_spacing.parquet": spacing,
    }
    for name, frame in files.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "shots": len(enriched),
            "shot_event_context_coverage": float(enriched["qualifier_text"].notna().mean()),
            "possession_timing_coverage": float(enriched["seconds_since_possession_start"].notna().mean()),
            "annual_rows": len(annual),
            "five_year_rows": len(five_year),
            "spacing_rows": len(spacing),
            "season_2027_loaded": False,
        },
        "files": {name: {"rows": len(frame), "sha256": sha256_file(output / name)} for name, frame in files.items()},
        "forbidden_interpretation": (
            "This is not LASER, causal spacing, defender credit, or player impact. "
            "The 2026 result is reused diagnostic evidence."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_suite(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
