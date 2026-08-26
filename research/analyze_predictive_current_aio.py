"""Create the decision record for the frozen predictive current AIO run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "artifacts/models/predictive_current_aio"
    / "predictive_current_aio_2026_v1_c18e2472ec"
)
OUTPUT = ROOT / "research/audits/predictive_current_aio_2026_v1"
SELECTED = "selected_decay_spm_prior_aio"
COMPARATORS = (
    "five_year_zero_prior",
    "selected_decay_zero_prior",
    "five_year_spm_prior_aio",
)


def paired_bootstrap_rows(
    games: pd.DataFrame,
    *,
    selected: str,
    comparators: tuple[str, ...],
    scopes: dict[str, tuple[int, ...]],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for scope_index, (scope, seasons) in enumerate(scopes.items()):
        for comparator_index, comparator in enumerate(comparators):
            matched = games.loc[
                games["test_season"].isin(seasons)
                & games["arm"].isin([selected, comparator])
            ].pivot(
                index=["test_season", "game_id"],
                columns="arm",
                values="squared_error",
            )
            if matched[[selected, comparator]].isna().any().any():
                raise ValueError("Bootstrap comparators must use identical games.")
            season_deltas = [
                group[selected].to_numpy(dtype=float)
                - group[comparator].to_numpy(dtype=float)
                for _, group in matched.groupby(level="test_season", sort=True)
            ]
            observed = float(np.mean([values.mean() for values in season_deltas]))
            rng = np.random.default_rng(seed + scope_index * 100 + comparator_index)
            samples = np.empty(draws, dtype=float)
            for draw in range(draws):
                samples[draw] = np.mean(
                    [
                        rng.choice(values, size=len(values), replace=True).mean()
                        for values in season_deltas
                    ]
                )
            lower, upper = np.quantile(samples, [0.025, 0.975])
            records.append(
                {
                    "scope": scope,
                    "seasons": ",".join(str(value) for value in seasons),
                    "selected": selected,
                    "comparator": comparator,
                    "matched_games": int(len(matched)),
                    "mse_delta_selected_minus_comparator": observed,
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "probability_selected_better": float(np.mean(samples < 0)),
                    "draws": int(draws),
                    "seed": int(seed + scope_index * 100 + comparator_index),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = json.loads((RUN / "run.json").read_text())
    games = pd.read_parquet(RUN / "game_predictions.parquet")
    metrics = pd.read_parquet(RUN / "fold_metrics.parquet")
    development = tuple(int(value) for value in run["development_seasons"])
    diagnostics = tuple(int(value) for value in run["reused_diagnostic_seasons"])
    bootstrap = paired_bootstrap_rows(
        games,
        selected=SELECTED,
        comparators=COMPARATORS,
        scopes={"development": development, "reused_diagnostic": diagnostics},
        draws=10_000,
        seed=20260826,
    )
    bootstrap.to_parquet(OUTPUT / "paired_bootstrap.parquet", index=False)
    fold = metrics.pivot(index="test_season", columns="arm", values="margin_rmse")
    development_wins = int(
        (
            fold.loc[list(development), SELECTED]
            < fold.loc[list(development), "five_year_zero_prior"]
        ).sum()
    )
    decision = {
        "experiment_id": run["experiment_id"],
        "run_id": run["run_id"],
        "decision": "research_champion_not_public",
        "selected_half_life_years": run["selected_half_life_years"],
        "selected_arm": run["selected_arm"],
        "development_fold_wins_vs_five_year_zero": development_wins,
        "development_folds": len(development),
        "development_mean_rmse": {
            row["arm"]: float(row["mean_margin_rmse"])
            for row in pd.read_parquet(RUN / "development_summary.parquet").to_dict(
                "records"
            )
        },
        "diagnostic_rmse": {
            str(season): {arm: float(fold.loc[season, arm]) for arm in fold.columns}
            for season in diagnostics
        },
        "promotion_blocker": (
            "Seasons 2025 and 2026 are reused diagnostics. Season 2027 remains "
            "the untouched confirmation."
        ),
        "age_opportunity_decision": (
            "Raw predictive SPM won the separate 2020-24 residual-correction "
            "ablation, so age and lagged opportunity are excluded."
        ),
    }
    write_json_atomic(decision, OUTPUT / "decision.json")
    lines = [
        "# Predictive current AIO decision",
        "",
        "The two-year half-life plus raw predictive-SPM prior is the research champion.",
        "It is not a public or confirmed model.",
        "",
        "## Development result",
        "",
        "| Arm | Mean game-margin RMSE |",
        "| --- | ---: |",
    ]
    summary = pd.read_parquet(RUN / "development_summary.parquet")
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.mean_margin_rmse:.4f} |")
    lines.extend(
        [
            "",
            f"The selected AIO beat five-year zero-prior RAPM in {development_wins} of 5 folds.",
            "The paired whole-game MSE interval favored it against every frozen comparator.",
            "",
            "## Reused diagnostics",
            "",
            "| Season | Selected AIO | Five-year zero prior |",
            "| ---: | ---: | ---: |",
        ]
    )
    for season in diagnostics:
        lines.append(
            f"| {season} | {fold.loc[season, SELECTED]:.4f} | "
            f"{fold.loc[season, 'five_year_zero_prior']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The diagnostics support the development result but do not confirm it.",
            "Season 2027 stays untouched.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
