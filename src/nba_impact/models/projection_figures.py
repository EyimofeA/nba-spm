"""DARKO-style projection figures from a dispersion-calibrated backbone run.

Reads the calibration run's intervals.parquet plus the pinned state-space
trajectories and writes fan-chart and coverage figures under
<calibration_run>/figures/. Purely a rendering layer over pinned artifacts;
no model logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_SEASON_END = 2026
PLAYER_GRID_SIZE = 6



def build_projection_figures(
    calibration_run_dir: str | Path,
    state_space_trajectories_path: str | Path,
    players: list[str] | None = None,
) -> list[Path]:
    """Render fan charts and the coverage diagnostic; return written paths."""
    run_dir = Path(calibration_run_dir)
    intervals = pd.read_parquet(run_dir / "intervals.parquet")
    trajectories = pd.read_parquet(state_space_trajectories_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name_by_player = dict(zip(trajectories["PLAYER_ID"].astype(int), trajectories["PLAYER_NAME"]))
    required = {"forecast_net", "net_lo100", "net_hi100", "net_lo196", "net_hi196"}

    if not required.issubset(intervals.columns):
        raise ValueError(f"Intervals parquet lacks net forecast/band columns: {sorted(required - set(intervals.columns))}")

    if players:
        wanted = {name.strip().lower() for name in players}
        pool = [
            (pid, name_by_player.get(pid, "?"))
            for pid, name in _all_names(intervals, name_by_player)
            if name.lower() in wanted
        ]
        if len(pool) != len(wanted):
            missing = wanted - {name for _, name in pool}
            raise ValueError(f"Requested players without forecast rows: {sorted(missing)}")
        selected = pool
    else:
        selected = _select_top_named(intervals, name_by_player)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    grid_rows = int(np.ceil(len(selected) / 3))
    fig, axes = plt.subplots(grid_rows, 3, figsize=(15, 4.2 * grid_rows), squeeze=False)
    for ax, (player_id, name) in zip(axes.ravel(), selected):
        _draw_fan(ax, intervals, trajectories, player_id, name)
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.suptitle("Next-season impact projections with calibrated 68%/95% bands", y=1.0, fontsize=13)
    fig.tight_layout()
    path = figures_dir / "projection_fan_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    seasons = sorted(intervals["Target_Season"].unique())
    coverage = [
        float(
            (
                (
                    (group["forecast_offense"] - group["panel_target_offense"]).abs()
                    <= group["sd_offense"]
                ).mean()
            )
        )
        for _, group in intervals.groupby("Target_Season")
    ]
    ax.plot(seasons, coverage, marker="o", label="offense empirical 68% coverage")
    defense = [
        float(
            (
                (
                    (group["forecast_defense"] - group["panel_target_defense"]).abs()
                    <= group["sd_defense"]
                ).mean()
            )
        )
        for _, group in intervals.groupby("Target_Season")
    ]
    ax.plot(seasons, defense, marker="s", label="defense empirical 68% coverage")
    ax.axhline(0.6827, color="gray", linestyle="--", linewidth=1, label="nominal 68%")
    ax.axvspan(2024.5, 2026.5, alpha=0.08, color="green", label="confirmation seasons")
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel("forecast season")
    ax.set_ylabel("coverage")
    ax.legend(fontsize=9)
    ax.set_title("Interval coverage by forecast season (primary population)")
    fig.tight_layout()
    path = figures_dir / "coverage_by_season.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def _all_names(intervals: pd.DataFrame, name_by_player: dict[int, str]) -> list[tuple[int, str]]:
    return [(int(pid), name_by_player.get(int(pid), "?")) for pid in intervals["PLAYER_ID"].unique()]


def _select_top_named(
    intervals: pd.DataFrame, name_by_player: dict[int, str]
) -> list[tuple[int, str]]:
    latest_season = int(intervals["Target_Season"].max())
    latest = intervals.loc[intervals["Target_Season"].eq(latest_season)]
    ordered = latest.iloc[latest["forecast_net"].abs().to_numpy().argsort()[::-1]]
    return [
        (int(row.PLAYER_ID), name_by_player.get(int(row.PLAYER_ID), "?"))
        for row in ordered.head(PLAYER_GRID_SIZE).itertuples()
    ]


def _draw_fan(
    ax: plt.Axes,
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    player_id: int,
    name: str,
) -> None:
    personal = intervals.loc[intervals["PLAYER_ID"].eq(player_id)].sort_values("Target_Season")
    history = trajectories.loc[trajectories["PLAYER_ID"].eq(player_id)].sort_values("Season")
    history = history.loc[history["Season"] <= HISTORY_SEASON_END]
    ax.plot(history["Season"], history["filtered_net"], color="steelblue", linewidth=1.6, label="state-space history")
    seasons = personal["Target_Season"].to_numpy()
    point = personal["forecast_net"].to_numpy()
    lo68, hi68 = personal["net_lo100"].to_numpy(), personal["net_hi100"].to_numpy()
    lo95, hi95 = personal["net_lo196"].to_numpy(), personal["net_hi196"].to_numpy()
    ax.fill_between(seasons, lo95, hi95, alpha=0.18, label="95% band")
    ax.fill_between(seasons, lo68, hi68, alpha=0.32, label="68% band")
    ax.plot(seasons, point, marker="o", markersize=4, linewidth=0, color="darkorange", label="backbone forecast")
    ax.scatter(personal["Target_Season"], personal["panel_target_net"], marker="x", s=28, color="black", label="actual")
    ax.axhline(0.0, color="lightgray", linewidth=0.8)
    ax.set_title(f"{name} ({player_id})", fontsize=10)
    ax.grid(alpha=0.25)
