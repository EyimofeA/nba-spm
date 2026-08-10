"""Stable read-only query contract over published rating artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANNUAL_METRICS = (
    "aio_net",
    "aio_offense",
    "aio_defense",
    "normal_rapm_net",
    "normal_rapm_offense",
    "normal_rapm_defense",
    "spm_center_net",
    "spm_center_offense",
    "spm_center_defense",
    "spm_raw_net",
    "spm_raw_offense",
    "spm_raw_defense",
    "rapm_update_net",
    "rapm_update_offense",
    "rapm_update_defense",
)
PEAK_COMPONENTS = ("net", "offense", "defense")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _read_run(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rating run manifest: {path}")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class RatingsApiConfig:
    contract_version: str
    annual_run_id: str
    rolling_run_id: str
    default_limit: int = 25
    maximum_limit: int = 100

    @classmethod
    def from_json(cls, path: str | Path) -> "RatingsApiConfig":
        values = json.loads(Path(path).read_text())
        return cls(**values)


class RatingsStore:
    """Load one explicitly pinned annual run and one rolling run into memory."""

    def __init__(
        self,
        config: RatingsApiConfig,
        artifact_root: str | Path,
    ) -> None:
        self.config = config
        artifact_root = Path(artifact_root)
        self.annual_dir = artifact_root / "annual_aio_ratings" / config.annual_run_id
        self.rolling_dir = artifact_root / "rolling_rapm_peaks" / config.rolling_run_id
        self.annual_manifest = _read_run(self.annual_dir / "run.json")
        self.rolling_manifest = _read_run(self.rolling_dir / "run.json")
        self.annual = pd.read_parquet(self.annual_dir / "ratings.parquet")
        self.rolling = pd.read_parquet(self.rolling_dir / "rolling_ratings.parquet")
        self.peaks = pd.read_parquet(self.rolling_dir / "player_peaks.parquet")
        self._validate()

    def _validate(self) -> None:
        annual_required = {
            "PLAYER_ID",
            "PLAYER_NAME",
            "Season",
            "Poss_Off",
            "Poss_Def",
            *ANNUAL_METRICS,
        }
        rolling_required = {
            "PLAYER_ID",
            "PLAYER_NAME",
            "window_start",
            "window_end",
            "window_seasons",
            "Poss_Off",
            "Poss_Def",
            *PEAK_COMPONENTS,
        }
        peak_required = rolling_required | {"peak_component", "peak_value", "all_time_rank"}
        for label, frame, required in (
            ("annual", self.annual, annual_required),
            ("rolling", self.rolling, rolling_required),
            ("peaks", self.peaks, peak_required),
        ):
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{label} rating artifact lacks columns: {missing}")
        if self.annual.duplicated(["PLAYER_ID", "Season"]).any():
            raise ValueError("Annual rating keys are not unique.")
        if self.rolling.duplicated(["PLAYER_ID", "window_seasons", "window_end"]).any():
            raise ValueError("Rolling rating keys are not unique.")
        if self.peaks.duplicated(["PLAYER_ID", "window_seasons", "peak_component"]).any():
            raise ValueError("Peak rating keys are not unique.")

    def _limit(self, limit: int | None) -> int:
        value = self.config.default_limit if limit is None else int(limit)
        if value < 1 or value > self.config.maximum_limit:
            raise ValueError(f"limit must be between 1 and {self.config.maximum_limit}")
        return value

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_version": self.config.contract_version,
            "annual_run_id": self.config.annual_run_id,
            "rolling_run_id": self.config.rolling_run_id,
            "annual_status": self.annual_manifest["status"],
            "rolling_status": self.rolling_manifest["status"],
            "annual_estimand": self.annual_manifest["estimand"],
            "rolling_estimand": self.rolling_manifest["estimand"],
            "annual_seasons": sorted(int(value) for value in self.annual["Season"].unique()),
            "rolling_windows": sorted(
                int(value) for value in self.rolling["window_seasons"].unique()
            ),
            "annual_metrics": list(ANNUAL_METRICS),
            "peak_components": list(PEAK_COMPONENTS),
            "caveats": sorted(
                set(self.annual_manifest.get("caveats", []))
                | set(self.rolling_manifest.get("caveats", []))
            ),
        }

    def annual_leaderboard(
        self,
        season: int,
        metric: str = "aio_net",
        *,
        limit: int | None = None,
        offset: int = 0,
        minimum_possessions: int = 0,
    ) -> dict[str, Any]:
        if metric not in ANNUAL_METRICS:
            raise ValueError(f"unsupported annual metric: {metric}")
        season = int(season)
        if season not in set(self.annual["Season"]):
            raise ValueError(f"unsupported annual season: {season}")
        if offset < 0 or minimum_possessions < 0:
            raise ValueError("offset and minimum_possessions must be nonnegative")
        size = self._limit(limit)
        frame = self.annual.loc[
            (self.annual["Season"] == season)
            & (self.annual[["Poss_Off", "Poss_Def"]].min(axis=1) >= minimum_possessions)
        ].copy()
        frame = frame.sort_values(
            [metric, "Poss_Off", "PLAYER_ID"],
            ascending=[False, False, True],
            kind="stable",
        )
        frame["rank"] = np.arange(1, len(frame) + 1)
        columns = [
            "rank",
            "PLAYER_ID",
            "PLAYER_NAME",
            "Season",
            metric,
            "Poss_Off",
            "Poss_Def",
        ]
        return {
            "season": season,
            "metric": metric,
            "total": len(frame),
            "offset": int(offset),
            "limit": size,
            "results": _records(frame.iloc[offset : offset + size][columns]),
        }

    def peak_leaderboard(
        self,
        window_seasons: int,
        component: str = "net",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        if component not in PEAK_COMPONENTS:
            raise ValueError(f"unsupported peak component: {component}")
        window_seasons = int(window_seasons)
        if window_seasons not in set(self.peaks["window_seasons"]):
            raise ValueError(f"unsupported peak window: {window_seasons}")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        size = self._limit(limit)
        frame = self.peaks.loc[
            (self.peaks["window_seasons"] == window_seasons)
            & (self.peaks["peak_component"] == component)
        ].sort_values("all_time_rank", kind="stable")
        columns = [
            "all_time_rank",
            "PLAYER_ID",
            "PLAYER_NAME",
            "window_start",
            "window_end",
            "window_seasons",
            "offense",
            "defense",
            "net",
            "Poss_Off",
            "Poss_Def",
            "peak_component",
            "peak_value",
        ]
        return {
            "window_seasons": window_seasons,
            "component": component,
            "total": len(frame),
            "offset": int(offset),
            "limit": size,
            "results": _records(frame.iloc[offset : offset + size][columns]),
        }

    def search_players(self, query: str, *, limit: int | None = None) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        size = self._limit(limit)
        names = pd.concat(
            [
                self.annual[["PLAYER_ID", "PLAYER_NAME"]],
                self.rolling[["PLAYER_ID", "PLAYER_NAME"]],
            ],
            ignore_index=True,
        ).drop_duplicates("PLAYER_ID")
        text = names["PLAYER_NAME"].fillna("").str.casefold()
        needle = query.casefold()
        names = names.loc[text.str.contains(needle, regex=False)].copy()
        names["match_rank"] = np.select(
            [text.loc[names.index] == needle, text.loc[names.index].str.startswith(needle)],
            [0, 1],
            default=2,
        )
        names = names.sort_values(
            ["match_rank", "PLAYER_NAME", "PLAYER_ID"], kind="stable"
        )
        return {
            "query": query,
            "total": len(names),
            "limit": size,
            "results": _records(names.head(size)[["PLAYER_ID", "PLAYER_NAME"]]),
        }

    def player(self, player_id: int) -> dict[str, Any] | None:
        player_id = int(player_id)
        annual = self.annual.loc[self.annual["PLAYER_ID"] == player_id].sort_values("Season")
        rolling = self.rolling.loc[self.rolling["PLAYER_ID"] == player_id].sort_values(
            ["window_seasons", "window_end"]
        )
        peaks = self.peaks.loc[self.peaks["PLAYER_ID"] == player_id].sort_values(
            ["window_seasons", "peak_component"]
        )
        if annual.empty and rolling.empty:
            return None
        available = annual if not annual.empty else rolling
        name = available["PLAYER_NAME"].dropna().iloc[-1]
        annual_columns = [
            "Season",
            "Poss_Off",
            "Poss_Def",
            *ANNUAL_METRICS,
        ]
        rolling_columns = [
            "window_start",
            "window_end",
            "window_seasons",
            "Poss_Off",
            "Poss_Def",
            "offense",
            "defense",
            "net",
        ]
        return {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": name,
            "annual": _records(annual[annual_columns]),
            "rolling": _records(rolling[rolling_columns]),
            "peaks": _records(peaks.drop(columns=["PLAYER_ID", "PLAYER_NAME"])),
        }
