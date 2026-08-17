"""Stable read-only query contract over published rating artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_impact.research.control_plane import load_pinned_contracts


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
CURRENT_METRICS = {
    "net": "net_per_100",
    "offense": "offense_per_100",
    "defense": "defense_per_100",
}
UNCERTAINTY_METRICS = ("net", "offense", "defense")
MATCHUP_FACTOR_METRICS = (
    "matchup_fga_suppressed_vs_scorer_p100_eb",
    "matchup_shotmaking_points_saved_vs_scorer_p100_eb",
    "matchup_three_pa_suppressed_vs_scorer_p100_eb",
    "matchup_turnovers_forced_vs_scorer_p100_eb",
    "matchup_assists_suppressed_vs_scorer_p100_eb",
    "matchup_shooting_fouls_prevented_vs_scorer_p100_eb",
)
ROLE_LABELS = {
    "offense": {
        "off_role_0": "Primary creator",
        "off_role_1": "Secondary handler",
        "off_role_2": "Movement shooter",
        "off_role_3": "Versatile big",
        "off_role_4": "Post big",
        "off_role_5": "Rim finisher",
    },
    "defense": {
        "def_role_0": "Wing assignment",
        "def_role_1": "Creator assignment",
        "def_role_2": "Rebound / contest",
        "def_role_3": "Versatile forward",
        "def_role_4": "Interior / rim",
    },
}


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
    current_rapm_run_id: str
    default_limit: int = 25
    maximum_limit: int = 100
    matchup_defense_run_id: str | None = None
    lineage_contract_path: str | None = None
    current_uncertainty_run_id: str | None = None
    normal_rapm_uncertainty_run_ids: dict[str, str] | None = None
    side_roles_run_id: str | None = None

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
        self.current_dir = artifact_root / "rapm" / config.current_rapm_run_id
        self.annual_manifest = _read_run(self.annual_dir / "run.json")
        self.rolling_manifest = _read_run(self.rolling_dir / "run.json")
        self.current_manifest = _read_run(self.current_dir / "run.json")
        self.annual = pd.read_parquet(self.annual_dir / "ratings.parquet")
        self.rolling = pd.read_parquet(self.rolling_dir / "rolling_ratings.parquet")
        self.peaks = pd.read_parquet(self.rolling_dir / "player_peaks.parquet")
        self.current = pd.read_parquet(self.current_dir / "ratings.parquet")
        # Kept for wire compatibility only.  A single uncertainty run must not
        # be joined to the 2024--26 current-rating artifact unless it has the
        # identical season scope.  Scoped uncertainty runs are loaded below.
        self.current_uncertainty: pd.DataFrame | None = None
        if config.current_uncertainty_run_id is not None:
            uncertainty_path = (
                artifact_root
                / "rapm_uncertainty"
                / config.current_uncertainty_run_id
                / "ratings_uncertainty.parquet"
            )
            self.current_uncertainty = pd.read_parquet(uncertainty_path)
        self.normal_rapm_uncertainty: dict[str, pd.DataFrame] = {}
        self.normal_rapm_uncertainty_manifests: dict[str, dict[str, Any]] = {}
        for scope, run_id in (config.normal_rapm_uncertainty_run_ids or {}).items():
            run_dir = artifact_root / "rapm_uncertainty" / run_id
            self.normal_rapm_uncertainty[scope] = pd.read_parquet(
                run_dir / "ratings_uncertainty.parquet"
            )
            self.normal_rapm_uncertainty_manifests[scope] = _read_run(
                run_dir / "run.json"
            )
        self.matchup_manifest: dict[str, Any] | None = None
        self.matchup: pd.DataFrame | None = None
        if config.matchup_defense_run_id is not None:
            local_matchup = (
                artifact_root / "matchup_defense" / config.matchup_defense_run_id
            )
            sibling_matchup = (
                artifact_root.parent
                / "features"
                / "matchup_defense"
                / config.matchup_defense_run_id
            )
            self.matchup_dir = (
                local_matchup if local_matchup.exists() else sibling_matchup
            )
            self.matchup_manifest = _read_run(self.matchup_dir / "run.json")
            self.matchup = pd.read_parquet(self.matchup_dir / "features.parquet")
        self.side_roles_manifest: dict[str, Any] | None = None
        self.offense_roles: pd.DataFrame | None = None
        self.defense_roles: pd.DataFrame | None = None
        if config.side_roles_run_id is not None:
            self.side_roles_dir = (
                artifact_root.parent
                / "features"
                / "side_roles"
                / config.side_roles_run_id
            )
            self.side_roles_manifest = _read_run(self.side_roles_dir / "run.json")
            self.offense_roles = pd.read_parquet(
                self.side_roles_dir / "offense_assignments.parquet"
            )
            self.defense_roles = pd.read_parquet(
                self.side_roles_dir / "defense_assignments.parquet"
            )
        self._validate()

    def _lineage_contract(self) -> dict[str, Any]:
        configured = self.config.lineage_contract_path
        project_root = Path(__file__).resolve().parents[3]
        path = (
            (project_root / configured)
            if configured is not None and not Path(configured).is_absolute()
            else Path(configured)
            if configured is not None
            else project_root / "research" / "pinned_artifact_contracts.json"
        )
        return load_pinned_contracts(path)

    @staticmethod
    def _row_set_hash(frame: pd.DataFrame, keys: list[str]) -> str:
        canonical = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        payload = pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()
        import hashlib

        return hashlib.sha256(payload).hexdigest()

    def _v2_lineage(self, api_field: str, frame: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
        contract = self._lineage_contract()
        entry = next(
            item for item in contract["artifacts"] if item["api_field"] == api_field
        )
        return {
            "estimand_id": entry["estimand_id"],
            "evidence_status": entry["evidence_status"],
            "uncertainty_status": entry["uncertainty_status"],
            "season_scope": entry["season_scope"],
            "season_completeness": entry["season_completeness"],
            "model_config_sha256": entry["config_sha256"],
            "code_sha256": entry["code_sha256"],
            "data_hashes_status": entry["data_hashes_status"],
            "row_set_sha256": self._row_set_hash(frame, keys),
            "forbidden_interpretation": entry["forbidden_interpretation"],
        }

    @staticmethod
    def _v2_uncertainty(record: dict[str, Any]) -> dict[str, Any]:
        status = record.get("uncertainty_status", "not_estimated")
        method = record.get("uncertainty_method")
        components: dict[str, Any] = {}
        for component in ("offense", "defense", "net"):
            estimate = record.get(f"{component}_estimate")
            if estimate is None:
                continue
            components[component] = {
                "estimate": estimate,
                "standard_error": record.get(f"{component}_bootstrap_se"),
                "analytic_standard_error": record.get(f"{component}_analytic_se"),
                "interval_80": {
                    "low": record.get(f"{component}_ci80_low"),
                    "high": record.get(f"{component}_ci80_high"),
                },
                "interval_95": {
                    "low": record.get(f"{component}_ci95_low"),
                    "high": record.get(f"{component}_ci95_high"),
                },
                "probability_above_zero": record.get(
                    f"{component}_probability_above_zero"
                ),
                "draw_coverage": record.get(f"{component}_draw_coverage"),
            }
        return {"status": status, "method": method, "components": components}

    def v2_metadata(self) -> dict[str, Any]:
        uncertainty_artifacts = {
            scope: self._v2_lineage(
                f"normal_rapm_uncertainty_{scope}_run_id",
                frame,
                ["player_id"],
            )
            for scope, frame in self.normal_rapm_uncertainty.items()
        }
        return {
            "contract_version": "ratings_api_v2",
            "artifacts": {
                "annual": self._v2_lineage(
                    "annual_run_id", self.annual, ["PLAYER_ID", "Season"]
                ),
                "rolling_peaks": self._v2_lineage(
                    "rolling_run_id",
                    self.peaks,
                    ["PLAYER_ID", "window_seasons", "peak_component"],
                ),
                "current_normal_rapm": self._v2_lineage(
                    "current_rapm_run_id", self.current, ["player_id"]
                ),
                "matchup_defense": (
                    self._v2_lineage(
                        "matchup_defense_run_id", self.matchup, ["PLAYER_ID", "Season"]
                    )
                    if self.matchup is not None
                    else None
                ),
                "normal_rapm_uncertainty": uncertainty_artifacts,
            },
        }

    def v2_wrap(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        field = {
            "annual": "annual_run_id",
            "peaks": "rolling_run_id",
            "current": "current_rapm_run_id",
            "matchup-defense": "matchup_defense_run_id",
        }.get(endpoint)
        if field is None:
            return {"contract_version": "ratings_api_v2", "data": payload}
        frame, keys = {
            "annual_run_id": (self.annual, ["PLAYER_ID", "Season"]),
            "rolling_run_id": (
                self.peaks,
                ["PLAYER_ID", "window_seasons", "peak_component"],
            ),
            "current_rapm_run_id": (self.current, ["player_id"]),
            "matchup_defense_run_id": (self.matchup, ["PLAYER_ID", "Season"]),
        }[field]
        output = {
            "contract_version": "ratings_api_v2",
            "lineage": self._v2_lineage(field, frame, keys),
            "data": payload,
        }
        if endpoint == "current":
            lookup = (
                self.current_uncertainty.set_index("player_id").to_dict(orient="index")
                if self.current_uncertainty is not None
                else {}
            )
            for record in output["data"].get("results", []):
                record["uncertainty"] = self._v2_uncertainty(
                    lookup.get(record["player_id"], {"uncertainty_status": output["lineage"]["uncertainty_status"]})
                )
        return output

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
        current_required = {
            "player_id",
            "player_name",
            "offense_per_100",
            "defense_per_100",
            "net_per_100",
            "off_possessions",
            "def_possessions",
        }
        for label, frame, required in (
            ("annual", self.annual, annual_required),
            ("rolling", self.rolling, rolling_required),
            ("peaks", self.peaks, peak_required),
            ("current", self.current, current_required),
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
        if self.current["player_id"].duplicated().any():
            raise ValueError("Current RAPM player IDs are not unique.")
        if self.matchup is not None:
            matchup_required = {
                "PLAYER_ID",
                "Season",
                "matchup_possessions",
                *MATCHUP_FACTOR_METRICS,
            }
            if missing := sorted(matchup_required - set(self.matchup.columns)):
                raise ValueError(f"Matchup feature artifact lacks columns: {missing}")
            if self.matchup.duplicated(["PLAYER_ID", "Season"]).any():
                raise ValueError("Matchup feature keys are not unique.")
        for side, frame, prefix in (
            ("offense", self.offense_roles, "off"),
            ("defense", self.defense_roles, "def"),
        ):
            if frame is None:
                continue
            role_required = {
                "PLAYER_ID",
                "Season",
                f"{prefix}_role_cluster",
                f"{prefix}_role_confidence",
                *{
                    f"{prefix}_role_affinity_{index}"
                    for index in range(len(ROLE_LABELS[side]))
                },
            }
            if missing := sorted(role_required - set(frame.columns)):
                raise ValueError(f"{side} role artifact lacks columns: {missing}")
            if frame.duplicated(["PLAYER_ID", "Season"]).any():
                raise ValueError(f"{side} role keys are not unique.")
        uncertainty_required = {
            "player_id",
            "player_name",
            "off_possessions",
            "def_possessions",
            "uncertainty_method",
            "uncertainty_status",
        }
        for component in UNCERTAINTY_METRICS:
            uncertainty_required |= {
                f"{component}_estimate",
                f"{component}_bootstrap_se",
                f"{component}_ci80_low",
                f"{component}_ci80_high",
                f"{component}_ci95_low",
                f"{component}_ci95_high",
                f"{component}_probability_above_zero",
                f"{component}_draw_coverage",
            }
        for scope, frame in self.normal_rapm_uncertainty.items():
            if missing := sorted(uncertainty_required - set(frame.columns)):
                raise ValueError(
                    f"Normal RAPM uncertainty artifact {scope!r} lacks columns: {missing}"
                )
            if frame["player_id"].duplicated().any():
                raise ValueError(
                    f"Normal RAPM uncertainty player IDs are not unique for {scope!r}."
                )

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
            "current_rapm_run_id": self.config.current_rapm_run_id,
            "matchup_defense_run_id": self.config.matchup_defense_run_id,
            "side_roles_run_id": self.config.side_roles_run_id,
            "annual_status": self.annual_manifest["status"],
            "rolling_status": self.rolling_manifest["status"],
            "current_rapm_status": self.current_manifest["status"],
            "annual_estimand": self.annual_manifest["estimand"],
            "rolling_estimand": self.rolling_manifest["estimand"],
            "current_rapm_estimand": self.current_manifest["estimand"],
            "annual_seasons": sorted(int(value) for value in self.annual["Season"].unique()),
            "rolling_windows": sorted(
                int(value) for value in self.rolling["window_seasons"].unique()
            ),
            "annual_metrics": list(ANNUAL_METRICS),
            "peak_components": list(PEAK_COMPONENTS),
            "current_rapm_metrics": list(CURRENT_METRICS),
            "matchup_factor_metrics": list(MATCHUP_FACTOR_METRICS),
            "matchup_factor_status": (
                self.matchup_manifest["status"]
                if self.matchup_manifest is not None
                else None
            ),
            "side_roles_status": (
                self.side_roles_manifest["status"]
                if self.side_roles_manifest is not None
                else None
            ),
            "caveats": sorted(
                set(self.annual_manifest.get("caveats", []))
                | set(self.rolling_manifest.get("caveats", []))
                | set(self.current_manifest.get("caveats", []))
                | (
                    {self.matchup_manifest["caveat"]}
                    if self.matchup_manifest is not None
                    and self.matchup_manifest.get("caveat")
                    else set()
                )
            ),
        }

    def matchup_defense_leaderboard(
        self,
        season: int,
        metric: str = "matchup_shotmaking_points_saved_vs_scorer_p100_eb",
        *,
        limit: int | None = None,
        offset: int = 0,
        minimum_matchup_possessions: int = 0,
    ) -> dict[str, Any]:
        if self.matchup is None or self.matchup_manifest is None:
            raise ValueError("matchup-defense research data is not configured")
        if metric not in MATCHUP_FACTOR_METRICS:
            raise ValueError(f"unsupported matchup-defense metric: {metric}")
        season = int(season)
        if season not in set(self.matchup["Season"]):
            raise ValueError(f"unsupported matchup-defense season: {season}")
        if offset < 0 or minimum_matchup_possessions < 0:
            raise ValueError(
                "offset and minimum_matchup_possessions must be nonnegative"
            )
        size = self._limit(limit)
        frame = self.matchup.loc[
            (self.matchup["Season"] == season)
            & (self.matchup["matchup_possessions"] >= minimum_matchup_possessions)
        ].copy()
        names = pd.concat(
            [
                self.annual[["PLAYER_ID", "PLAYER_NAME"]],
                self.rolling[["PLAYER_ID", "PLAYER_NAME"]],
                self.current[["player_id", "player_name"]].rename(
                    columns={"player_id": "PLAYER_ID", "player_name": "PLAYER_NAME"}
                ),
            ],
            ignore_index=True,
        ).dropna(subset=["PLAYER_NAME"]).drop_duplicates("PLAYER_ID", keep="last")
        frame = frame.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
        frame = frame.sort_values(
            [metric, "matchup_possessions", "PLAYER_ID"],
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
            "matchup_possessions",
        ]
        return {
            "status": "research_only",
            "run_id": self.config.matchup_defense_run_id,
            "season": season,
            "metric": metric,
            "total": len(frame),
            "offset": int(offset),
            "limit": size,
            "caveat": self.matchup_manifest.get("caveat"),
            "results": _records(frame.iloc[offset : offset + size][columns]),
        }

    def current_leaderboard(
        self,
        metric: str = "net",
        *,
        limit: int | None = None,
        offset: int = 0,
        minimum_possessions: int = 0,
    ) -> dict[str, Any]:
        if metric not in CURRENT_METRICS:
            raise ValueError(f"unsupported current RAPM metric: {metric}")
        if offset < 0 or minimum_possessions < 0:
            raise ValueError("offset and minimum_possessions must be nonnegative")
        size = self._limit(limit)
        column = CURRENT_METRICS[metric]
        frame = self.current.loc[
            self.current[["off_possessions", "def_possessions"]].min(axis=1)
            >= minimum_possessions
        ].copy()
        frame = frame.sort_values(
            [column, "off_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        frame["rank"] = np.arange(1, len(frame) + 1)
        columns = [
            "rank",
            "player_id",
            "player_name",
            "offense_per_100",
            "defense_per_100",
            "net_per_100",
            "off_possessions",
            "def_possessions",
        ]
        return {
            "metric": metric,
            "seasons": self.current_manifest["config"]["seasons"],
            "total": len(frame),
            "offset": int(offset),
            "limit": size,
            "results": _records(frame.iloc[offset : offset + size][columns]),
        }

    def normal_rapm_uncertainty_leaderboard(
        self,
        scope: str,
        metric: str = "net",
        *,
        limit: int | None = None,
        offset: int = 0,
        minimum_possessions: int = 0,
    ) -> dict[str, Any]:
        """Return one scope-matched bootstrap uncertainty leaderboard.

        This intentionally does not reuse the current-rating endpoint: the
        resampled runs have their own frozen season scope and source lineage.
        """
        if scope not in self.normal_rapm_uncertainty:
            raise ValueError(f"unsupported normal RAPM uncertainty scope: {scope}")
        if metric not in UNCERTAINTY_METRICS:
            raise ValueError(f"unsupported normal RAPM uncertainty metric: {metric}")
        if offset < 0 or minimum_possessions < 0:
            raise ValueError("offset and minimum_possessions must be nonnegative")
        size = self._limit(limit)
        frame = self.normal_rapm_uncertainty[scope]
        manifest = self.normal_rapm_uncertainty_manifests[scope]
        component = metric
        frame = frame.loc[
            frame[["off_possessions", "def_possessions"]].min(axis=1)
            >= minimum_possessions
        ].copy()
        estimate = f"{component}_estimate"
        frame = frame.sort_values(
            [estimate, "off_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        frame["rank"] = np.arange(1, len(frame) + 1)
        columns = [
            "rank",
            "player_id",
            "player_name",
            "off_possessions",
            "def_possessions",
            "offense_estimate",
            "defense_estimate",
            "net_estimate",
            "uncertainty_method",
            "uncertainty_status",
        ]
        results = []
        for row in _records(frame.iloc[offset : offset + size]):
            record = {column: row[column] for column in columns}
            record["uncertainty"] = self._v2_uncertainty(row)
            results.append(record)
        return {
            "scope": scope,
            "run_id": self.config.normal_rapm_uncertainty_run_ids[scope],
            "status": manifest["status"],
            "estimand_id": manifest["estimand_id"],
            "seasons": manifest["config"]["seasons"],
            "metric": metric,
            "total": len(frame),
            "offset": int(offset),
            "limit": size,
            "caveats": manifest.get("caveats", []),
            "results": results,
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
                self.current[["player_id", "player_name"]].rename(
                    columns={"player_id": "PLAYER_ID", "player_name": "PLAYER_NAME"}
                ),
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
            current = self.current.loc[self.current["player_id"] == player_id]
            if current.empty:
                return None
        else:
            current = self.current.loc[self.current["player_id"] == player_id]
        name_values = pd.concat(
            [
                current["player_name"] if not current.empty else pd.Series(dtype=object),
                annual["PLAYER_NAME"] if not annual.empty else pd.Series(dtype=object),
                rolling["PLAYER_NAME"] if not rolling.empty else pd.Series(dtype=object),
            ],
            ignore_index=True,
        ).dropna()
        name = name_values.iloc[-1] if not name_values.empty else f"Player {player_id}"
        matchup = (
            self.matchup.loc[self.matchup["PLAYER_ID"] == player_id].sort_values(
                "Season"
            )
            if self.matchup is not None
            else pd.DataFrame()
        )
        uncertainty = {
            scope: frame.loc[frame["player_id"] == player_id]
            for scope, frame in self.normal_rapm_uncertainty.items()
        }
        role_rows: dict[int, dict[str, Any]] = {}
        for side, frame, prefix in (
            ("offense", self.offense_roles, "off"),
            ("defense", self.defense_roles, "def"),
        ):
            if frame is None:
                continue
            selected = frame.loc[frame["PLAYER_ID"] == player_id].sort_values("Season")
            for row in selected.to_dict(orient="records"):
                season = int(row["Season"])
                memberships = [
                    {
                        "role_id": role_id,
                        "label": label,
                        "affinity": float(row[f"{prefix}_role_affinity_{index}"]),
                    }
                    for index, (role_id, label) in enumerate(ROLE_LABELS[side].items())
                ]
                memberships.sort(key=lambda item: item["affinity"], reverse=True)
                primary_role_id = row[f"{prefix}_role_cluster"]
                role_rows.setdefault(season, {"Season": season})[side] = {
                    "primary_role_id": primary_role_id,
                    "primary_role": ROLE_LABELS[side][primary_role_id],
                    "confidence": float(row[f"{prefix}_role_confidence"]),
                    "memberships": memberships,
                }
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
            "current_normal_rapm": (
                _records(current.drop(columns=["player_id", "player_name"]))[0]
                if not current.empty
                else None
            ),
            "matchup_defense_factors": (
                _records(
                    matchup[
                        [
                            "Season",
                            "matchup_possessions",
                            *MATCHUP_FACTOR_METRICS,
                        ]
                    ]
                )
                if not matchup.empty
                else []
            ),
            "normal_rapm_uncertainty": {
                scope: (
                    {
                        "run_id": self.config.normal_rapm_uncertainty_run_ids[scope],
                        "status": self.normal_rapm_uncertainty_manifests[scope]["status"],
                        "seasons": self.normal_rapm_uncertainty_manifests[scope]["config"]["seasons"],
                        "rating": {
                            "off_possessions": int(frame.iloc[0]["off_possessions"]),
                            "def_possessions": int(frame.iloc[0]["def_possessions"]),
                            "uncertainty": self._v2_uncertainty(frame.iloc[0].to_dict()),
                        },
                    }
                    if not frame.empty
                    else None
                )
                for scope, frame in uncertainty.items()
            },
            "roles": [role_rows[season] for season in sorted(role_rows)],
        }
