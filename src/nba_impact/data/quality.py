"""High-signal quality gates for the legacy possession snapshot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from .contracts import (
    AWAY_PLAYER_COLUMNS,
    HOME_PLAYER_COLUMNS,
    PLAYER_COLUMNS,
    POSSESSION_COLUMNS,
    POSSESSION_KEY,
)
from .normalize import classify_game_type


Severity = Literal["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    count: int
    message: str


@dataclass
class QualityReport:
    path: str
    expected_season: int | None
    row_count: int
    column_count: int
    game_count: int = 0
    date_min: str | None = None
    date_max: str | None = None
    exact_duplicate_rows: int = 0
    duplicate_key_rows: int = 0
    game_type_counts: dict[str, int] = field(default_factory=dict)
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity in {"critical", "high"} for issue in self.issues)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _issue(report: QualityReport, code: str, severity: Severity, count: int, message: str) -> None:
    if count:
        report.issues.append(QualityIssue(code, severity, int(count), message))


def _lineup_overlap_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows where an ID appears in both five-player lineups.

    The column-wise comparisons avoid a Python loop over millions of
    possessions while keeping peak memory small.
    """
    overlap = pd.Series(False, index=frame.index)
    for away_column in AWAY_PLAYER_COLUMNS:
        for home_column in HOME_PLAYER_COLUMNS:
            overlap |= frame[away_column].eq(frame[home_column])
    return overlap


def audit_possession_frame(
    frame: pd.DataFrame,
    *,
    path: str = "<memory>",
    expected_season: int | None = None,
) -> QualityReport:
    report = QualityReport(
        path=path,
        expected_season=expected_season,
        row_count=int(len(frame)),
        column_count=int(len(frame.columns)),
    )
    if frame.empty:
        _issue(report, "empty_partition", "critical", 1, "Partition contains zero rows.")
        return report

    missing = [column for column in POSSESSION_COLUMNS if column not in frame.columns]
    if missing:
        _issue(
            report,
            "missing_required_columns",
            "critical",
            len(missing),
            f"Missing required columns: {', '.join(missing)}",
        )
        return report

    required = frame.loc[:, POSSESSION_COLUMNS]
    null_rows = int(required.isna().any(axis=1).sum())
    _issue(report, "required_null_rows", "critical", null_rows, "Rows contain null required fields.")

    report.exact_duplicate_rows = int(frame.duplicated().sum())
    _issue(
        report,
        "exact_duplicate_rows",
        "high",
        report.exact_duplicate_rows,
        "Exact duplicate possession rows found.",
    )
    report.duplicate_key_rows = int(frame.duplicated(list(POSSESSION_KEY), keep=False).sum())
    _issue(
        report,
        "duplicate_possession_keys",
        "critical",
        report.duplicate_key_rows,
        "Composite possession keys are not unique.",
    )

    report.game_count = int(frame["gameid"].nunique(dropna=True))
    game_types = (
        frame[["gameid"]]
        .drop_duplicates()
        .assign(game_type=lambda value: value["gameid"].map(classify_game_type))
    )
    report.game_type_counts = {
        str(key): int(value) for key, value in game_types["game_type"].value_counts().items()
    }
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    _issue(report, "invalid_dates", "high", invalid_dates, "Dates could not be parsed.")
    if parsed_dates.notna().any():
        report.date_min = str(parsed_dates.min().date())
        report.date_max = str(parsed_dates.max().date())

    if expected_season is not None:
        season_numeric = pd.to_numeric(frame["season"], errors="coerce")
        mismatched = int((season_numeric != expected_season).sum())
        _issue(
            report,
            "season_mismatch",
            "critical",
            mismatched,
            f"Rows do not belong to expected end season {expected_season}.",
        )

    home_values = set(frame["home_poss"].dropna().astype(int).unique().tolist())
    invalid_home_values = len(home_values - {0, 1})
    _issue(
        report,
        "invalid_home_possession_flag",
        "critical",
        invalid_home_values,
        f"home_poss contains values outside 0/1: {sorted(home_values)}",
    )

    points = pd.to_numeric(frame["pts"], errors="coerce")
    invalid_points = int((points.isna() | (points < 0) | (points > 20)).sum())
    _issue(report, "invalid_points", "critical", invalid_points, "Possession points fall outside [0, 20].")
    unusual_points = int(((points > 6) & (points <= 20)).sum())
    _issue(
        report,
        "unusual_points",
        "medium",
        unusual_points,
        "Possessions above six points require event-level reconciliation.",
    )

    invalid_player_ids = int(
        sum(pd.to_numeric(frame[column], errors="coerce").isna().sum() for column in PLAYER_COLUMNS)
    )
    _issue(report, "invalid_player_ids", "critical", invalid_player_ids, "Lineup player IDs are invalid.")

    away_duplicate = frame.loc[:, AWAY_PLAYER_COLUMNS].nunique(axis=1) != 5
    home_duplicate = frame.loc[:, HOME_PLAYER_COLUMNS].nunique(axis=1) != 5
    _issue(
        report,
        "duplicate_player_within_lineup",
        "critical",
        int((away_duplicate | home_duplicate).sum()),
        "A lineup contains fewer than five distinct players.",
    )

    overlap_count = int(_lineup_overlap_mask(frame).sum())
    _issue(
        report,
        "player_on_both_teams",
        "critical",
        overlap_count,
        "A player appears on both teams in the same possession.",
    )
    return report


def audit_possession_file(path: str | Path, expected_season: int | None = None) -> QualityReport:
    source = Path(path)
    try:
        frame = pd.read_parquet(source)
    except Exception as exc:
        report = QualityReport(str(source), expected_season, 0, 0)
        report.issues.append(
            QualityIssue("unreadable_parquet", "critical", 1, f"Could not read Parquet: {type(exc).__name__}: {exc}")
        )
        return report
    return audit_possession_frame(frame, path=str(source), expected_season=expected_season)


def quarantine_invalid_games(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove whole games touched by structural corruption.

    A duplicated player in one possession usually means lineup reconstruction is
    broken until a later substitution. Dropping only the flagged row would retain
    contaminated neighboring possessions, so the quarantine unit is the game.
    """
    reasons: dict[str, set[str]] = {}

    def mark(mask: pd.Series, reason: str) -> None:
        for game_id in frame.loc[mask, "gameid"].astype(str).unique():
            reasons.setdefault(game_id, set()).add(reason)

    mark(frame.duplicated(list(POSSESSION_KEY), keep=False), "duplicate_possession_key")
    mark(frame.loc[:, AWAY_PLAYER_COLUMNS].nunique(axis=1) != 5, "invalid_away_lineup")
    mark(frame.loc[:, HOME_PLAYER_COLUMNS].nunique(axis=1) != 5, "invalid_home_lineup")

    overlap = _lineup_overlap_mask(frame)
    mark(overlap, "player_on_both_teams")

    invalid_ids = pd.Series(False, index=frame.index)
    for column in PLAYER_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_ids |= values.isna() | (values <= 0)
    mark(invalid_ids, "invalid_player_id")

    points = pd.to_numeric(frame["pts"], errors="coerce")
    mark(points.isna() | (points < 0) | (points > 20), "invalid_points")
    mark(~frame["home_poss"].astype(str).isin({"0", "1", "False", "True", "false", "true"}), "invalid_home_flag")

    bad_games = set(reasons)
    rejected = frame.loc[frame["gameid"].astype(str).isin(bad_games)].copy()
    if not rejected.empty:
        rejected["quarantine_reasons"] = rejected["gameid"].astype(str).map(
            lambda game_id: ",".join(sorted(reasons[game_id]))
        )
    valid = frame.loc[~frame["gameid"].astype(str).isin(bad_games)].copy()
    counts: dict[str, int] = {"games": len(bad_games), "rows": int(len(rejected))}
    for reason_set in reasons.values():
        for reason in reason_set:
            counts[reason] = counts.get(reason, 0) + 1
    return valid, rejected, counts
