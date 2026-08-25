"""Coach ledger construction and joint player-coach RAPM diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve

from nba_impact.models.rapm import RapmConfig, RapmDesign, _penalty
from nba_impact.models.age_adjusted_rapm import AgeDesign


BBREF_TEAM_ALIASES = {
    "BRK": "BKN",
    "NJN": "BKN",
    "SEA": "OKC",
    "VAN": "MEM",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHH": "CHA",
    "CHO": "CHA",
    "WSB": "WAS",
    "PHO": "PHX",
}


class _CoachTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.cell_stat: str | None = None
        self.cell_text: list[str] = []
        self.row: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []
        self.coach_href: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.row = {}
            self.coach_href = None
        elif self.in_row and tag in {"th", "td"}:
            self.cell_stat = attributes.get("data-stat")
            self.cell_text = []
        elif self.in_row and tag == "a" and self.cell_stat == "coach":
            self.coach_href = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self.in_row and self.cell_stat:
            self.cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and tag in {"th", "td"} and self.cell_stat:
            self.row[self.cell_stat] = " ".join("".join(self.cell_text).split())
            self.cell_stat = None
            self.cell_text = []
        elif tag == "tr" and self.in_row:
            if {"coach", "team", "cur_g"}.issubset(self.row) and self.row["cur_g"]:
                output = dict(self.row)
                output["coach_href"] = self.coach_href or ""
                output["source_row"] = str(len(self.rows))
                self.rows.append(output)
            self.in_row = False


def parse_bbref_coaches(html: str, *, season: int) -> pd.DataFrame:
    """Parse the coach, team, and regular-season games fields from one page."""
    parser = _CoachTableParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError(f"No coach rows found for season {season}.")
    frame = pd.DataFrame(parser.rows)
    frame["season"] = int(season)
    frame["games"] = pd.to_numeric(frame["cur_g"], errors="coerce")
    frame = frame.dropna(subset=["games"]).loc[lambda value: value["games"].gt(0)].copy()
    frame["games"] = frame["games"].astype(int)
    frame["team_tricode"] = frame["team"].replace(BBREF_TEAM_ALIASES)
    frame["coach_id"] = frame["coach_href"].str.extract(r"/coaches/([^/.]+)")
    if frame["coach_id"].isna().any():
        raise ValueError(f"Coach IDs are incomplete for season {season}.")
    frame["source_row"] = pd.to_numeric(frame["source_row"], errors="raise").astype(int)
    return frame[
        ["season", "team_tricode", "coach_id", "coach", "games", "source_row"]
    ].reset_index(drop=True)


def build_coach_game_ledger(
    coach_seasons: pd.DataFrame,
    games: pd.DataFrame,
    team_dim: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign coach spans to each full-schedule team-game in chronological order."""
    team_lookup = team_dim.set_index("canonical_tricode")["team_id"]
    coaches = coach_seasons.copy()
    coaches["team_id"] = coaches["team_tricode"].map(team_lookup)
    if coaches["team_id"].isna().any():
        missing = sorted(coaches.loc[coaches["team_id"].isna(), "team_tricode"].unique())
        raise ValueError(f"Coach teams are missing NBA IDs: {missing}.")
    coaches["team_id"] = coaches["team_id"].astype(int)
    home = games.rename(columns={"home_team_id": "team_id"})[
        ["project_season", "game_id", "game_date", "team_id"]
    ]
    away = games.rename(columns={"away_team_id": "team_id"})[
        ["project_season", "game_id", "game_date", "team_id"]
    ]
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(
        ["project_season", "team_id", "game_date", "game_id"], kind="stable"
    )
    team_games["team_game_number"] = (
        team_games.groupby(["project_season", "team_id"]).cumcount() + 1
    )
    rows: list[pd.DataFrame] = []
    audit_rows = []
    for (season, team_id), scheduled in team_games.groupby(
        ["project_season", "team_id"], sort=False
    ):
        assignments = coaches.loc[
            coaches["season"].eq(season) & coaches["team_id"].eq(team_id)
        ].sort_values("source_row", kind="stable")
        scheduled_games = len(scheduled)
        listed_games = int(assignments["games"].sum())
        audit_rows.append(
            {
                "season": int(season),
                "team_id": int(team_id),
                "scheduled_games": scheduled_games,
                "listed_coach_games": listed_games,
                "difference": listed_games - scheduled_games,
                "coach_count": int(len(assignments)),
            }
        )
        if assignments.empty:
            continue
        if len(assignments) == 1:
            assigned = scheduled.copy()
            assigned["coach_id"] = assignments.iloc[0]["coach_id"]
            assigned["coach_name"] = assignments.iloc[0]["coach"]
            rows.append(assigned)
            continue
        if listed_games != scheduled_games:
            continue
        lower = 1
        assigned = scheduled.copy()
        assigned["coach_id"] = None
        assigned["coach_name"] = None
        for coach in assignments.itertuples(index=False):
            upper = lower + int(coach.games) - 1
            mask = assigned["team_game_number"].between(lower, upper)
            assigned.loc[mask, "coach_id"] = coach.coach_id
            assigned.loc[mask, "coach_name"] = coach.coach
            lower = upper + 1
        if assigned["coach_id"].isna().any():
            raise AssertionError("Coach assignment left an in-scope team-game unassigned.")
        rows.append(assigned)
    audit = pd.DataFrame(audit_rows)
    ledger = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not ledger.empty and ledger.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Coach ledger has duplicate game-team keys.")
    return ledger, audit


@dataclass(frozen=True)
class CoachRapmFit:
    player_beta: np.ndarray
    coach_beta: np.ndarray
    intercept: float
    coaches: np.ndarray


@dataclass(frozen=True)
class CoachAgeRapmFit:
    player_beta: np.ndarray
    age_beta: np.ndarray
    coach_beta: np.ndarray
    intercept: float
    coaches: np.ndarray


def build_coach_design(
    player_design: RapmDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    coach_matrix, coaches = build_coach_matrix(
        player_design, offense_coach, defense_coach
    )
    return hstack([player_design.X, coach_matrix], format="csr"), coaches


def build_coach_matrix(
    player_design: RapmDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Build only the two-sided coach indicator block."""
    coaches = np.asarray(sorted(set(offense_coach) | set(defense_coach)), dtype=object)
    lookup = {value: index for index, value in enumerate(coaches)}
    n_rows = player_design.X.shape[0]
    rows = np.concatenate([np.arange(n_rows), np.arange(n_rows)])
    columns = np.concatenate(
        [
            np.fromiter((lookup[value] for value in offense_coach), dtype=np.int64),
            len(coaches)
            + np.fromiter((lookup[value] for value in defense_coach), dtype=np.int64),
        ]
    )
    matrix = csr_matrix(
        (np.ones(2 * n_rows), (rows, columns)),
        shape=(n_rows, 2 * len(coaches)),
    )
    return matrix, coaches


def fit_joint_coach_rapm(
    player_design: RapmDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
    config: RapmConfig,
    *,
    coach_penalty: float,
    row_mask: np.ndarray,
) -> CoachRapmFit:
    if coach_penalty <= 0:
        raise ValueError("coach_penalty must be positive.")
    X_all, coaches = build_coach_design(player_design, offense_coach, defense_coach)
    X = X_all[row_mask]
    y = player_design.y[row_mask]
    intercept = float(y.mean())
    penalties = np.concatenate(
        [
            _penalty(config, len(player_design.players)),
            np.full(2 * len(coaches), coach_penalty, dtype=float),
        ]
    )
    lhs = (X.T @ X).tocsr() + diags(penalties, format="csr")
    rhs = X.T @ (y - intercept)
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta)
    player_columns = player_design.X.shape[1]
    player_beta = beta[:player_columns]
    coach_beta = beta[player_columns:]
    n_players = len(player_design.players)
    player_off_counts = np.asarray(X[:, :n_players].sum(axis=0)).ravel()
    player_def_counts = np.asarray(X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    coach_start = player_columns
    coach_off_counts = np.asarray(X[:, coach_start : coach_start + len(coaches)].sum(axis=0)).ravel()
    coach_def_counts = np.asarray(X[:, coach_start + len(coaches) :].sum(axis=0)).ravel()
    means = [
        float(np.average(player_beta[:n_players], weights=player_off_counts)),
        float(np.average(player_beta[n_players : 2 * n_players], weights=player_def_counts)),
        float(np.average(coach_beta[: len(coaches)], weights=coach_off_counts)),
        float(np.average(coach_beta[len(coaches) :], weights=coach_def_counts)),
    ]
    player_beta[:n_players] -= means[0]
    player_beta[n_players : 2 * n_players] -= means[1]
    coach_beta[: len(coaches)] -= means[2]
    coach_beta[len(coaches) :] -= means[3]
    intercept += 5.0 * (means[0] + means[1]) + means[2] + means[3]
    return CoachRapmFit(player_beta, coach_beta, intercept, coaches)


def predict_joint_coach_rapm(
    fit: CoachRapmFit,
    player_design: RapmDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
) -> np.ndarray:
    X, coaches = build_coach_design(player_design, offense_coach, defense_coach)
    if not np.array_equal(coaches, fit.coaches):
        raise ValueError("Coach vocabulary changed between fit and prediction.")
    return fit.intercept + np.asarray(
        X @ np.concatenate([fit.player_beta, fit.coach_beta])
    ).ravel()


def coach_ratings(fit: CoachRapmFit) -> pd.DataFrame:
    n = len(fit.coaches)
    offense = 100.0 * fit.coach_beta[:n]
    defense = -100.0 * fit.coach_beta[n:]
    return pd.DataFrame(
        {"coach_id": fit.coaches, "offense": offense, "defense": defense, "net": offense + defense}
    ).sort_values("net", ascending=False)


def fit_joint_coach_age_rapm(
    player_design: RapmDesign,
    age_design: AgeDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
    config: RapmConfig,
    *,
    age_penalty: float,
    coach_penalty: float,
    row_mask: np.ndarray | None = None,
) -> CoachAgeRapmFit:
    """Fit player, lineup-age, and coach effects in one ridge system."""
    if age_penalty < 0 or coach_penalty <= 0:
        raise ValueError("Age penalty must be nonnegative and coach penalty positive.")
    coach_x, coaches = build_coach_matrix(
        player_design, offense_coach, defense_coach
    )
    mask = None if row_mask is None else np.asarray(row_mask, dtype=bool)
    if mask is not None and (mask.shape != (player_design.X.shape[0],) or not mask.any()):
        raise ValueError("row_mask must select at least one design row.")
    player_x = player_design.X if mask is None else player_design.X[mask]
    age_x = age_design.X if mask is None else age_design.X[mask]
    coach_selected = coach_x if mask is None else coach_x[mask]
    matrix = hstack(
        [player_x, age_x, coach_selected], format="csr"
    )
    y = player_design.y if mask is None else player_design.y[mask]
    intercept = float(y.mean())
    penalties = np.concatenate(
        [
            _penalty(config, len(player_design.players)),
            np.full(age_design.X.shape[1], age_penalty, dtype=float),
            np.full(coach_x.shape[1], coach_penalty, dtype=float),
        ]
    )
    lhs = (matrix.T @ matrix).tocsr() + diags(penalties, format="csr")
    rhs = np.asarray(matrix.T @ (y - intercept)).ravel()
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta, dtype=float)
    age_start = player_design.X.shape[1]
    coach_start = age_start + age_design.X.shape[1]
    player_beta = beta[:age_start].copy()
    age_beta = beta[age_start:coach_start].copy()
    coach_beta = beta[coach_start:].copy()

    n_players = len(player_design.players)
    n_coaches = len(coaches)
    blocks = (
        (player_beta[:n_players], player_x[:, :n_players]),
        (player_beta[n_players : 2 * n_players], player_x[:, n_players : 2 * n_players]),
        (coach_beta[:n_coaches], coach_selected[:, :n_coaches]),
        (coach_beta[n_coaches:], coach_selected[:, n_coaches:]),
    )
    means = []
    for values, block in blocks:
        exposure = np.asarray(block.sum(axis=0)).ravel()
        means.append(float(np.average(values, weights=exposure)))
        values -= means[-1]
    intercept += 5.0 * (means[0] + means[1]) + means[2] + means[3]
    return CoachAgeRapmFit(
        player_beta=player_beta,
        age_beta=age_beta,
        coach_beta=coach_beta,
        intercept=intercept,
        coaches=coaches,
    )


def predict_joint_coach_age_rapm(
    fit: CoachAgeRapmFit,
    player_design: RapmDesign,
    age_design: AgeDesign,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
    include_age: bool = True,
    include_coach: bool = True,
) -> np.ndarray:
    coach_x, coaches = build_coach_matrix(
        player_design, offense_coach, defense_coach
    )
    if not np.array_equal(coaches, fit.coaches):
        raise ValueError("Coach vocabulary changed between fit and prediction.")
    mask = None if row_mask is None else np.asarray(row_mask, dtype=bool)
    player_x = player_design.X if mask is None else player_design.X[mask]
    age_x = age_design.X if mask is None else age_design.X[mask]
    coach_selected = coach_x if mask is None else coach_x[mask]
    prediction = fit.intercept + np.asarray(
        player_x @ fit.player_beta
    ).ravel()
    if include_age:
        prediction += np.asarray(age_x @ fit.age_beta).ravel()
    if include_coach:
        prediction += np.asarray(coach_selected @ fit.coach_beta).ravel()
    return prediction


def coach_age_ratings(fit: CoachAgeRapmFit) -> pd.DataFrame:
    n = len(fit.coaches)
    offense = 100.0 * fit.coach_beta[:n]
    defense = -100.0 * fit.coach_beta[n:]
    return pd.DataFrame(
        {
            "coach_id": fit.coaches,
            "offense": offense,
            "defense": defense,
            "net": offense + defense,
        }
    ).sort_values("net", ascending=False)
