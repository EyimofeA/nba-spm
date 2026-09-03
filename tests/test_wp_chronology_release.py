import json

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.rapm import build_design
from research.rapm_lab import run_wp_chronology_release as release
from research.rapm_lab.run_wp_chronology_release import (
    CONTRACT, _past_only_predictions, paired_rmse, publication_gate,
    public_table, select_lambda, label_states,
)


@pytest.fixture
def contract():
    value = json.loads(CONTRACT.read_text())
    value["bootstrap_draws"] = 200
    return value


def predictions(contract, errors):
    return pd.DataFrame([
        {"candidate": f"logit_{penalty}", "outcome_season": season,
         "game_id": f"{season}-{game}", "actual_margin": float(game),
         "predicted_margin": float(game) + errors.get(penalty, 1.0)}
        for penalty in contract["shared_lambdas"]
        for season in contract["development_outcomes"] + contract["diagnostic_outcomes"]
        for game in range(8)
    ])


def test_selects_lowest_competitive_penalty_without_later_results(contract):
    frame = predictions(contract, {100: 1.10, 300: 1.02})
    result = select_lambda(frame, contract)
    assert result["selected_lambda"] == 300
    later = frame.outcome_season.isin(contract["diagnostic_outcomes"])
    frame.loc[later, "predicted_margin"] = 1000
    assert select_lambda(frame, contract) == result


def test_paired_bootstrap_rejects_missing_and_duplicate_games(contract):
    frame = predictions(contract, {})
    names = [f"logit_{value}" for value in contract["shared_lambdas"]]
    with pytest.raises(ValueError, match="identical finite"):
        paired_rmse(frame.iloc[1:], names, contract)
    with pytest.raises(ValueError):
        paired_rmse(pd.concat([frame, frame.iloc[:1]]), names, contract)


def test_equal_season_loss_not_mean_rmse(contract):
    frame = pd.DataFrame({"outcome_season": [2020, 2021, 2021], "game_id": ["a", "b", "c"],
                          "candidate": ["x"] * 3, "actual_margin": [0., 0., 0.],
                          "predicted_margin": [1., 3., 3.]})
    point, _ = paired_rmse(frame, ["x"], contract)
    assert point[0] == pytest.approx(np.sqrt(5))
    frame["actual_margin"] = frame.actual_margin.astype("Float64")
    frame["predicted_margin"] = frame.predicted_margin.astype("Float64")
    nullable, _ = paired_rmse(frame, ["x"], contract)
    np.testing.assert_array_equal(point, nullable)


def test_rejects_missing_season_even_when_retaining_reference(contract):
    frame = predictions(contract, {})
    with pytest.raises(ValueError, match="missing a declared"):
        publication_gate(frame.loc[frame.outcome_season.ne(2026)], 150000, contract)
    with pytest.raises(ValueError, match="missing a declared"):
        select_lambda(frame.loc[frame.outcome_season.ne(2020)], contract)


def test_later_failure_falls_back_without_selecting_another_candidate(contract):
    frame = predictions(contract, {100: 2.0})
    result = publication_gate(frame, 100, contract)
    assert not result["passed"]
    assert result["published_lambda"] == 150000
    assert result["maximum_season_rmse_delta"] == pytest.approx(1)


def test_calibration_never_reads_current_or_future_outcomes():
    frame = pd.DataFrame([
        {"candidate": "x", "outcome_season": year, "game_id": f"{year}-{i}",
         "raw_prediction": float(i), "actual_margin": float(i * 2 + 1)}
        for year in (2015, 2016, 2017) for i in range(5)
    ])
    before = _past_only_predictions(frame, [2016])
    frame.loc[frame.outcome_season.ge(2016), "actual_margin"] = 1000
    after = _past_only_predictions(frame, [2016])
    np.testing.assert_allclose(before.predicted_margin, after.predicted_margin)


def test_official_labels_do_not_change_proxy_features_and_never_fallback():
    states = pd.DataFrame({"season": [2024, 2024], "gameid": ["a", "b"],
                           "home_win": [0, 1], "home_score_diff_before": [2., -3.]})
    scores = pd.DataFrame({"season": [2024, 2024], "gameid": ["a", "b"], "official_home_win": [1, 0]})
    result, changes = label_states(states, scores)
    assert result.home_win.tolist() == [1, 0]
    assert changes == {2024: 2}
    pd.testing.assert_series_equal(result.home_score_diff_before, states.home_score_diff_before)
    with pytest.raises(ValueError, match="no official winner"):
        label_states(states, scores.iloc[:1])
    with pytest.raises(pd.errors.MergeError):
        label_states(states, pd.concat([scores, scores.iloc[:1]]))


def test_public_counts_use_fit_window_and_filter_absent_players():
    rows = []
    for season, offset in ((2025, 0), (2026, 10)):
        for home in (0, 1):
            rows.append({"season": season, "gameid": str(season), "home_poss": home,
                         "pts": .1, **{f"a{i}": offset + i for i in range(1, 6)},
                         **{f"h{i}": offset + i + 5 for i in range(1, 6)}})
    design = build_design(pd.DataFrame(rows))
    names = pd.DataFrame({"PLAYER_ID": range(1, 21), "PLAYER_NAME": [f"Player {i}" for i in range(1, 21)]})
    result = public_table(design, names, 2026, 2026, 100, 100, 300)
    assert set(result.player_id) == set(range(11, 21))
    assert result.off_possessions.eq(1).all() and result.def_possessions.eq(1).all()
    np.testing.assert_allclose(result.net_per_100, result.offense_per_100 + result.defense_per_100)


def test_surface_training_excludes_current_outcomes(monkeypatch, contract):
    rows = [{"season": year, "gameid": f"{year}-{game}", "period": 1, "num": possession,
             "home_poss": possession % 2, "pts": year - 1990.,
             **{f"a{i}": i for i in range(1, 6)}, **{f"h{i}": i + 5 for i in range(1, 6)}}
            for year in (1997, 2014) for game in (0, 1) for possession in range(2)]
    source = pd.DataFrame(rows)
    scores = pd.DataFrame([{"season": year, "gameid": f"{year}-{game}", "official_home_win": game}
                           for year in (1997, 2014) for game in (0, 1)])
    monkeypatch.setattr(release, "load_unified_terminal_possessions", lambda *args, **kwargs: source)
    monkeypatch.setattr(release, "official_scores", lambda *args: scores)
    seen = []

    class Surface:
        coef_ = np.zeros((1, 6))
        intercept_ = np.zeros(1)

        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            assert source.loc[features.index, "season"].eq(1997).all()
            seen.append(labels.tolist())

        def predict_proba(self, features):
            return np.tile([.4, .6], (len(features), 1))

    monkeypatch.setattr(release, "LogisticRegression", Surface)
    before, _ = release.corrected_target(contract, pilot=True)
    current = scores.season.eq(2014)
    scores.loc[current, "official_home_win"] = 1 - scores.loc[current, "official_home_win"]
    after, _ = release.corrected_target(contract, pilot=True)
    assert seen == [[0, 1], [0, 1]]
    np.testing.assert_array_equal(before.probability_context, after.probability_context)


def test_official_scores_exclude_unplayed_but_reject_played_ties(tmp_path, monkeypatch):
    from nba_impact.data.manifest import sha256_file

    monkeypatch.setattr(release, "SCORES", tmp_path)
    path = tmp_path / "project_season=2013/regular.parquet"
    path.parent.mkdir()
    frame = pd.DataFrame({"project_season": [2013, 2013], "season_type": ["regular"] * 2,
                          "game_id": ["unplayed", "played"], "home_score": [0, 101], "away_score": [0, 100]})
    def save():
        frame.to_parquet(path, index=False)
        path.with_suffix(".parquet.manifest.json").write_text(json.dumps({"passed": True, "output_sha256": sha256_file(path)}))
    save()
    assert release.official_scores(2013, 2013).gameid.tolist() == ["played"]
    frame.loc[1, "away_score"] = 101
    save()
    with pytest.raises(ValueError, match="tied official"):
        release.official_scores(2013, 2013)
