from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.rapm_lab.run_external_reproduction_benchmark import (
    comparison_metrics,
    normalize_name,
    parse_darko_wowy,
    parse_xrapm_3y,
    reproduce_aupm,
    ryan_window_bounds,
    season_end,
)


def test_name_and_season_normalization_are_deterministic() -> None:
    assert normalize_name("Nikola Jokić") == "nikolajokic"
    assert normalize_name("NIKOLA JOKIC") == "nikolajokic"
    assert season_end("1997-98") == 1998
    assert season_end("1999-00") == 2000
    assert season_end("2013-14") == 2014
    assert season_end("2014-19") == 2019
    assert season_end("2024-2025") == 2025
    assert ryan_window_bounds("2018-23") == (2019, 2023)


def test_comparison_metrics_reports_scale_and_rank() -> None:
    frame = pd.DataFrame(
        {"reference": [-2.0, 0.0, 2.0], "courtsignal": [-3.0, 1.0, 5.0]}
    )
    metrics = comparison_metrics(frame)
    assert metrics["n"] == 3
    assert np.isclose(metrics["slope"], 2.0)
    assert np.isclose(metrics["intercept"], 1.0)
    assert np.isclose(metrics["spearman"], 1.0)


def test_aupm_formula_reproduces_known_row() -> None:
    frame = pd.DataFrame(
        {
            "OnOffRtg": [110.0],
            "NET_RATING": [4.0],
            "BLK_per100_def": [1.0],
            "DREB_per100_def": [8.0],
            "SumAbove": [3.0],
        }
    )
    expected = (
        -6.357797067568494
        + 0.058647 * 110.0
        + 0.282983 * 4.0
        - 0.143842
        + 0.122480 * 8.0
        + 0.007007 * 3.0
    )
    frame["AuPM"] = expected
    reproduced, maximum_error = reproduce_aupm(frame)
    assert np.isclose(reproduced.iloc[0], expected)
    assert maximum_error == 0.0


def test_xrapm_parser_flips_points_allowed_defense(tmp_path: Path) -> None:
    source = tmp_path / "RAPM_3y.html"
    source.write_text(
        '<a href="https://xrapm.com/player_pages/203999.html">Nikola Jokic</td>\n'
        '<td>DEN</td>\n<td>5.0 (99)</td>\n<td>-1.6 (95)</td>\n'
        '<td class="color">6.6 (99)</td>'
    )
    parsed = parse_xrapm_3y(source)
    assert parsed.iloc[0]["PLAYER_ID"] == 203999
    assert parsed.iloc[0]["reference_defense"] == 1.6
    assert parsed.iloc[0]["reference_net"] == 6.6


def test_darko_parser_keeps_positive_good_defense(tmp_path: Path) -> None:
    source = tmp_path / "season_2026.html"
    source.write_text(
        '{nba_id:203999,season:2026,leaderboard_rank:null,player_name:"Nikola Jokic",'
        'wowy_rapm:8.4,wowy_orapm:6.4,wowy_drapm:2.0,exposure:12000,'
        'season_possessions:null,minutes:null,bpm:null,date:"2026-04-30"}'
    )
    parsed = parse_darko_wowy(source)
    assert parsed.iloc[0]["reference_offense"] == 6.4
    assert parsed.iloc[0]["reference_defense"] == 2.0
    assert parsed.iloc[0]["reference_net"] == 8.4
