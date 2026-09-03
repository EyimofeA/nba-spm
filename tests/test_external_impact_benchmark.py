from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.models.external_impact_benchmark import (
    build_external_impact_benchmark,
    normalize_player_name,
    parse_bpm_html,
    parse_xrapm_html,
)


XRAPM_HTML = """
<table id="sortableTable"><thead><tr><th>Player</th><th>Offense</th>
<th>Defense(*)</th><th>Total</th></tr></thead><tbody>
<td>Nikola Jokić</td><td>6.1 (99)</td><td>-2.5 (97)</td><td>8.6 (99)</td></tr>
<td>Test Player</td><td>-1.0 (10)</td><td>1.0 (10)</td><td>-2.0 (10)</td></tr>
</tbody></table>
"""

BPM_HTML = """
<table id="advanced"><thead><tr><th>Player</th><th>Team</th><th>MP</th><th>OBPM</th>
<th>DBPM</th><th>BPM</th></tr></thead><tbody>
<tr><td>Nikola Jokić</td><td>DEN</td><td>2000</td><td>5.0</td><td>2.0</td><td>7.0</td></tr>
<tr><td>Test Player</td><td>2TM</td><td>1000</td><td>-1.0</td><td>-1.0</td><td>-2.0</td></tr>
<tr><td>Test Player</td><td>AAA</td><td>600</td><td>-1.0</td><td>-1.0</td><td>-2.0</td></tr>
<tr><td>Test Player</td><td>BBB</td><td>400</td><td>-1.0</td><td>-1.0</td><td>-2.0</td></tr>
</tbody></table>
"""


def test_external_parsers_normalize_signs_and_names() -> None:
    xrapm = parse_xrapm_html(XRAPM_HTML, 2024)
    bpm = parse_bpm_html(BPM_HTML, 2024)
    assert normalize_player_name("Nikola Jokić") == "nikola jokic"
    assert normalize_player_name("Ömer Aşık") == "omer asik"
    assert normalize_player_name("Marcus Morris Sr.") == "marcus morris"
    jokic = xrapm.loc[xrapm["normalized_name"].eq("nikola jokic")].iloc[0]
    assert jokic["xrapm_defense"] == 2.5
    assert jokic["xrapm_net"] == 8.6
    assert bpm.loc[bpm["normalized_name"].eq("nikola jokic"), "bpm_net"].iloc[0] == 7.0


def test_external_parsers_accept_current_team_column_and_legacy_team_labels() -> None:
    current = XRAPM_HTML.replace("<th>Player</th>", "<th>Player</th><th>Team</th>")
    current = current.replace("<td>Nikola Jokić</td>", "<td>Nikola Jokić</td><td>DEN</td>")
    current = current.replace("<td>Test Player</td>", "<td>Test Player</td><td>AAA</td>")
    pd.testing.assert_frame_equal(parse_xrapm_html(current, 2026), parse_xrapm_html(XRAPM_HTML, 2026))
    legacy = BPM_HTML.replace("<th>Team</th>", "<th>Tm</th>").replace("2TM", "TOT")
    assert len(parse_bpm_html(legacy, 2014)) == 2


def test_xrapm_ambiguous_names_require_explicit_exclusion_and_receipt() -> None:
    duplicate = "<tr><td>Test Player</td><td>2</td><td>1</td><td>1</td></tr>"
    html = XRAPM_HTML.replace("</tbody>", duplicate + "</tbody>")
    with pytest.raises(ValueError, match="duplicate"):
        parse_xrapm_html(html, 2008)
    frame = parse_xrapm_html(html, 2008, exclude_ambiguous_names=True)
    assert len(frame) == 1
    assert len(frame.attrs["excluded_ambiguous_names"]) == 2


def test_external_benchmark_builds_unique_matched_windows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for season in (2017, 2018, 2019):
        for source, html in (
            ("xrapm", XRAPM_HTML),
            ("basketball_reference_bpm", BPM_HTML),
        ):
            path = raw / source / f"season={season}" / "page.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)
    priors = pd.DataFrame(
        [
            {
                "PLAYER_ID": player_id,
                "Window_End": 2019,
                "prior_offense_per_100": offense,
                "prior_defense_per_100": defense,
                "prior_net_per_100": offense + defense,
            }
            for player_id, offense, defense in ((1, 5.0, 2.0), (2, -1.0, -1.0))
        ]
    )
    features = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Window_End": [2019, 2019],
            "OffPoss": [5000, 5000],
            "DefPoss": [5000, 5000],
        }
    )
    names = pd.DataFrame(
        {"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Nikola Jokic", "Test Player"]}
    )
    priors_path = tmp_path / "priors.parquet"
    features_path = tmp_path / "features.parquet"
    names_path = tmp_path / "names.csv"
    priors.to_parquet(priors_path, index=False)
    features.to_parquet(features_path, index=False)
    names.to_csv(names_path, index=False)
    run = build_external_impact_benchmark(
        priors_path,
        features_path,
        names_path,
        raw,
        artifact_root=tmp_path,
        window_ends=(2019,),
    )
    output = Path(run["artifact_path"])
    matched = pd.read_parquet(output / "matched_player_windows.parquet")
    assert len(matched) == 2
    assert not matched.duplicated(["PLAYER_ID", "Window_End"]).any()
    assert matched["xrapm_net"].notna().all()
    assert run["quality"]["minimum_spm_to_xrapm_match_rate"] == 1.0
