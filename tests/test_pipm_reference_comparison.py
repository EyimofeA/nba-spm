from pathlib import Path

import pytest

from research.run_pipm_reference_comparison import parse_pipm_html


def test_parse_pipm_html_uses_data_order_not_rank_text(tmp_path: Path) -> None:
    source = tmp_path / "sample.html"
    source.write_text(
        """
        <table><tbody><tr>
          <td><a href="\\2544RegularSeasonBoxScore.html">LeBron James</a></td>
          <td>LAL</td>
          <td data-order="2316.0">2316 <span>14</span></td>
          <td data-order="4657.0">4657 <span>13</span></td>
          <td data-order="2.75">2.75 <span>10</span></td>
          <td data-order="4707.0">4707 <span>11</span></td>
          <td data-order="4.34">4.34 <span>4</span></td>
          <td data-order="7.09">7.09 <span>4</span></td>
        </tr></tbody></table>
        """,
        encoding="utf-8",
    )

    frame = parse_pipm_html(source, "2019-20")

    assert frame.loc[0, "PLAYER_ID"] == 2544
    assert frame.loc[0, "rating_season"] == 2020
    assert frame.loc[0, "pipm_offense"] == pytest.approx(2.75)
    assert frame.loc[0, "pipm_defense"] == pytest.approx(4.34)
    assert frame.loc[0, "pipm_net"] == pytest.approx(7.09)


def test_parse_pipm_html_rejects_failed_component_identity(tmp_path: Path) -> None:
    source = tmp_path / "bad.html"
    source.write_text(
        """
        <table><tbody><tr>
          <td><a href="\\1RegularSeasonBoxScore.html">Player</a></td><td>AAA</td>
          <td data-order="1">1</td><td data-order="2">2</td>
          <td data-order="1.5">1.5</td><td data-order="2">2</td>
          <td data-order="1.5">1.5</td><td data-order="9">9</td>
        </tr></tbody></table>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="component identity"):
        parse_pipm_html(source, "2019-20")
