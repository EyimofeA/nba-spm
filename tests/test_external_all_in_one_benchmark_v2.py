from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from research.run_external_all_in_one_benchmark_v2 import (
    component_frame,
    fit_box15_2014_onward,
    name_dimension,
    read_xlsx_sheet,
    season_end,
)


def test_read_xlsx_sheet_resolves_named_sheet_and_shared_strings(tmp_path: Path) -> None:
    path = tmp_path / "history.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Full DPM History" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<si><t>nba_id</t></si><si><t>season</t></si></sst>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2024</v></c></row>'
            '</sheetData></worksheet>',
        )
    result = read_xlsx_sheet(path, "Full DPM History")
    assert result.to_dict("records") == [{"nba_id": "1", "season": "2024"}]


def test_season_end_uses_repository_end_year_convention() -> None:
    assert season_end("2017-18") == 2018
    assert season_end("1999-00") == 2000
    assert season_end(2024) == 2024


def test_component_frame_preserves_side_identity() -> None:
    source = pd.DataFrame(
        {"id": [1], "season": ["2017-18"], "off": [2.0], "def": [-0.5]}
    )
    result = component_frame(
        source,
        candidate="test",
        id_column="id",
        season_column="season",
        offense_column="off",
        defense_column="def",
    )
    assert result.iloc[0]["rating_season"] == 2018
    assert result.iloc[0]["net"] == 1.5


def test_name_dimension_removes_ambiguous_season_names() -> None:
    epm = pd.DataFrame(
        {
            "EPM_player_id": [1, 2],
            "EPM_player_name": ["Same Name", "Same Name"],
            "EPM_season": ["2017-18", "2017-18"],
        }
    )
    lebron = pd.DataFrame(
        {"nba_id": [3], "Player": ["Unique Name"], "Season": [2018]}
    )
    result = name_dimension(epm, lebron)
    assert set(result["PLAYER_ID"]) == {3}


def test_restricted_box15_has_complete_scored_seasons_and_side_identity() -> None:
    result = fit_box15_2014_onward()
    assert set(result["rating_season"]) == set(range(2017, 2025))
    assert not result.duplicated(["rating_season", "PLAYER_ID"]).any()
    assert (result["offense"] + result["defense"] - result["net"]).abs().max() < 1e-12
