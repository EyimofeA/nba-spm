"""Create a finite SPM feature panel without confusing missing data with zero.

The observed-data coverage ledger remains authoritative. This module creates a
separate model-ready panel. Each fill rule follows the feature's unit:

* missing event counts per 100 become zero when exposure exists;
* undefined raw rates use their same-season empirical-Bayes estimate;
* missing level metrics use the same-season median;
* centered source-specific residuals use zero when their source is absent;
* zTS uses every available playtype row, then a season-neutral shot-mix fallback.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .feature_coverage import (
    DFG_FEATURES,
    HUSTLE_FEATURES,
    RIM_DFG_FEATURES,
    feature_source_family,
    normalize_source_keys,
)
from .statistical_features import (
    CORE_RATE_SPECS,
    NATURAL_WEIGHTED_AVERAGES,
    RATIO_SPECS,
    TRACKING_RATE_SPECS,
)


AVAILABILITY_FEATURES = {
    "offense": ("zts_source_tier",),
    "defense": (
        "has_hustle_tracking",
        "has_matchup_tracking",
        "has_dfg_tracking",
        "has_rim_defense_tracking",
    ),
}


def _membership(panel: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    observed = set(map(tuple, normalize_source_keys(keys).to_numpy()))
    return pd.Series(
        [
            (int(player), int(season)) in observed
            for player, season in panel[["PLAYER_ID", "Window_End"]].itertuples(
                index=False
            )
        ],
        index=panel.index,
    )


def _season_neutral(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(frame[feature], errors="coerce")
    season_center = values.groupby(frame["Window_End"]).transform("median")
    return values.fillna(season_center).fillna(values.median()).fillna(0.0)


def _fill_raw_player_sheet_feature(
    output: pd.DataFrame,
    enriched: pd.DataFrame,
    feature: str,
) -> tuple[pd.Series, str]:
    values = pd.to_numeric(output[feature], errors="coerce")
    if feature in CORE_RATE_SPECS or feature in TRACKING_RATE_SPECS:
        return values.fillna(0.0), "zero_event_rate_with_known_exposure"
    if feature in RATIO_SPECS:
        eb_feature = f"{feature}_eb"
        if eb_feature not in enriched:
            raise ValueError(f"No empirical-Bayes fallback for {feature}.")
        fallback = pd.to_numeric(enriched[eb_feature], errors="coerce")
        values = values.fillna(fallback)
        return _season_neutral(output.assign(**{feature: values}), feature), "same_season_eb"
    if feature in NATURAL_WEIGHTED_AVERAGES or feature == "true_shooting_pct":
        return _season_neutral(output, feature), "same_season_median"
    return _season_neutral(output, feature), "same_season_median_fallback"


def complete_selected_feature_panel(
    annual: pd.DataFrame,
    enriched: pd.DataFrame,
    selected: Mapping[str, tuple[str, ...]],
    *,
    strict_playtype: pd.DataFrame,
    loose_playtype: pd.DataFrame,
    source_keys: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], pd.DataFrame, dict]:
    """Return a finite annual panel, expanded contract, and completion ledger."""
    key = ["PLAYER_ID", "Window_End"]
    output = annual.copy()
    enriched = enriched.copy()
    if "Season" in enriched:
        enriched = enriched.rename(columns={"Season": "Window_End"})
    for name, frame in (("annual", output), ("enriched", enriched)):
        if frame.duplicated(key).any():
            raise ValueError(f"{name} features contain duplicate player-season keys.")
    enriched = output[key].merge(enriched, on=key, how="left", validate="one_to_one")

    selected_union = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    if missing := sorted(set(selected_union) - set(output)):
        raise ValueError(f"Annual panel lacks selected fields: {missing}")

    original = output[list(selected_union)].copy()
    ledger_rows: list[dict] = []
    external_families = {"playtype", "hustle", "matchup_defense", "dfg", "rim_dfg"}
    for feature in selected_union:
        family = feature_source_family(feature)
        if family in external_families:
            continue
        raw_missing = int(pd.to_numeric(output[feature], errors="coerce").isna().sum())
        output[feature], method = _fill_raw_player_sheet_feature(
            output, enriched, feature
        )
        ledger_rows.append(
            {
                "feature": feature,
                "source_family": family,
                "raw_missing_rows": raw_missing,
                "source_missing_rows": 0,
                "completion_method": method if raw_missing else "observed",
            }
        )

    normalized_sources = {
        family: normalize_source_keys(frame) for family, frame in source_keys.items()
    }
    masks = {
        family: _membership(output, keys) for family, keys in normalized_sources.items()
    }

    # Hustle rows include explicit zeros. Only source-absent rows receive zero.
    hustle_missing = ~masks["hustle"]
    for feature in selected_union:
        if feature not in HUSTLE_FEATURES:
            continue
        output.loc[hustle_missing, feature] = 0.0
        ledger_rows.append(
            {
                "feature": feature,
                "source_family": "hustle",
                "raw_missing_rows": int(original[feature].isna().sum()),
                "source_missing_rows": int(hustle_missing.sum()),
                "completion_method": "zero_with_source_availability_flag",
            }
        )

    # Seven selected matchup fields are centered EB residuals. The eighth is a
    # block event rate. Zero is the declared no-assignment fallback for both.
    matchup_missing = ~masks["matchup_defense"]
    for feature in selected_union:
        if not feature.startswith("matchup_"):
            continue
        output.loc[matchup_missing, feature] = 0.0
        ledger_rows.append(
            {
                "feature": feature,
                "source_family": "matchup_defense",
                "raw_missing_rows": int(original[feature].isna().sum()),
                "source_missing_rows": int(matchup_missing.sum()),
                "completion_method": "zero_with_source_availability_flag",
            }
        )

    dfg_missing = ~masks["dfg"]
    for feature in selected_union:
        if feature not in DFG_FEATURES:
            continue
        output.loc[dfg_missing, feature] = 0.0
        ledger_rows.append(
            {
                "feature": feature,
                "source_family": "dfg",
                "raw_missing_rows": int(original[feature].isna().sum()),
                "source_missing_rows": int(dfg_missing.sum()),
                "completion_method": "zero_with_source_availability_flag",
            }
        )

    rim_missing = ~masks["rim_dfg"]
    for feature in selected_union:
        if feature not in RIM_DFG_FEATURES:
            continue
        output.loc[rim_missing, feature] = 0.0
        ledger_rows.append(
            {
                "feature": feature,
                "source_family": "rim_dfg",
                "raw_missing_rows": int(original[feature].isna().sum()),
                "source_missing_rows": int(rim_missing.sum()),
                "completion_method": "zero_with_source_availability_flag",
            }
        )

    strict = normalize_source_keys(strict_playtype)
    loose = loose_playtype.rename(columns={"Season": "Window_End"}).copy()
    loose_keys = normalize_source_keys(loose)
    strict_mask = _membership(output, strict)
    loose_mask = _membership(output, loose_keys)
    loose_zts = output[key].merge(
        loose[["PLAYER_ID", "Window_End", "zts_pct_points"]],
        on=key,
        how="left",
        validate="one_to_one",
    )["zts_pct_points"]
    output.loc[loose_mask, "zts_pct_points"] = loose_zts.loc[loose_mask]

    expected_by_season = (
        loose.assign(
            weighted_expected=(
                loose["playtype_expected_ts_pct"] * loose["synergy_possessions"]
            )
        )
        .groupby("Window_End")
        .apply(
            lambda frame: frame["weighted_expected"].sum()
            / frame["synergy_possessions"].sum(),
            include_groups=False,
        )
    )
    player_ts = pd.to_numeric(output["true_shooting_pct"], errors="coerce")
    valid_player_ts = player_ts.between(0.0, 1.5)
    invalid_player_ts = ~valid_player_ts
    season_ts = player_ts.where(valid_player_ts).groupby(output["Window_End"]).transform(
        "median"
    )
    output["true_shooting_pct"] = player_ts.where(valid_player_ts).fillna(
        season_ts
    ).fillna(0.5)
    fallback = ~loose_mask
    fallback_expected = output["Window_End"].map(expected_by_season)
    fallback_player_ts = 100.0 * output["true_shooting_pct"]
    output.loc[fallback, "zts_pct_points"] = (
        fallback_player_ts - fallback_expected
    ).loc[fallback]
    availability = pd.DataFrame(
        {
            "zts_source_tier": np.select(
                [strict_mask, loose_mask], [2.0, 1.0], default=0.0
            ),
            "has_hustle_tracking": masks["hustle"].astype(float),
            "has_matchup_tracking": masks["matchup_defense"].astype(float),
            "has_dfg_tracking": masks["dfg"].astype(float),
            "has_rim_defense_tracking": masks["rim_dfg"].astype(float),
        },
        index=output.index,
    )
    output = pd.concat([output, availability], axis=1)
    ledger_rows.append(
        {
            "feature": "zts_pct_points",
            "source_family": "playtype",
            "raw_missing_rows": int(original["zts_pct_points"].isna().sum()),
            "source_missing_rows": int((~strict_mask).sum()),
            "completion_method": "all_rows_then_season_neutral_shot_mix_fallback",
        }
    )

    expanded = {
        side: tuple(dict.fromkeys((*selected[side], *AVAILABILITY_FEATURES[side])))
        for side in ("offense", "defense")
    }
    expanded_union = tuple(dict.fromkeys((*expanded["offense"], *expanded["defense"])))
    for feature in expanded_union:
        output[feature] = pd.to_numeric(output[feature], errors="coerce")
        if output[feature].isna().any():
            raise ValueError(f"Completion left missing values in {feature}.")
        if not np.isfinite(output[feature]).all():
            raise ValueError(f"Completion left nonfinite values in {feature}.")

    ledger = pd.DataFrame(ledger_rows).drop_duplicates("feature", keep="last")
    ledger["completed_missing_rows"] = [
        int(output[feature].isna().sum()) for feature in ledger["feature"]
    ]
    quality = {
        "rows": int(len(output)),
        "selected_features_before": len(selected_union),
        "selected_features_after": len(expanded_union),
        "missing_values_before": int(original.isna().sum().sum()),
        "missing_values_after": int(output[list(expanded_union)].isna().sum().sum()),
        "strict_zts_rows": int(strict_mask.sum()),
        "low_sample_zts_rows": int((loose_mask & ~strict_mask).sum()),
        "fallback_zts_rows": int((~loose_mask).sum()),
        "invalid_true_shooting_rows_repaired": int(invalid_player_ts.sum()),
        "hustle_source_missing_rows": int(hustle_missing.sum()),
        "matchup_source_missing_rows": int(matchup_missing.sum()),
        "dfg_source_missing_rows": int(dfg_missing.sum()),
        "rim_source_missing_rows": int(rim_missing.sum()),
    }
    keep = ["PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *expanded_union]
    return output[keep].copy(), expanded, ledger, quality
