"""Read-only chronology audit. Prints aggregates; never rewrites the checkpoint."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file
from nba_impact.models.win_probability_rapm import build_log_odds_wp_target


def main():
    root = Path(__file__).resolve().parents[3]
    path = root / "research/rapm_lab/outputs/wp_spm_aio/checkpoints/wp_target_2014_2026.parquet"
    columns = [
        "possession_id", "gameid", "period", "num", "season",
        "possession_index_before", "home_poss", "home_win",
        "probability_context", "offense_wp_change",
    ]
    source = pd.read_parquet(path, columns=columns)
    backwards = source.groupby("gameid", sort=False)["possession_index_before"].diff().lt(0)
    corrected, _ = build_log_odds_wp_target(source, epsilon=0.025)
    old = source.set_index("possession_id")["offense_wp_change"]
    aligned_old = old.reindex(corrected["possession_id"]).to_numpy()
    changed = ~np.isclose(aligned_old, corrected["offense_wp_change"], rtol=0, atol=1e-12)
    terminal = corrected.groupby("gameid", sort=False).tail(1).index
    nonterminal = corrected.drop(index=terminal)["home_log_odds_change"].abs()
    result = {
        "source": str(path.relative_to(root)), "sha256": sha256_file(path),
        "rows": len(source), "games": int(source["gameid"].nunique()),
        "nonfinite_probability_rows": int((~np.isfinite(source["probability_context"])).sum()),
        "duplicated_game_num_rows": int(source.duplicated(["gameid", "num"], keep=False).sum()),
        "backward_steps": int(backwards.sum()),
        "backward_steps_by_season": backwards.groupby(source["season"]).sum().astype(int).to_dict(),
        "changed_credit_by_season": pd.Series(changed, index=corrected.index).groupby(corrected["season"]).sum().astype(int).to_dict(),
        "corrected_nonterminal_abs_logit_median": float(nonterminal.median()),
        "corrected_nonterminal_abs_logit_p99": float(nonterminal.quantile(0.99)),
        "corrected_all_abs_logit_median": float(corrected["home_log_odds_change"].abs().median()),
        "corrected_all_abs_logit_p99": float(corrected["home_log_odds_change"].abs().quantile(0.99)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
