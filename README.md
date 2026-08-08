# New SPM

NBA player-impact modeling workspace. Three sub-projects live side by side:

- **SPM** — gradient-boosted predictions of Offensive / Defensive RAPM from box
  and tracking stats. Produces a per-player _prior_ used by the RAPM run.
- **RAPM** (`rapm/`) — ridge regression on play-by-play possessions, with
  optional SPM priors and playoff / meta-column variants.
- **zTS** (`zts/`) — playtype-adjusted relative True Shooting.
- **NBA Impact** (`src/nba_impact/`) — canonical current events, lineups,
  possessions, RAPM, and win-probability research with external benchmarks.

See [`AGENTS.md`](./AGENTS.md) for the full layout, data flow, and rules.
Follow [`ROADMAP.md`](./ROADMAP.md) for the active queue and
[`docs/README.md`](./docs/README.md) for the documentation index. The frozen WP
model card is in [`docs/win_probability/MODEL_CARD.md`](./docs/win_probability/MODEL_CARD.md).
Sub-projects have their own charters in [`rapm/AGENTS.md`](./rapm/AGENTS.md)
and [`zts/AGENTS.md`](./zts/AGENTS.md).

## Quick start

Python 3.11+ is assumed.

```bash
pip install -r requirements.txt
```

The clean package can also be installed with `pip install -e .`. Its current
WP benchmark is resumable and compares identical play states:

```bash
uv run python -m nba_impact.cli ingest-espn-win-probability --seasons 2025-26
uv run python -m nba_impact.cli benchmark-win-probability \
  --model-run artifacts/models/win_probability_ablation/<run-id>
uv run python -m nba_impact.cli compare-wp-lineup-strength
uv run python -m nba_impact.cli compare-wp-possession
uv run python -m nba_impact.cli compare-wp-stage1
uv run python -m nba_impact.cli compare-wp-mlp
uv run python -m nba_impact.cli compare-rapm-lineups
```

The three-season canonical data rebuild is also resumable:

```bash
uv run python -m nba_impact.cli ingest \
  --manifest configs/ingest/nba_data_archive_2023.json
uv run python -m nba_impact.cli build-game-dim
uv run python -m nba_impact.cli build-event-states
uv run python -m nba_impact.cli build-player-games
uv run python -m nba_impact.cli build-lineups
# Only when lineup QA identifies bad historical boxes:
uv run python -m nba_impact.cli ingest-official-boxscores --seasons 2023-24
uv run python -m nba_impact.cli build-player-games
uv run python -m nba_impact.cli build-lineups
uv run python -m nba_impact.cli build-possessions
```

### Regenerate the SPM prior

```bash
python src/fetch_playtypes.py    # refresh Synergy playtype POE features
python src/train_spm.py          # fit Off + Def GBMs (LightGBM / XGBoost)
python src/generate_priors.py    # write data/outputs/prior.csv
python src/apply_prior.py        # Bayesian blend prior + observed RAPM
```

Artifacts land in `models/current/` and `data/outputs/`.

### Run ridge RAPM

Requires a local MySQL with the `nba_api` database loaded from
`rapm/data/Dump20240524.sql` and reachable at `/tmp/mysql.sock`.

```bash
cd rapm
python src/rapm.py                     # experimental meta-column ridge
python src/rapm_with_prior.py          # per-season ridge with SPM prior
python src/playoff_rapm_with_prior.py  # 3-year playoff + prior variants
```

Results write to `rapm/outputs/dump/` (raw coefficients) and
`rapm/outputs/rapm_results/` (human-readable).

### Run ad hoc research

Exploratory one-off studies live under `random research/`.

```bash
rapm/venv/bin/python "random research/age_adjusted_rookie_impact.py"
```

This currently builds rookie cohorts from Basketball Reference debut seasons,
joins RAPM aging outputs and Basketball Reference advanced metrics, and writes
charts/CSVs to `random research/outputs/age_adjusted_rookie_impact/`.

## Layout (one level)

```
src/        SPM pipeline (train, infer, apply)
models/     Serialized SPM models (current/ + rolling/)
data/       raw/, processed/, outputs/
rapm/       Ridge RAPM sub-project (src/, data/, outputs/)
zts/        zTS sub-project (self-contained)
notebooks/  Exploratory notebooks
random research/  Ad hoc analysis scripts + generated research outputs
archive/    Legacy scripts + old CSVs, read-only
```

## Where things live

| Need                   | Path                                                       |
| ---------------------- | ---------------------------------------------------------- |
| Training data          | `data/processed/smaller_player_stats_with_rapm.csv`        |
| Inference data         | `data/processed/merged_per100_dataset.csv`                 |
| Active models          | `models/current/spm_{off,def}_model.pkl`, `spm_scaler.pkl` |
| Generated prior        | `data/outputs/prior.csv`                                   |
| Per-possession DB dump | `rapm/data/Dump20240524.sql` (670 MB)                      |
| RAPM outputs           | `rapm/outputs/{dump,rapm_results}/`                        |
| zTS pipeline           | `zts/compute_zts.ipynb`                                    |
| Rookie impact research | `random research/age_adjusted_rookie_impact.py`            |

## Contributing

Read `AGENTS.md` first. Key rules:

1. Put new scripts under `src/` or `rapm/src/`, never at the workspace root.
   Ad hoc analysis notebooks/scripts may live under `random research/`.
2. Use `paths.py` constants — don't hardcode file paths in scripts.
3. `archive/` is frozen history. Don't depend on anything inside it.
4. Keep `zts/` self-contained.
