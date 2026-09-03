"""Create and execute a local audit companion; no training or data downloads."""

from pathlib import Path

import nbformat as nb
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
cells = [
    nb.v4.new_markdown_cell("""# Benchmark coverage and PULSE chronology
## tl;dr
PULSE has 12 past-only coefficient-fit folds: ratings 2014–2025, outcomes 2015–2026.
All three internal candidates score 14,439 identical games. These are reused development
seasons, not untouched confirmation. The old external comparison is not current PULSE,
and its internal RAPM updates do not preserve the claimed final common-player support.
The corrected benchmark uses 13,209 identical games from 2016–2026. PULSE RMSE is
13.7545, compared with 13.8644 for RAPM and 13.6791 for xRAPM.

## Context & Methods
This audit profiles existing local inputs and validates saved predictions. It does not
fit models, fetch data, open reserved 2027 outcomes, or change production ratings.
Years identify the year in which an NBA season ends.

### Key Assumptions
File coverage establishes availability, not point-in-time provenance for third-party
training. PIPM 2021 is partial (maximum 22 GP). MAMBA labels after 2024 are not usable
season panels. RAPTOR's historical and modern variants use different information.
The requested main panel uses saved CourtSignal PIPM/RAPTOR reconstructions.
RAPTOR's pooled training through 2022 makes its early results non-chronological.
"""),
    nb.v4.new_code_cell("""from pathlib import Path
import hashlib, json
import pandas as pd
from IPython.display import display
from nba_impact.models.pulse_validation import load_pulse_validation
from nba_impact.api.web_snapshot import _pulse_evidence
from research.run_external_all_in_one_benchmark_v2 import read_xlsx_sheet, season_end

ROOT = Path.cwd()
DOWNLOADS = Path.home() / 'Downloads'
PULSE = ROOT / 'artifacts/models/pulse/pulse_canonical_v1_cd3c14750a'
OLD = ROOT / 'artifacts/research/external_all_in_one_benchmark/external_all_in_one_benchmark_v2_6a898e99d9'
sources = {}
coverage = []
def record(name, path, frame, year_column, off, defense, note='', stop=2026):
    sources[name] = {'file': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    years = frame[year_column].map(season_end)
    valid = frame[off].notna() & frame[defense].notna() & years.le(stop)
    present = sorted(years[valid].unique().tolist())
    coverage.append({'model': name, 'first_rating_year': min(present),
                     'last_rating_year': max(present), 'seasons': len(present),
                     'rows': int(valid.sum()), 'note': note})
"""),
    nb.v4.new_markdown_cell("## Data\n### Source coverage\nNo model is refit. Only non-null offensive and defensive ratings count."),
    nb.v4.new_code_cell("""inputs = [
 ('EPM', 'EPM_All_Seasons.csv', 'EPM_season', 'EPM_off', 'EPM_def'),
 ('LEBRON', 'lebron-data-2026-2025-2024-2023-2022-2021-2020-2019-2018-2017-2016-2015-2014-2013-2012-2011-2010.csv', 'Season', 'O-LEBRON', 'D-LEBRON'),
 ('PIPM', 'PIPM Player Finder through 2021 - Database.csv', 'Season', 'O-PIPM', 'D-PIPM'),
 ('RAPTOR modern', 'Data/modern_RAPTOR_by_player.csv', 'season', 'raptor_offense', 'raptor_defense'),
 ('RAPTOR latest', 'Data/latest_RAPTOR_by_player.csv', 'season', 'raptor_offense', 'raptor_defense'),
 ('RAPTOR historical', 'Data/historical_RAPTOR_by_player.csv', 'season', 'raptor_offense', 'raptor_defense'),
 ('MAMBA', 'MAMBAVALUES.xlsx - Sheet1.csv', 'Season', 'Offense', 'Defense'),
]
for name, filename, year, off, defense in inputs:
    path = DOWNLOADS / filename
    raw = pd.read_csv(path, low_memory=False)
    note = {'PIPM': '2021 is partial; use through 2020 for complete-season comparison',
            'MAMBA': 'Exclude malformed labels after 2024'}.get(name, '')
    record(name, path, raw, year, off, defense, note, 2024 if name == 'MAMBA' else 2026)
path = DOWNLOADS / 'DARKO - Daily Adjusted and Regressed Kalman Optimized projections - Full DPM History.csv'
darko = pd.read_csv(path, low_memory=False)
assert not darko.duplicated(['nba_id', 'season']).any()
assert darko.season.max() == 2026
record('DARKO DPM', path, darko, 'season', 'o_dpm', 'd_dpm',
       'Updated user-supplied CSV replaces the older workbook; per-player date is available.')
path = ROOT / 'research/rapm_lab/data/external/benchmark_20260903/annual_sources.parquet'
raw = pd.read_parquet(path)
for name, off, defense in [('BPM 2.0', 'bpm_offense', 'bpm_defense'), ('xRAPM', 'xrapm_offense', 'xrapm_defense')]:
    record(name, path, raw, 'season', off, defense)
for family, run, prefix in [('pipm', 'pipm_reconstruction_v1_e0625de5fe', 'pipm'),
                            ('raptor', 'raptor_reconstruction_v1_938d1becf9', 'raptor')]:
    path = ROOT / f'research/rapm_lab/outputs/{family}_reconstruction/{run}/reconstructions.parquet'
    record(f'CourtSignal {family.upper()} reconstruction', path, pd.read_parquet(path),
           'season', f'{prefix}_offense', f'{prefix}_defense', 'Saved reconstruction used as-is')
display(pd.DataFrame(coverage))
"""),
    nb.v4.new_code_cell("""pipm = pd.read_csv(DOWNLOADS / inputs[2][1], low_memory=False)
pipm = pipm[pipm.Season.isin(['2018-19', '2019-20', '2020-21'])]
display(pipm.groupby('Season').agg(rows=('Player','size'), max_games=('GP','max')))
mamba = pd.read_csv(DOWNLOADS / inputs[6][1])
display(mamba[mamba.Season.le(2026)].groupby('Season').size().rename('rows').to_frame())
"""),
    nb.v4.new_markdown_cell("## Results\n### Chronological predictions, not full-history display ratings"),
    nb.v4.new_code_cell("""games, folds = load_pulse_validation(PULSE)
display(folds[['rating_season','outcome_season','training_start','training_end']].drop_duplicates())
display(games.groupby('candidate').agg(game_rows=('game_id','size'), seasons=('outcome_season','nunique')))
manifest = json.loads((PULSE/'run.json').read_text())
assert manifest['final_prior']['training_end'] == 2026
assert folds.training_end.lt(folds.rating_season).all()
assert games.outcome_season.eq(games.rating_season + 1).all()
public = json.loads((ROOT/'web/public/data/catalog.json').read_text())
checked_summary = _pulse_evidence(ROOT, manifest)['comparison']
assert checked_summary == public['methods']['pulse']['comparison']
print('All declared folds passed. The existing public internal summary is numerically unchanged.')
for name in ('run.json','validation_games.parquet','validation_folds.parquet','validation_priors.parquet'):
    path = PULSE/name
    sources['PULSE '+name] = {'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
"""),
    nb.v4.new_markdown_cell("""### What the old external table actually measured
The source below contains older Box15, rich-SPM and residual priors. It contains no
current canonical PULSE ratings. The runner intersects priors, then fits each internal
RAPM update on the full player matrix. Thus it shares games and prior coverage, not
identical final player support. Do not republish it under a current PULSE label.
"""),
    nb.v4.new_code_cell("""old = pd.read_parquet(OLD/'ratings.parquet', columns=['candidate','rating_season'])
display(old.groupby('candidate').rating_season.agg(['min','max','nunique']))
assert 'pulse' not in set(old.candidate.str.lower())
print('Website pinned older external run:', public['validation']['metric_comparison']['run_id'])
print('Canonical recorded builder hash:', manifest['source_hashes']['builder'])
actual_builder = hashlib.sha256((ROOT/'src/nba_impact/models/canonical_pulse.py').read_bytes()).hexdigest()
print('Current builder hash:', actual_builder)
print('Builder source hash matches:', actual_builder == manifest['source_hashes']['builder'])
"""),
    nb.v4.new_markdown_cell("""### Corrected common-support comparison
The export verifier checks source and output hashes, chronological PULSE metadata,
exact replay, identical game keys and actual margins, equal final player inclusion masks,
zero coefficients for every excluded player, and independently recomputed RMSE.
It returns summaries only and does not write a website release when imported.
"""),
    nb.v4.new_code_cell("""import runpy
builder = runpy.run_path(str(ROOT / 'web/scripts/build-external-benchmark.py'))
payload = builder['build_payload']()
run = builder['RUN']
for panel in payload['panels']:
    print(panel['scope'], panel['outcome_start'], panel['outcome_end'], panel['games'])
    display(pd.DataFrame(panel['rows'])[['candidate','aggregate_rmse','mean_correlation']])
display(pd.read_parquet(run/'paired_intervals.parquet').query("scope == 'main'"))
replay = json.loads((run/'run.json').read_text())['pulse_replay']
assert len(replay) == 11 and all(row['maximum_prediction_difference'] == 0 for row in replay)
print('All 11 PULSE folds reproduced saved predictions exactly before masking and centering.')
"""),
    nb.v4.new_markdown_cell("""## Takeaways
- The main panel tests ratings 2015–2025 on 2016–2026 games, using identical final
  scored player support for all nine models. MAMBA ends with 2025 outcomes.
- PULSE beats RAPM by 0.1098 RMSE. The paired 95% interval is -0.1347 to -0.0862.
  xRAPM, EPM and DARKO have lower error; the gap to LEBRON is inconclusive.
- PULSE uses past-only prior fits. All 11 saved predictions replayed exactly before
  benchmark normalization. Final full-history display ratings never enter the test.
- CourtSignal RAPTOR and PIPM remain as-is. RAPTOR's pooled 2014–2022 training
  prevents a clean out-of-time claim for its early results. Source-author metric
  files do not certify original training cutoffs or point-in-time publication.
- Historical outcomes are selection-exposed and predictions use observed future
  lineups. This is a matched-support diagnostic, not untouched confirmation.
- Scores use official final margins and a common missing-player/baseline policy.
  They differ from the original unmasked internal PULSE table. No deployment occurred.
"""),
    nb.v4.new_code_cell("""audit = ROOT/'research/audits/benchmark_coverage_20260903'
(audit/'source_coverage.json').write_text(json.dumps({'coverage': coverage, 'sources': sources}, indent=2))
"""),
]
notebook = nb.v4.new_notebook(cells=cells, metadata={"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3",
}})
NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
nb.write(notebook, HERE / "verification.ipynb")
print(HERE / "verification.ipynb")
