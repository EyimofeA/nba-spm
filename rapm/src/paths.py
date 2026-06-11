"""Central path constants for the RAPM sub-project.

All RAPM scripts resolve paths through this module so they work regardless of
the directory they are launched from.
"""
from pathlib import Path

RAPM_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = RAPM_ROOT.parent

DATA = RAPM_ROOT / "data"
OUTPUTS = RAPM_ROOT / "outputs"
DUMP = OUTPUTS / "dump"
RAPM_RESULTS = OUTPUTS / "rapm_results"
DIAGNOSTICS_DIR = OUTPUTS / "diagnostics"
CAREER_DIR = OUTPUTS / "career"                     # per-season 1-year RAPM runs
AGING_DIR = OUTPUTS / "aging"                       # aging curve + age-adjusted outputs

# Primary RAPM inputs
ALL_NAMES_CSV = DATA / "all_names.csv"
PRIOR_CSV = DATA / "prior.csv"                 # local snapshot used by RAPM
SPM_PRIOR_CSV = PROJECT_ROOT / "data" / "outputs" / "prior.csv"  # fresh from SPM pipeline

# Shared with the SPM pipeline
PLAYERSHEETS_YEAR_TOTALS = PROJECT_ROOT / "data" / "raw" / "playersheets" / "year_totals"

# Combined / summary outputs
COMBINED_EXPERIMENTAL = OUTPUTS / "Combined_Rapm_experimental.csv"
COMBINED_DUMMY = OUTPUTS / "Combined_Rapm.csv"
RAPM_WITH_PRIOR_ALL = OUTPUTS / "RAPM_with_prior_all_seasons.csv"
COMPREHENSIVE_RAPM = OUTPUTS / "comprehensive_rapm_1997_2024.csv"
FAST_OPT_RESULTS = OUTPUTS / "fast_optimization_results.json"
QUICK_OPT_RESULTS = OUTPUTS / "quick_meta_optimization_results.json"
META_OPT_RESULTS = OUTPUTS / "meta_column_optimization_results.json"

# rapm_core outputs
RAPM_CORE_DUMP = DUMP                              # raw coefficients (rapm_core_*.csv)
RAPM_CORE_RESULTS = RAPM_RESULTS                   # human-readable (Core_Rapm_*.csv)
COMBINED_CORE = OUTPUTS / "Combined_Rapm_core.csv"

# rebuilt standard RAPM outputs
STANDARD_RAPM_DUMP = DUMP                          # raw coefficients + cached priors
STANDARD_RAPM_RESULTS = RAPM_RESULTS               # human-readable standard RAPM CSVs
STANDARD_RAPM_DIAGNOSTICS = DIAGNOSTICS_DIR / "standard_rapm"

# Career / aging outputs
CAREER_RAPM_CSV = CAREER_DIR / "career_rapm_1year.csv"       # long table: season × player
CAREER_META_JSON = CAREER_DIR / "career_rapm_1year_meta.json"
AGING_CURVE_CSV = AGING_DIR / "aging_curve.csv"              # smoothed Δ per age
AGE_ADJUSTED_CSV = AGING_DIR / "age_adjusted_rapm.csv"       # season × player, age-adjusted
PEAK_RAPM_CSV = AGING_DIR / "peak_rapm.csv"                  # player — peak age-adjusted RAPM
PEAK_RAW_RAPM_CSV = AGING_DIR / "peak_raw_rapm.csv"          # player — peak single-season raw RAPM
CAREER_SUMMARY_CSV = AGING_DIR / "career_summary.csv"        # player — mean + weighted mean across career

# Selection-aware career age model outputs
CAREER_AGE_MODEL_CURVE_CSV = AGING_DIR / "career_age_model_aging_curve.csv"
CAREER_AGE_MODEL_SEASONS_CSV = AGING_DIR / "career_age_model_player_seasons.csv"
CAREER_AGE_MODEL_PEAK_CSV = AGING_DIR / "career_age_model_peak.csv"
CAREER_AGE_MODEL_PEAK_3YR_CSV = AGING_DIR / "career_age_model_peak_3yr.csv"
CAREER_AGE_MODEL_SUMMARY_CSV = AGING_DIR / "career_age_model_summary.csv"
CAREER_AGE_MODEL_META_JSON = AGING_DIR / "career_age_model_meta.json"

# Joint full-span player-season career RAPM
JOINT_CAREER_DIR = OUTPUTS / "career_joint"
JOINT_CAREER_PLAYER_SEASONS_CSV = JOINT_CAREER_DIR / "joint_career_player_seasons.csv"
JOINT_CAREER_SUMMARY_CSV = JOINT_CAREER_DIR / "joint_career_summary.csv"
JOINT_CAREER_PEAK_3YR_CSV = JOINT_CAREER_DIR / "joint_career_peak_3yr.csv"
JOINT_CAREER_CONTEXT_CSV = JOINT_CAREER_DIR / "joint_career_context.csv"
JOINT_CAREER_META_JSON = JOINT_CAREER_DIR / "joint_career_meta.json"


def ensure_dirs() -> None:
    for p in (
        DATA,
        OUTPUTS,
        DUMP,
        RAPM_RESULTS,
        DIAGNOSTICS_DIR,
        STANDARD_RAPM_DIAGNOSTICS,
        CAREER_DIR,
        AGING_DIR,
        JOINT_CAREER_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
