"""Central path constants for the SPM pipeline.

Every script under ``src/`` resolves its inputs/outputs through the constants
defined here so no script has to care where it is invoked from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PER_YEAR = PROCESSED / "per_year"
OUTPUTS = DATA / "outputs"

MODELS = ROOT / "models"
MODELS_CURRENT = MODELS / "current"
MODELS_ROLLING = MODELS / "rolling"

RAPM_DIR = ROOT / "rapm"
RAPM_DATA = RAPM_DIR / "data"
RAPM_OUTPUTS = RAPM_DIR / "outputs"

# Processed datasets
TRAINING_DATA = PROCESSED / "smaller_player_stats_with_rapm.csv"
INFERENCE_DATA = PROCESSED / "merged_per100_dataset.csv"
PLAYTYPE_POE_FEATURES = PROCESSED / "playtype_poe_features.csv"

# Active model artifacts
SPM_OFF_MODEL = MODELS_CURRENT / "spm_off_model.pkl"
SPM_DEF_MODEL = MODELS_CURRENT / "spm_def_model.pkl"
SPM_SCALER = MODELS_CURRENT / "spm_scaler.pkl"
MODEL_FEATURES = MODELS_CURRENT / "model_features.pkl"

# Pipeline outputs
PRIOR_CSV = OUTPUTS / "prior.csv"
TRAIN_PREDICTIONS = OUTPUTS / "bpm_optimized_predictions.csv"
POSTERIOR_CSV = OUTPUTS / "rapm_posterior.csv"


def ensure_dirs() -> None:
    """Create any output directories that don't exist yet."""
    for p in (PROCESSED, PER_YEAR, OUTPUTS, MODELS_CURRENT, MODELS_ROLLING):
        p.mkdir(parents=True, exist_ok=True)
