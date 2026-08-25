# Feature candidates

## Autoresearch = `build.py`, not column subsets

| type | what it does | autoresearch? |
|------|----------------|---------------|
| **`build`** | `build.py` engineers **new columns** → harness evaluates them | **yes** |
| `subset` | filters existing `spm_features_windows.parquet` cols | no (ablation only) |

## Add a new candidate

```
candidates/gen_NNN/
  manifest.json   # type: build, code, alpha, c_grid
  build.py        # def build(feats) -> (feats, list[new_col_names])
```

`build(feats)` receives the base player-window parquet. Return augmented frame + **only the new column names** you created.

Gen 006 = derived ratios/z-scores (template).
Gen 007 = playtype PPP from staging (sparse until join fixed).

Run: `python3 src/feature_foundry.py N`
