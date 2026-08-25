"""Lambda tuning + linear-vs-nonlinear bake-off on forward folds (lab v1).

Part A: lambda search on sufficient statistics, one fold at a time.
Part B: ridge_tuned vs LightGBM vs bilinear embeddings. ExtraTrees remains an
explicit reproducibility option but is excluded from default runs.
Checkpointing: every model x fold result appends to bakeoff_last3.csv;
restarts skip completed pairs. Seasons 2024–2026 are development and
selection only. Season 2027 remains the untouched confirmation.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rapm_ridge import load_canonical, load_cache, player_universe, actual_margins  # noqa: E402

PLAYER_COLUMNS = [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]
GRID_OFF = [500.0, 1000.0, 2000.0, 3000.0, 4500.0]
GRID_DEF = [2000.0, 3000.0, 4500.0, 6000.0, 8000.0]
LAM_HOME, LAM_INT = 300.0, 1.0
EVAL_SEASONS = (2024, 2025, 2026)
MODEL_CHOICES = ("ridge_tuned", "lightgbm", "extra_trees", "bilinear_emb")
DEFAULT_MODELS = ("ridge_tuned", "lightgbm", "bilinear_emb")


def design_with_intercept(frame, players):
    values = frame[PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    home = frame["home_poss"].to_numpy(dtype=bool)
    offense = np.where(home[:, None], values[:, 5:], values[:, :5])
    defense = np.where(home[:, None], values[:, :5], values[:, 5:])
    n_players = len(players)
    index_of = {int(pid): i for i, pid in enumerate(players)}
    n_rows = len(frame)

    def block(side, offset):
        flat = side.ravel()
        keep = flat != 0
        known = np.fromiter((int(p) in index_of for p in flat[keep]), dtype=bool, count=int(keep.sum()))
        rows = np.repeat(np.arange(n_rows), 5)[keep][known]
        cols = np.fromiter((index_of[int(p)] for p in flat[keep][known]),
                           dtype=np.int64, count=int(known.sum())) + offset
        return rows, cols, np.ones(int(known.sum()), dtype=np.float64)

    o_r, o_c, o_v = block(offense, 0)
    d_r, d_c, d_v = block(defense, n_players)
    h_r = np.arange(n_rows)
    X = csr_matrix(
        (
            np.concatenate([o_v, d_v, np.where(home, 1.0, -1.0), np.ones(n_rows)]),
            (
                np.concatenate([o_r, d_r, h_r, np.arange(n_rows)]),
                np.concatenate([o_c, d_c, np.full(n_rows, 2 * n_players), np.full(n_rows, 2 * n_players + 1)]),
            ),
        ),
        shape=(n_rows, 2 * n_players + 2),
    )
    y = frame["pts"].to_numpy(dtype=np.float64)
    pos_map = index_of
    off_pos = np.vectorize(lambda v: pos_map.get(int(v), -1), otypes=[np.int64])(offense)
    def_pos = np.vectorize(lambda v: pos_map.get(int(v), -1), otypes=[np.int64])(defense)
    meta = {"game": frame["gameid"].to_numpy(), "home": home,
            "off_idx": off_pos, "def_idx": def_pos,
            "off_mask": (off_pos >= 0).astype(np.float32),
            "def_mask": (def_pos >= 0).astype(np.float32)}
    return X, y, meta


class SuffStats:
    def __init__(self, X, y):
        self.XtX = (X.T @ X).tocsc()
        self.Xty = X.T @ y

    def beta(self, lam_off, lam_def, x0=None):
        n_cols = self.XtX.shape[0]
        Np = (n_cols - 2) // 2
        D = diags(np.concatenate([
            np.full(Np, lam_off), np.full(Np, lam_def), [LAM_HOME], [LAM_INT]]))
        A = (self.XtX + D).tocsr()
        b_vec = self.Xty.copy()
        M = diags(1.0 / np.maximum(A.diagonal(), 1e-9))
        x = np.zeros(n_cols) if x0 is None else x0.copy()
        r = b_vec - A @ x
        z = M @ r
        pvec = z.copy()
        rz = float(r @ z)
        for _ in range(4000):
            if rz < 1e-18:
                break
            Ap = A @ pvec
            alpha = rz / float(pvec @ Ap)
            x += alpha * pvec
            r -= alpha * Ap
            if np.linalg.norm(r) < 1e-8 * np.linalg.norm(b_vec):
                break
            z_new = M @ r
            pvec = z_new + (float(z_new @ r) / rz) * pvec
            rz = float(z_new @ r)
        return x


class BilinearRAPM:
    """Additive off/def scalars plus low-rank off x def interaction terms."""

    def __init__(self, n_players, dim=8, seed=7):
        rng = np.random.default_rng(seed)
        self.a = np.zeros(n_players)
        self.b = np.zeros(n_players)
        self.E = rng.normal(0, 0.02, size=(n_players, dim))
        self.F = rng.normal(0, 0.02, size=(n_players, dim))
        self.c = 1.1
        self.g = 0.0
        self.dim = dim

    def _pred_parts(self, oi, di, mo, md, home):
        ao = self.a[oi] * mo
        bd = self.b[di] * md
        Esum = (self.E[oi] * mo[..., None]).sum(axis=1)
        Fsum = (self.F[di] * md[..., None]).sum(axis=1)
        raw = ao.sum(axis=1) - bd.sum(axis=1) + (Esum * Fsum).sum(axis=1) \
            + self.c + self.g * home.astype(np.float64)
        return raw, Esum, Fsum

    def fit(self, oi, di, mo, md, home, y, epochs=8, batch=65536, lr=3e-3, seed=11):
        rng = np.random.default_rng(seed)
        n = len(y)
        params = {"a": self.a, "b": self.b, "E": self.E, "F": self.F,
                  "c": np.array([self.c]), "g": np.array([self.g])}
        state = {k: np.zeros_like(v) for k, v in params.items()}
        for epoch in range(epochs):
            order = rng.permutation(n)
            losses = []
            for start in range(0, n, batch):
                sel = order[start:start + batch]
                raw, Esum, Fsum = self._pred_parts(
                    oi[sel], di[sel], mo[sel], md[sel], home[sel])
                err = 2.0 * (raw - y[sel])
                losses.append(float(np.mean(err ** 2)) / 4.0)
                ga = np.zeros_like(self.a); gb = np.zeros_like(self.b)
                gE = np.zeros_like(self.E); gF = np.zeros_like(self.F)
                e_w = err[:, None, None] * mo[sel][:, :, None]
                d_w = err[:, None, None] * md[sel][:, :, None]
                np.add.at(ga, oi[sel].ravel(), e_w.reshape(-1))
                np.add.at(gb, di[sel].ravel(), (-d_w).reshape(-1))
                np.add.at(gE, oi[sel].ravel(),
                          (e_w * Fsum[:, None, :]).reshape(-1, self.dim))
                np.add.at(gF, di[sel].ravel(),
                          (d_w * Esum[:, None, :]).reshape(-1, self.dim))
                grads = {"a": ga, "b": gb, "E": gE, "F": gF,
                         "c": np.array([err.sum()]),
                         "g": np.array([(err * home[sel].astype(np.float64)).sum()])}
                for key in params:
                    state[key] = 0.9 * state[key] + 0.1 * (grads[key] ** 2)
                    params[key] -= lr * grads[key] / (np.sqrt(state[key]) + 1e-12)
            print(f"    bilinear epoch {epoch + 1}: mse={np.mean(losses):.4f}", flush=True)

    def predict(self, oi, di, mo, md, home):
        raw, *_ = self._pred_parts(oi, di, mo, md, home)
        return raw


def game_eval(pred_row, meta, frame):
    signed = pred_row * np.where(meta["home"], 1.0, -1.0)
    p = pd.DataFrame({"g": meta["game"], "s": signed}).groupby("g")["s"].mean() * 100.0
    al = pd.concat([p.rename("p"), actual_margins(frame).rename("a")], axis=1).dropna()
    if len(al) < 10:
        return len(al), float("nan"), float("nan")
    return len(al), float(np.corrcoef(al.p, al.a)[0, 1]), float((al.p - al.a).abs().mean())


def load_fold_design(held):
    """Build one forward fold without retaining any other fold in memory."""
    train_frames = [load_cache(list(range(1997, 2024)))]
    train_frames.extend(
        load_canonical((season,))
        for season in EVAL_SEASONS
        if season < held
    )
    if len(train_frames) == 1:
        train = train_frames[0]
    else:
        train = pd.concat(train_frames, ignore_index=True)
    del train_frames

    test = load_canonical((held,))
    players = player_universe(train, test)
    X_train, y_train, train_meta = design_with_intercept(train, players)
    X_test, _, test_meta = design_with_intercept(test, players)
    print(
        f"  fold end={held}: train {X_train.shape[0]:,} / "
        f"test {X_test.shape[0]:,} / players {len(players)}",
        flush=True,
    )
    return train, test, X_train, y_train, train_meta, X_test, test_meta, players


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=EVAL_SEASONS,
        default=list(EVAL_SEASONS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--lambdas",
        nargs=2,
        type=float,
        metavar=("OFF", "DEF"),
        help="Use frozen development-selected penalties instead of rerunning the grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "outputs" / "bakeoff_last3.csv",
    )
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    wanted = sorted(set(args.folds))
    chosen_models = tuple(dict.fromkeys(args.models))
    print(
        "CONTRACT: 2024-2026 development/selection; "
        "2027 untouched confirmation.",
        flush=True,
    )

    lightgbm = None
    if "lightgbm" in chosen_models:
        import lightgbm as lightgbm

    extra_trees_class = None
    if "extra_trees" in chosen_models:
        from sklearn.ensemble import ExtraTreesRegressor
        extra_trees_class = ExtraTreesRegressor

    out_csv = args.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done_pairs = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        done_pairs = {
            (row.model, int(row.fold))
            for row in pd.read_csv(out_csv).itertuples()
        }

    def record(model, held, games, correlation, mae):
        row = {
            "model": model,
            "fold": held,
            "games": games,
            "corr": correlation,
            "mae": mae,
        }
        header = not out_csv.exists() or out_csv.stat().st_size == 0
        pd.DataFrame([row]).to_csv(
            out_csv,
            mode="a",
            index=False,
            header=header,
        )
        done_pairs.add((model, held))
        print(
            f"  {model:22s} end={held}: corr={correlation:.3f} "
            f"mae={mae:.2f} games={games}  [saved]",
            flush=True,
        )

    if args.lambdas is None:
        print("\n== PART A: lambda grid, sequential folds ==", flush=True)
        results = {}
        for held in wanted:
            (
                train,
                test,
                X_train,
                y_train,
                train_meta,
                X_test,
                test_meta,
                players,
            ) = load_fold_design(held)
            stats = SuffStats(X_train, y_train)
            fold_scores = []
            for lambda_off in GRID_OFF:
                for lambda_def in GRID_DEF:
                    beta = stats.beta(lambda_off, lambda_def)
                    games, correlation, mae = game_eval(
                        X_test @ beta,
                        test_meta,
                        test,
                    )
                    fold_scores.append(
                        ((lambda_off, lambda_def), correlation, mae)
                    )
            results[held] = fold_scores
            top = sorted(
                fold_scores,
                key=lambda item: -(
                    item[1] if item[1] == item[1] else -9
                ),
            )[:3]
            for config, correlation, mae in top:
                print(
                    f"  fold end={held}: lam_off={config[0]:.0f} "
                    f"lam_def={config[1]:.0f} corr={correlation:.3f} "
                    f"mae={mae:.2f}",
                    flush=True,
                )
            del (
                train,
                test,
                X_train,
                y_train,
                train_meta,
                X_test,
                test_meta,
                players,
                stats,
                beta,
            )
            gc.collect()

        lookups = {
            held: {
                config: correlation
                for config, correlation, _ in results[held]
            }
            for held in results
        }
        mean_surface = {}
        for lambda_off in GRID_OFF:
            for lambda_def in GRID_DEF:
                scores = [
                    lookups[held].get(
                        (lambda_off, lambda_def),
                        float("nan"),
                    )
                    for held in wanted
                ]
                valid = [score for score in scores if score == score]
                if valid:
                    mean_surface[(lambda_off, lambda_def)] = float(
                        np.mean(valid)
                    )
        best_config = max(mean_surface, key=mean_surface.get)
        print(
            f"\nDEV-TUNED optimum: lam_off={best_config[0]:.0f} "
            f"lam_def={best_config[1]:.0f} "
            f"mean_corr={mean_surface[best_config]:.4f}",
            flush=True,
        )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            matrix = np.full((len(GRID_DEF), len(GRID_OFF)), np.nan)
            for (lambda_off, lambda_def), value in mean_surface.items():
                matrix[
                    GRID_DEF.index(lambda_def),
                    GRID_OFF.index(lambda_off),
                ] = value
            figure, axis = plt.subplots(figsize=(8, 6))
            image = axis.imshow(matrix, cmap="viridis", aspect="auto")
            axis.set_xticks(
                range(len(GRID_OFF)),
                [f"{value:.0f}" for value in GRID_OFF],
            )
            axis.set_yticks(
                range(len(GRID_DEF)),
                [f"{value:.0f}" for value in GRID_DEF],
            )
            axis.set_xlabel("lambda OFF")
            axis.set_ylabel("lambda DEF")
            plt.colorbar(image, label="forward mean corr")
            axis.add_patch(
                plt.Rectangle(
                    (
                        GRID_OFF.index(best_config[0]) - 0.5,
                        GRID_DEF.index(best_config[1]) - 0.5,
                    ),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
            axis.set_title(
                "lambda grid - forward mean corr (dev-tuned only)"
            )
            figure.tight_layout()
            figure.savefig(
                LAB_ROOT / "outputs" / "lambda_grid_last3.png",
                dpi=150,
            )
            plt.close(figure)
            print("saved heatmap", flush=True)
        except Exception as error:  # noqa: BLE001
            print("heatmap skipped:", error)
    else:
        best_config = tuple(args.lambdas)
        print(
            "\nUsing frozen development-selected penalties: "
            f"lam_off={best_config[0]:.0f} "
            f"lam_def={best_config[1]:.0f}",
            flush=True,
        )

    print("\n== PART B: linear vs nonlinear, sequential folds ==", flush=True)
    for held in wanted:
        pending_models = [
            model
            for model in chosen_models
            if (model, held) not in done_pairs
        ]
        if not pending_models:
            print(
                f"  fold end={held}: all requested models checkpointed",
                flush=True,
            )
            continue

        (
            train,
            test,
            X_train,
            y_train,
            train_meta,
            X_test,
            test_meta,
            players,
        ) = load_fold_design(held)
        n_players = len(players)

        for model_name in chosen_models:
            if (model_name, held) in done_pairs:
                print(
                    f"  {model_name:22s} end={held}: checkpointed",
                    flush=True,
                )
                continue

            if model_name == "ridge_tuned":
                stats = SuffStats(X_train, y_train)
                beta = stats.beta(best_config[0], best_config[1])
                prediction = X_test @ beta
                games, correlation, mae = game_eval(
                    prediction,
                    test_meta,
                    test,
                )
                record(
                    model_name,
                    held,
                    games,
                    correlation,
                    mae,
                )
                del stats, beta, prediction

            elif model_name == "lightgbm":
                assert lightgbm is not None
                X_train_float = X_train.astype(np.float32)
                X_test_float = X_test.astype(np.float32)
                model = lightgbm.LGBMRegressor(
                    n_estimators=(200 if args.fast else 350),
                    learning_rate=0.06,
                    num_leaves=(64 if args.fast else 96),
                    min_child_samples=300,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    n_jobs=8,
                    verbose=-1,
                )
                model.fit(X_train_float, y_train)
                prediction = model.predict(X_test_float)
                games, correlation, mae = game_eval(
                    prediction,
                    test_meta,
                    test,
                )
                record(
                    model_name,
                    held,
                    games,
                    correlation,
                    mae,
                )
                del model, X_train_float, X_test_float, prediction

            elif model_name == "extra_trees":
                assert extra_trees_class is not None
                X_train_float = X_train.astype(np.float32)
                X_test_float = X_test.astype(np.float32)
                model = extra_trees_class(
                    n_estimators=(80 if args.fast else 120),
                    min_samples_leaf=(800 if args.fast else 400),
                    max_features=0.5,
                    n_jobs=8,
                    random_state=7,
                )
                model.fit(X_train_float, y_train)
                prediction = model.predict(X_test_float)
                games, correlation, mae = game_eval(
                    prediction,
                    test_meta,
                    test,
                )
                record(
                    model_name,
                    held,
                    games,
                    correlation,
                    mae,
                )
                del model, X_train_float, X_test_float, prediction

            else:
                model = BilinearRAPM(n_players)
                model.fit(
                    train_meta["off_idx"],
                    train_meta["def_idx"],
                    train_meta["off_mask"],
                    train_meta["def_mask"],
                    train_meta["home"],
                    y_train,
                    epochs=(5 if args.fast else 8),
                    batch=65536,
                )
                prediction = model.predict(
                    test_meta["off_idx"],
                    test_meta["def_idx"],
                    test_meta["off_mask"],
                    test_meta["def_mask"],
                    test_meta["home"],
                )
                games, correlation, mae = game_eval(
                    prediction,
                    test_meta,
                    test,
                )
                record(
                    model_name,
                    held,
                    games,
                    correlation,
                    mae,
                )
                del model, prediction

            gc.collect()

        del (
            train,
            test,
            X_train,
            y_train,
            train_meta,
            X_test,
            test_meta,
            players,
        )
        gc.collect()

    if out_csv.exists() and out_csv.stat().st_size > 0:
        table = pd.read_csv(out_csv)
        summary = (
            table.groupby("model")
            .agg(
                folds=("fold", "nunique"),
                mean_corr=("corr", "mean"),
                mean_mae=("mae", "mean"),
            )
            .sort_values("mean_corr", ascending=False)
        )
        print("\nBAKE-OFF SUMMARY (development/selection only):")
        print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    print("REQUESTED WORK COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
