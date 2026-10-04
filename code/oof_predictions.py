"""Pooled out-of-fold predictions for the headline methods on one spatially blocked repeat.
Feeds the ROC / precision-recall and calibration figures."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd
import data_prep as dp, baselines as B, run_main as RM

OUT = "Birds/results/oof_spatial_NDVI.csv"
METHODS = ["GLM-m4 (published)", "GLM-splines", "LightGBM + terrain", "XGBoost + terrain",
           "HERON-NN", "HERON"]

def main(rep=0, n_folds=10, only=None, merge=True):
    """only: restrict to these method names and merge them into an existing OOF file."""
    b = RM.Bench("NDVI")
    f  = dp.spatial_folds(b.d, n_folds, 0.08, seed=rep)
    fa = dp.spatial_folds(b.a, n_folds, 0.08, seed=rep)
    f, fa = f[0], fa[0]
    meths = [m for m in METHODS if only is None or m in only]
    P = {m: np.full(len(b.d), np.nan) for m in meths}
    for k in range(n_folds):
        tr, te = f != k, f == k
        for m in meths:
            spec = RM.MAIN[m]
            try:
                if spec[0] == "baseline":
                    p, _ = B.BASELINES[spec[1]](b.d[tr], b.d[te], seed=rep)
                elif spec[0] == "stack":
                    p = b.stack(tr, k, f, fa, spec[1], seed=rep)[te]
                else:
                    p = b.heron(tr, k, f, fa, spec[1], seed=rep)[te]
                P[m][te] = p
            except Exception as e:
                print(f"  !! {m} fold{k}: {type(e).__name__}: {e}", flush=True)
        print(f"[oof] fold {k} done", flush=True)
    if merge and os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        for m, v in P.items(): prev[m] = v
        out = prev
    else:
        out = pd.DataFrame({"y": b.d.y.values, "fold": f, **P})
    out.to_csv(OUT, index=False)
    print("wrote", OUT, out.shape)

if __name__ == "__main__":
    only = sys.argv[1].split("|") if len(sys.argv) > 1 else None
    main(only=only)
