"""Evaluation metrics. Sixteen per fold, mirroring the reporting style of the reference paper
(AIC/pseudo-R2) while adding the discrimination, calibration and ranking metrics it lacks."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, f1_score, matthews_corrcoef, balanced_accuracy_score)
from scipy.stats import spearmanr

def ece(y, p, bins=15):
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.digitize(p, e[1:-1]); tot = 0.0
    for k in np.unique(b):
        m = b == k
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot

def all_metrics(y, p, base_rate=None):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    br = y.mean() if base_rate is None else base_rate
    ll = log_loss(y, p, labels=[0, 1])
    ll0 = log_loss(y, np.full_like(p, br), labels=[0, 1])
    thr = np.quantile(p, 1 - y.mean())            # flag as many as the true prevalence
    yh = (p >= thr).astype(int)
    k = max(1, int(0.10 * len(y)))
    top = np.argsort(-p)[:k]
    return dict(
        AUC=roc_auc_score(y, p),
        AP=average_precision_score(y, p),
        Brier=brier_score_loss(y, p),
        BrierSkill=1 - brier_score_loss(y, p) / max(1e-12, np.mean((y - br) ** 2)),
        LogLoss=ll,
        pseudoR2=1 - ll / ll0,                     # McFadden, the metric the paper reports
        TjurR2=p[y == 1].mean() - p[y == 0].mean(),
        ECE=ece(y, p),
        F1=f1_score(y, yh, zero_division=0),
        MCC=matthews_corrcoef(y, yh) if len(np.unique(yh)) > 1 else 0.0,
        BalAcc=balanced_accuracy_score(y, yh),
        Precision_at_prev=yh[y == 1].sum() / max(1, yh.sum()),
        Recall_at_prev=yh[y == 1].sum() / max(1, int(y.sum())),
        Recall_at_10pct=y[top].sum() / max(1, int(y.sum())),
        Lift_at_10pct=(y[top].mean() / max(1e-9, y.mean())),
        SpearmanRank=spearmanr(p, y).statistic,
    )

HIGHER_BETTER = {"AUC": 1, "AP": 1, "Brier": -1, "BrierSkill": 1, "LogLoss": -1, "pseudoR2": 1,
                 "TjurR2": 1, "ECE": -1, "F1": 1, "MCC": 1, "BalAcc": 1,
                 "Precision_at_prev": 1, "Recall_at_prev": 1, "Recall_at_10pct": 1,
                 "Lift_at_10pct": 1, "SpearmanRank": 1}
