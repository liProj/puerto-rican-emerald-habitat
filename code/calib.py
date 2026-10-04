"""Probability calibration that does not disturb the ranking.

Isotonic regression is the usual choice, but it is piecewise constant: it collapses many
distinct scores onto the same value, creating ties that measurably damage AUC and average
precision. Platt scaling is strictly increasing, so every ranking metric is left exactly
unchanged while Brier, log loss and ECE improve.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

def platt_fit(p, y):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:                      # degenerate fit would invert the ranking; fall back to identity
        a, b = 1.0, 0.0
    return a, b

def platt_apply(p, ab):
    a, b = ab
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    return np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), 1e-7, 1 - 1e-7)
