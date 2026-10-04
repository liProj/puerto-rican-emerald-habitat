"""HERON: a stacked, effort-disentangled ensemble, distilled into a single student.

The diagnostic runs showed a clean split of competence under spatial blocking:
  * the factorised neural branch and the published GLM rank the *positives* best
    (highest average precision) but are weaker on global ordering;
  * gradient-boosted trees given the SRTM terrain block have the best AUC but worse
    average precision, because terrain lets them separate regions rather than records.
Neither family dominates. HERON therefore combines them with a meta-learner fitted on
INNER SPATIALLY BLOCKED folds of the training region, so the combination weights are
estimated under the same kind of geographic shift the outer test imposes. The stack is
then distilled into one compact network so the deployed model is a single object that
inherits the terrain signal through soft labels rather than by fitting terrain directly
(which overfits it).
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "Birds/code")
import data_prep as dp, heron as H, baselines as B
from calib import platt_fit, platt_apply
from sklearn.linear_model import LogisticRegression

EPS = 1e-7
def _logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))

def inner_blocks(lon, lat, n_inner=3, block_deg=0.08, seed=0):
    """Spatially blocked partition of the training region, same construction as the outer folds."""
    rng = np.random.default_rng(1000 + seed)
    ox, oy = rng.uniform(0, block_deg, 2)
    key = (np.floor((lon + ox) / block_deg).astype(int).astype(str) + "_" +
           np.floor((lat + oy) / block_deg).astype(int).astype(str))
    u = pd.unique(key); rng.shuffle(u)
    size = pd.Series(key).value_counts()
    load = np.zeros(n_inner); assign = {}
    for b in sorted(u, key=lambda z: -size[z]):
        j = int(np.argmin(load)); assign[b] = j; load[j] += size[b]
    return pd.Series(key).map(assign).values

class HeronStack:
    """Base learners -> inner-spatial-CV meta-learner -> Platt calibration -> optional student."""
    def __init__(self, n_inner=3, nn_ens=2, epochs=350, batch=32768, seed=0,
                 use_nn=True, use_trees=True, use_glm=True, distil=False,
                 nn_terrain=False, nn_rff=False, heron_kw=None, verbose=False):
        self.__dict__.update(locals()); del self.self
        self.heron_kw = heron_kw or dict(gamma=0.0, pos_weight=1.0, lam_group=0.1, lam_vrex=1.0)

    # ---- base learners -------------------------------------------------
    def _psi(self, t, seed=0):
        r = dp.RFF(seed=0)(t) if self.nn_rff else None
        return dp.psi_features(t, r, self.nn_terrain)

    def _nn(self, dtr, dev, blk_tr, seed):
        Xp, Xd, Xe = self._psi(dtr), dp.det_features(dtr), dp.effort_raw(dtr)
        g = pd.factorize(dtr.site.astype(str) + "|" + dtr.Period.astype(str))[0]
        ps = []
        for s in range(self.nn_ens):
            _, pred = H.fit_heron(Xp, Xd, Xe, dtr.y.values, g, blk_tr, seed=seed * 131 + s,
                                  epochs=self.epochs, batch=self.batch, **self.heron_kw)
            ps.append(pred(self._psi(dev), dp.det_features(dev), dp.effort_raw(dev)))
        return np.mean(ps, 0)

    def _bases(self, dtr, dev, blk_tr, seed, nn_src=None, nn_blk=None):
        out = {}
        if self.use_nn:
            out["nn"] = self._nn(nn_src if nn_src is not None else dtr, dev,
                                 nn_blk if nn_blk is not None else blk_tr, seed)
        if self.use_trees:
            out["lgb"] = B.BASELINES["LightGBM + terrain"](dtr, dev, seed=seed)[0]
            out["xgb"] = B.BASELINES["XGBoost + terrain"](dtr, dev, seed=seed)[0]
        if self.use_glm:   out["glm"] = B.BASELINES["GLM-m4 (published)"](dtr, dev, seed=seed)[0]
        return out

    # ---- fit / predict -------------------------------------------------
    def fit(self, dtr, blk_tr, dtr_nn=None, blk_nn=None):
        """dtr: complete-case training frame for the trees/GLM and the meta-learner.
        dtr_nn: optional larger frame (all checklists, missing NDVI included) for the neural branch.

        Base learners are fitted ONCE on the whole training region and kept. The meta-learner and
        the calibrator are both estimated from inner out-of-fold predictions, never from
        predictions a base learner made on its own training rows.
        """
        inner = inner_blocks(dtr.Longitude.values, dtr.Latitude.values, self.n_inner, seed=self.seed)
        self.inner_ = inner
        inn_nn = (inner_blocks(dtr_nn.Longitude.values, dtr_nn.Latitude.values,
                               self.n_inner, seed=self.seed) if dtr_nn is not None else None)
        oof, self.keys_ = None, None
        for j in range(self.n_inner):
            m = inner != j
            nm = (inn_nn != j) if inn_nn is not None else None
            P = self._bases(dtr[m], dtr, blk_tr[m], self.seed + j,
                            nn_src=(dtr_nn[nm] if nm is not None else None),
                            nn_blk=(blk_nn[nm] if nm is not None else None))
            if self.keys_ is None:
                self.keys_ = sorted(P); oof = {k: np.zeros(len(dtr)) for k in self.keys_}
            for k in self.keys_: oof[k][inner == j] = P[k][inner == j]
        self.oof_ = oof
        Z = np.column_stack([_logit(oof[k]) for k in self.keys_])
        self.meta_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(Z, dtr.y.values)
        p_oof = self.meta_.predict_proba(Z)[:, 1]
        self.cal_ = platt_fit(p_oof, dtr.y.values)      # calibrated on honest out-of-fold scores
        self._fit_full(dtr, blk_tr, dtr_nn, blk_nn)
        return self

    def _fit_full(self, dtr, blk_tr, dtr_nn, blk_nn):
        """Final base learners, fitted once on the whole training region."""
        self._tr = dtr
        self.full_nn_ = []
        if self.use_nn:
            src, bsrc = (dtr_nn, blk_nn) if dtr_nn is not None else (dtr, blk_tr)
            Xp, Xd, Xe = self._psi(src), dp.det_features(src), dp.effort_raw(src)
            g = pd.factorize(src.site.astype(str) + "|" + src.Period.astype(str))[0]
            for s_ in range(getattr(self, "nn_ens_full", self.nn_ens)):
                self.full_nn_.append(
                    H.fit_heron(Xp, Xd, Xe, src.y.values, g, bsrc, seed=self.seed * 131 + 900 + s_,
                                epochs=self.epochs, batch=self.batch, **self.heron_kw)[1])

    def _full_bases(self, dev):
        out = {}
        if self.use_nn:
            Xp, Xd, Xe = self._psi(dev), dp.det_features(dev), dp.effort_raw(dev)
            out["nn"] = np.mean([pr(Xp, Xd, Xe) for pr in self.full_nn_], 0)
        if self.use_trees:
            out["lgb"] = B.BASELINES["LightGBM + terrain"](self._tr, dev, seed=self.seed)[0]
            out["xgb"] = B.BASELINES["XGBoost + terrain"](self._tr, dev, seed=self.seed)[0]
        if self.use_glm:
            out["glm"] = B.BASELINES["GLM-m4 (published)"](self._tr, dev, seed=self.seed)[0]
        return out

    def predict(self, dev, calibrate=True):
        P = self._full_bases(dev)
        Z = np.column_stack([_logit(P[k]) for k in self.keys_])
        p = self.meta_.predict_proba(Z)[:, 1]
        self.last_bases_ = P
        if calibrate: p = platt_apply(p, self.cal_)
        return np.clip(p, EPS, 1 - EPS)

    def distil(self, dpool, blk_pool, epochs=None, seed=0):
        """Train one compact factorised network to imitate the stack over a large unlabelled pool.
        Soft targets transfer the trees' terrain signal without the student fitting terrain."""
        soft = self.predict(dpool, calibrate=True)
        Xp, Xd, Xe = dp.psi_features(dpool, None, False), dp.det_features(dpool), dp.effort_raw(dpool)
        g = pd.factorize(dpool.site.astype(str) + "|" + dpool.Period.astype(str))[0]
        _, pred = H.fit_heron(Xp, Xd, Xe, soft.astype(np.float32), g, blk_pool, seed=seed,
                              epochs=epochs or self.epochs, batch=self.batch,
                              gamma=0.0, pos_weight=1.0, lam_group=0.0, lam_vrex=0.5)
        self.student_ = pred
        return pred
