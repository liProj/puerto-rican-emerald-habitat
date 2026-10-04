"""Self-test for the significance machinery: Holm correction and the Nadeau-Bengio test.
Run: ./.venv/bin/python Birds/code/test_stats.py"""
import sys; sys.path.insert(0, "Birds/code")
import numpy as np
from scipy import stats
from stats_tests import holm, nadeau_bengio

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    ok = ok and cond

print("Holm step-down:")
p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
h = holm(p)
# textbook: sorted p_(i) * (m-i+1), then enforced monotone non-decreasing
expect = np.maximum.accumulate([0.01*5, 0.02*4, 0.03*3, 0.04*2, 0.05*1])
check("matches the textbook definition", np.allclose(h, expect), f"{np.round(h,4)}")
check("monotone non-decreasing in p", np.all(np.diff(h[np.argsort(p)]) >= -1e-12))
check("never smaller than the raw p-value", np.all(h >= p - 1e-12))
check("never exceeds 1", np.all(h <= 1.0))
h1 = holm(np.array([0.004]))
check("single test is unchanged", np.isclose(h1[0], 0.004))
hn = holm(np.array([0.01, np.nan, 0.03]))
check("NaNs pass through", np.isnan(hn[1]) and np.isfinite(hn[0]) and np.isfinite(hn[2]))

print("\nNadeau-Bengio corrected resampled t-test:")
rng = np.random.default_rng(0)
d = rng.normal(0.02, 0.05, 30)
t_nb, p_nb = nadeau_bengio(d, n_train=60000, n_test=6800)
t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
p_naive = 2 * stats.t.sf(abs(t_naive), df=len(d) - 1)
check("is more conservative than the naive paired t-test", p_nb > p_naive,
      f"p_nb={p_nb:.4g} vs p_naive={p_naive:.4g}")
check("variance inflation factor is 1/n + n_test/n_train",
      np.isclose(t_nb, d.mean() / np.sqrt(d.var(ddof=1) * (1/30 + 6800/60000)), rtol=1e-9))
t0, p0 = nadeau_bengio(np.zeros(10), 1000, 100)
check("a zero difference gives no signal", np.isnan(t0) or abs(t0) < 1e-9)
t1, p1 = nadeau_bengio([0.1], 1000, 100)
check("fewer than two folds returns NaN", np.isnan(t1))
# a large, consistent difference must be detected
t2, p2 = nadeau_bengio(rng.normal(0.30, 0.02, 30), 60000, 6800)
check("detects a large consistent effect", p2 < 0.001, f"p={p2:.3g}")
# a null difference must not be detected, on average
false_pos = sum(nadeau_bengio(rng.normal(0, 0.05, 30), 60000, 6800)[1] < 0.05 for _ in range(400))
check("false-positive rate under the null is at or below nominal",
      false_pos / 400 <= 0.06, f"{false_pos}/400 = {false_pos/400:.3f}")

print("\nALL PASS" if ok else "\nFAILURES PRESENT"); sys.exit(0 if ok else 1)
