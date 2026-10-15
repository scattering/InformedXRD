#!/usr/bin/env python3
"""
Decisive IT-track test: how much of the localization gain (oracle vs global) survives when
family positions come from a REAL peak finder instead of the oracle geometric positions.

Design (isolates DETECTION limit, holds indexing perfect):
  * Oracle reader: 61-pt raw window at the geometric family position (perfect localization
    + soft/learned detection on the raw window).
  * Global reader: PCA of the whole pattern (no localization).            -> +20 gap is localization.
  * Peak-finder reader (this script): we still use the geometric position as the SEARCH
    ANCHOR (= "indexing is perfect"), but the family's window is centered on the nearest
    PEAK ACTUALLY DETECTED within +-R of the anchor by the adaptive wavelet finder; if the
    finder detects nothing there (a real miss -> 41% weak recall), the window is zeros.
    So this = perfect localization + HARD detection (a real finder's threshold).

The oracle->peakfinder drop = information the finder's hard threshold discards =
how much of the localization gain is reader-recoverable vs lost to intrinsic detectability.
We also run a compact [found, |dpos|, snr] feature reader, and both LogReg and RandomForest.

Usage: run_peakfinder_reader.py <h5> <npz> [indexing_dir]
"""
import os, sys, numpy as np, h5py
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "Information Theory")
from bayes_ceiling import fit_channel, bayes_topk_exact            # noqa: E402
from analyze_gate2_two_stage_decomposition import (                 # noqa: E402
    topk_from_symbols, accuracy_from_topk, decode_symbol, encode_symbol,
    N_FAMILIES, FULL_SYMBOL_ALPHABET)

H, NPZ = sys.argv[1], sys.argv[2]
IDX_DIR = sys.argv[3] if len(sys.argv) > 3 else "/Users/UserE/project-indexing/indexing"
os.environ.setdefault("PYWAVES_DIR", "/tmp/PyWaveletPeakID")
os.environ.setdefault("WAVELETDATABASEPATH", "/tmp/PyWaveletPeakID/waveletdatabase.npz")
sys.path.insert(0, "/tmp/PyWaveletPeakID")
sys.path.insert(0, IDX_DIR)
from gregoire_adaptive import AdaptiveGregoire                      # noqa: E402

HALF = 30                          # oracle window half-width (61-pt)
SEARCH_R_DEG = 0.30                # search radius around the geometric anchor (= oracle window half-width)
K_SNR = 3.0

with h5py.File(H) as f:
    X = np.asarray(f["X_test"], np.float32)
    OF = np.asarray(f["oracle_features"], np.float32)
    tt = np.asarray(f["two_theta"], np.float64)
step = float(tt[1] - tt[0])
d = np.load(NPZ, allow_pickle=True)
labels = d["labels"].astype(int)
sym_true = d["symbols_ideal_mask"].astype(int)
true_states = np.stack([decode_symbol(int(s)) for s in sym_true], 0)
nclasses = int(labels.max()) + 1
W, prior = fit_channel(labels, sym_true, nclasses, FULL_SYMBOL_ALPHABET, 1.0 / FULL_SYMBOL_ALPHABET)
ceil = bayes_topk_exact(W, prior, ks=(1, 5))
N = len(labels)

# ---- recover geometric center index per (pattern, family) from the stored windows ----
# the oracle window is an exact 61-pt slice of X_test (zeros if center None / off-grid).
def recover_centers():
    centers = np.full((N, N_FAMILIES), -1, int)
    for i in range(N):
        sw = sliding_window_view(X[i], 2 * HALF + 1)        # (n-60, 61)
        for fam in range(N_FAMILIES):
            win = OF[i, fam]
            if not np.any(win):                              # all-zero -> no position
                continue
            dist = ((sw - win) ** 2).sum(1)
            p = int(np.argmin(dist))                         # window covers X[p:p+61]; center = p+HALF
            centers[i, fam] = p + HALF
    return centers

centers = recover_centers()
print(f"recovered geom centers: {(centers >= 0).sum()}/{centers.size} families positioned", flush=True)

# ---- run the adaptive peak finder on each pattern (same X_test rows) ----
def find_all():
    det = AdaptiveGregoire(step_deg=step)
    out = []
    for i in range(N):
        pos, snr, _ = det.detect(tt, X[i], k_snr=K_SNR, return_arrays=True)
        out.append((np.asarray(pos, float), np.asarray(snr, float)))
        if (i + 1) % 100 == 0:
            print(f"  finder {i+1}/{N}", flush=True)
    return out

dets = find_all()

# ---- build peak-finder features ----
# realistic window: centered at nearest detected peak within +-R of the geometric anchor; zeros if miss.
# compact: [found, signed delta(deg), snr] per family.
def build_peak_features():
    pk_win = np.zeros((N, N_FAMILIES, 2 * HALF + 1), np.float32)
    pk_cmp = np.zeros((N, N_FAMILIES, 3), np.float32)
    found_ct = miss_ct = 0
    for i in range(N):
        pos, snr = dets[i]
        for fam in range(N_FAMILIES):
            c = centers[i, fam]
            if c < 0:
                continue
            anchor = tt[c]
            if len(pos):
                dd = np.abs(pos - anchor)
                j = int(np.argmin(dd))
            else:
                dd = None; j = -1
            if j >= 0 and dd[j] <= SEARCH_R_DEG:               # detected near the anchor
                found_ct += 1
                didx = int(round((pos[j] - tt[0]) / step))
                lo, hi = didx - HALF, didx + HALF + 1
                a, b = max(0, lo), min(len(X[i]), hi)
                if a < b:
                    pk_win[i, fam, a - lo:b - lo] = X[i, a:b]
                pk_cmp[i, fam] = [1.0, float(anchor - pos[j]), float(snr[j])]
            else:                                              # real miss -> absent feature
                miss_ct += 1
                pk_cmp[i, fam] = [0.0, 0.0, 0.0]
    tot = found_ct + miss_ct
    print(f"peak-finder at anchors: found {found_ct}/{tot} ({100*found_ct/tot:.1f}%), miss {miss_ct}", flush=True)
    return pk_win, pk_cmp

pk_win, pk_cmp = build_peak_features()

# ---- readers ----
def run(feat_fn, clf_factory, n_splits=5, seed=123):
    rec = np.zeros_like(true_states)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Ftr, Fte = feat_fn(tr, te)
        for j in range(N_FAMILIES):
            ytr = true_states[tr, j]
            if len(np.unique(ytr)) < 2:                        # degenerate family in this fold
                rec[te, j] = ytr[0] if len(ytr) else 0
                continue
            clf = clf_factory(); clf.fit(Ftr, ytr); rec[te, j] = clf.predict(Fte)
    tk = topk_from_symbols(encode_symbol(rec), W, prior, (1, 5))
    return accuracy_from_topk(tk[1], labels) * 100, accuracy_from_topk(tk[5], labels) * 100

def f_global(tr, te):
    sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
    p = PCA(n_components=min(64, len(tr) - 1), random_state=123); return p.fit_transform(Xtr), p.transform(Xte)
def f_oracle(tr, te):
    sc = StandardScaler(); return sc.fit_transform(OF[tr].reshape(len(tr), -1)), sc.transform(OF[te].reshape(len(te), -1))
def f_pkwin(tr, te):
    sc = StandardScaler(); return sc.fit_transform(pk_win[tr].reshape(len(tr), -1)), sc.transform(pk_win[te].reshape(len(te), -1))
def f_pkcmp(tr, te):
    sc = StandardScaler(); return sc.fit_transform(pk_cmp[tr].reshape(len(tr), -1)), sc.transform(pk_cmp[te].reshape(len(te), -1))

LOG = lambda: LogisticRegression(max_iter=4000)
RF = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)

print(f"\nCEILING (true absences): {ceil['top1_bayes_exact']*100:.1f} / {ceil['top5_bayes_exact']*100:.1f}")
for clfname, clf in [("LogReg", LOG), ("RandomForest", RF)]:
    g1, g5 = run(f_global, clf); o1, o5 = run(f_oracle, clf)
    w1, w5 = run(f_pkwin, clf); c1, c5 = run(f_pkcmp, clf)
    surv = (w1 - g1) / (o1 - g1) * 100 if (o1 - g1) != 0 else float("nan")
    survc = (c1 - g1) / (o1 - g1) * 100 if (o1 - g1) != 0 else float("nan")
    print(f"\n==== {clfname} ====")
    print(f"  global (no localization):        {g1:5.1f} / {g5:5.1f}")
    print(f"  ORACLE (geom window):            {o1:5.1f} / {o5:5.1f}   [+{o1-g1:.1f} localization]")
    print(f"  PEAK-FINDER window (realistic):  {w1:5.1f} / {w5:5.1f}   [+{w1-g1:.1f} -> {surv:.0f}% of localization survives]")
    print(f"  PEAK-FINDER compact [found,dpos,snr]: {c1:5.1f} / {c5:5.1f}   [+{c1-g1:.1f} -> {survc:.0f}%]")
