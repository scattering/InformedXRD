#!/usr/bin/env python3
"""Clean localization isolation: LogReg on oracle local windows vs global PCA."""
import sys, numpy as np, h5py
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "Information Theory")
from bayes_ceiling import fit_channel, bayes_topk_exact
from analyze_gate2_two_stage_decomposition import (topk_from_symbols, accuracy_from_topk,
    decode_symbol, encode_symbol, N_FAMILIES, FULL_SYMBOL_ALPHABET)

H, NPZ = sys.argv[1], sys.argv[2]
with h5py.File(H) as f:
    X = np.asarray(f["X_test"], np.float32); OF = np.asarray(f["oracle_features"], np.float32)
d = np.load(NPZ, allow_pickle=True)
labels = d["labels"].astype(int)
sym_true = d["symbols_ideal_mask"].astype(int)
true_states = np.stack([decode_symbol(int(s)) for s in sym_true], 0)
nclasses = int(labels.max()) + 1
W, prior = fit_channel(labels, sym_true, nclasses, FULL_SYMBOL_ALPHABET, 1.0 / FULL_SYMBOL_ALPHABET)
ceil = bayes_topk_exact(W, prior, ks=(1, 5))

def run(feat_fn, n_splits=5, seed=123):
    rec = np.zeros_like(true_states)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Ftr, Fte = feat_fn(tr, te)
        for j in range(N_FAMILIES):
            clf = LogisticRegression(max_iter=4000); clf.fit(Ftr, true_states[tr, j]); rec[te, j] = clf.predict(Fte)
    tk = topk_from_symbols(encode_symbol(rec), W, prior, (1, 5))
    return accuracy_from_topk(tk[1], labels) * 100, accuracy_from_topk(tk[5], labels) * 100

def gfeat(tr, te):
    sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
    p = PCA(n_components=min(64, len(tr) - 1), random_state=123); return p.fit_transform(Xtr), p.transform(Xte)
def ofeat(tr, te):
    sc = StandardScaler(); return sc.fit_transform(OF[tr].reshape(len(tr), -1)), sc.transform(OF[te].reshape(len(te), -1))

g1, g5 = run(gfeat); o1, o5 = run(ofeat)
tk = topk_from_symbols(d["symbols_background_noise_like"].astype(int), W, prior, (1, 5))
th1, th5 = accuracy_from_topk(tk[1], labels) * 100, accuracy_from_topk(tk[5], labels) * 100
print(f"CEILING (true absences):                  {ceil['top1_bayes_exact']*100:5.1f} / {ceil['top5_bayes_exact']*100:5.1f}")
print(f"ORACLE-POSITION LogReg (local windows):   {o1:5.1f} / {o5:5.1f}")
print(f"GLOBAL-PCA LogReg (no positions):         {g1:5.1f} / {g5:5.1f}")
print(f"[ctx] threshold reader (oracle+free abs): {th1:5.1f} / {th5:5.1f}")
print(f"\n>>> CLEAN LOCALIZATION VALUE (oracle - global): Top-1 {o1-g1:+.1f},  Top-5 {o5-g5:+.1f}")
