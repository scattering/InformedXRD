#!/usr/bin/env python3
"""
Sweep an impurity-like absence-corrupting nuisance over the realized Gate 3
rule-mask and six-family channels.

This uses the completed ladder artifact directly:
  * six-family rows are decoded from `six_symbols_*`
  * rule-mask rows are reconstructed from `rule_symbols_* + rule_vocab_*`

For each impurity level, it computes:
  * exact Bayes Top-k ceiling
  * held-out Top-k ceiling
  * exact-decoder error signature (cross-bin, descendant/ancestor/branch, and
    crystal-system-change rates)

The impurity operator is intentionally simple and explicit:
  * six-family: `absent -> present` on family states
  * rule-mask: `absent -> present` on per-reflection rule states
with either uniform or inverse-norm (low-order stressed) weighting.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, deque
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bayes_ceiling import bayes_topk_exact, bayes_topk_heldout, fit_channel


N_CLASSES = 99

# six-family encoding comes from discrete_channel: present=0, absent=1, unresolved=2
SIX_PRESENT, SIX_ABSENT, SIX_UNRESOLVED = 0, 1, 2

# rule-mask ladder encoding comes from gate3_rule_mask_ladder: absent=0, present=1, unresolved=2
RULE_ABSENT, RULE_PRESENT, RULE_UNRESOLVED = 0, 1, 2

SIX_REFLS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

RULE_BASE = [
    (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0),
    (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 0), (4, 2, 0), (3, 3, 0), (4, 4, 0),
    (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1),
]


def _perm_closure(base):
    out = set()
    for hkl in base:
        for perm in itertools.permutations(range(3)):
            out.add((hkl[perm[0]], hkl[perm[1]], hkl[perm[2]]))
    return sorted(out)


RULE_REFLS = _perm_closure(RULE_BASE)

# Physical-family category codes using the *correct* ladder semantics
S_UNDET, S_NONE, S2, S3, S4, S6, S_OTHER = range(7)
G_UNDET, G_NONE, G_N, G_NONE_OR_N, G_AXIAL, G_D, G_PRESENT_UNDET = range(7)
C_UNDET, C_P, C_I, C_F, C_C, C_OTHER = range(6)


def parse_args():
    parser = argparse.ArgumentParser(description="Impurity sweep over realized six-family and rule-mask channels.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--rung", default="background_noise_like")
    parser.add_argument("--all-rungs", action="store_true")
    parser.add_argument("--rates", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--weighting", choices=("uniform", "inverse_norm"), default="inverse_norm")
    parser.add_argument("--alpha", type=float, default=0.0, help="Dirichlet smoothing; <=0 uses 1/n_symbols.")
    parser.add_argument("--heldout-boot", type=int, default=200)
    parser.add_argument("--heldout-test-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canonical-signatures-csv", default="/Users/UserE/project/Information Theory/EG questions/canonical_eg_signatures.csv")
    parser.add_argument("--graph-json", default="/Users/UserE/project/artifacts/topology_graphs/extinction_group_adjacency.json")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_root = THIS_DIR.parent
    alt = repo_root / path
    if alt.exists():
        return alt
    if "Information Theory/EG questions/canonical_eg_signatures.csv" in path:
        alt = repo_root / "Information Theory" / "EG questions" / "canonical_eg_signatures.csv"
        if alt.exists():
            return alt
    if "artifacts/topology_graphs/extinction_group_adjacency.json" in path:
        alt = repo_root / "artifacts" / "topology_graphs" / "extinction_group_adjacency.json"
        if alt.exists():
            return alt
    return candidate


def parse_rates(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def decode_base3_symbols(symbols: np.ndarray, n_features: int) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int64)
    out = np.zeros((len(symbols), n_features), dtype=np.int8)
    vals = symbols.copy()
    for j in range(n_features - 1, -1, -1):
        out[:, j] = vals % 3
        vals //= 3
    return out


def reconstruct_rule_rows(data, rung: str) -> np.ndarray:
    vocab = np.asarray(data[f"rule_vocab_{rung}"], dtype=object)
    sym = np.asarray(data[f"rule_symbols_{rung}"], dtype=np.int64)
    return np.asarray([np.asarray(vocab[int(i)], dtype=np.int8) for i in sym], dtype=np.int8)


def load_crossbin_map(path: str) -> dict[int, str]:
    out = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[int(row["eg"])] = row["canonical_signature_hash"]
    return out


def build_graph(graph_json_path: str):
    with open(graph_json_path, "r") as handle:
        data = json.load(handle)

    nodes = {str(key): value for key, value in data["nodes"].items()}
    directed = {str(key): [str(child) for child in value] for key, value in data["adjacency"].items()}
    undirected = {node: set() for node in nodes}
    for parent, children in directed.items():
        for child in children:
            undirected.setdefault(parent, set()).add(child)
            undirected.setdefault(child, set()).add(parent)

    label_to_node = {}
    for node, attrs in nodes.items():
        for raw_eg in attrs.get("merged_from", []):
            label_to_node[int(raw_eg) + 1] = node
    return nodes, directed, undirected, label_to_node


def shortest_distance(undirected, src, dst):
    if src == dst:
        return 0
    queue = deque([(src, 0)])
    seen = {src}
    while queue:
        node, dist = queue.popleft()
        for nxt in undirected.get(node, ()):
            if nxt == dst:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def has_directed_path(directed, src, dst):
    if src == dst:
        return True
    queue = deque([src])
    seen = {src}
    while queue:
        node = queue.popleft()
        for nxt in directed.get(node, ()):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def weights_for_reflections(reflections: list[tuple[int, int, int]], mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(reflections), dtype=np.float64)
    vals = np.asarray([1.0 / np.linalg.norm(hkl) for hkl in reflections], dtype=np.float64)
    vals /= vals.mean()
    return vals


def apply_impurity_six(rows: np.ndarray, rate: float, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(rows, dtype=np.int8).copy()
    p = np.clip(rate * weights[None, :], 0.0, 1.0)
    flip = (out == SIX_ABSENT) & (rng.random(out.shape) < p)
    out[flip] = SIX_PRESENT
    return out


def apply_impurity_rule(rows: np.ndarray, rate: float, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(rows, dtype=np.int8).copy()
    p = np.clip(rate * weights[None, :], 0.0, 1.0)
    flip = (out == RULE_ABSENT) & (rng.random(out.shape) < p)
    out[flip] = RULE_PRESENT
    return out


def rule_states_from_row(row):
    return {hkl: int(c) for hkl, c in zip(RULE_REFLS, np.asarray(row).tolist())}


def serial_category(states, axis: int) -> int:
    def st(n: int) -> int:
        h = [0, 0, 0]
        h[axis] = n
        return states.get(tuple(h), RULE_UNRESOLVED)

    s1 = st(1)
    if s1 == RULE_UNRESOLVED:
        return S_UNDET
    if s1 == RULE_PRESENT:
        return S_NONE
    for n, code in ((2, S2), (3, S3), (4, S4), (6, S6)):
        if st(n) == RULE_PRESENT:
            return code
    return S_OTHER


def _consensus_rule(states, hkls):
    vals = [states[h] for h in hkls if h in states]
    if not vals:
        return RULE_UNRESOLVED
    if RULE_PRESENT in vals:
        return RULE_PRESENT
    if RULE_ABSENT in vals:
        return RULE_ABSENT
    return RULE_UNRESOLVED


def glide_category(states, zero_axis: int) -> int:
    ax = [a for a in range(3) if a != zero_axis]

    def ip(u, v):
        h = [0, 0, 0]
        h[ax[0]] = u
        h[ax[1]] = v
        return tuple(h)

    s11 = _consensus_rule(states, [ip(1, 1)])
    s21 = _consensus_rule(states, [ip(2, 1), ip(1, 2)])
    if s11 == RULE_UNRESOLVED:
        return G_UNDET
    if s11 == RULE_PRESENT:
        if s21 == RULE_UNRESOLVED:
            return G_NONE_OR_N
        return G_NONE if s21 == RULE_PRESENT else G_N
    if s21 == RULE_UNRESOLVED:
        return G_PRESENT_UNDET
    return G_AXIAL if s21 == RULE_PRESENT else G_D


def centering_category(states) -> int:
    s111 = _consensus_rule(states, [(1, 1, 1)])
    s211 = _consensus_rule(states, [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
    s221 = _consensus_rule(states, [(2, 2, 1), (2, 1, 2), (1, 2, 2)])
    if s111 == RULE_UNRESOLVED or s211 == RULE_UNRESOLVED:
        return C_UNDET
    if (s111, s211) == (RULE_PRESENT, RULE_PRESENT):
        return C_P
    if s111 == RULE_ABSENT and s211 == RULE_PRESENT:
        return C_I
    if s111 == RULE_PRESENT and s211 == RULE_ABSENT:
        if s221 == RULE_ABSENT:
            return C_F
        if s221 == RULE_PRESENT:
            return C_C
        return C_OTHER
    return C_OTHER


def physical_rule_signature(row: np.ndarray) -> tuple[int, ...]:
    states = rule_states_from_row(row)
    serials = tuple(sorted(serial_category(states, a) for a in range(3)))
    glides = tuple(sorted(glide_category(states, a) for a in range(3)))
    return serials + glides + (centering_category(states),)


def ids_from_signatures(signatures: list[tuple[int, ...]]) -> np.ndarray:
    vocab = {}
    out = np.empty(len(signatures), dtype=np.int64)
    for i, sig in enumerate(signatures):
        if sig not in vocab:
            vocab[sig] = len(vocab)
        out[i] = vocab[sig]
    return out


def exact_top1_predictions(labels: np.ndarray, symbols: np.ndarray, n_classes: int, alpha: float):
    n_symbols = int(symbols.max()) + 1 if len(symbols) else 1
    alpha_eff = float(alpha) if alpha > 0 else 1.0 / max(n_symbols, 1)
    W, prior = fit_channel(labels, symbols, n_classes=n_classes, n_symbols=n_symbols, alpha=alpha_eff)
    joint = prior[:, None] * W
    post_by_symbol = joint.T
    preds = np.argmax(post_by_symbol, axis=1)
    return preds[symbols], bayes_topk_exact(W, prior)


def error_signature(labels: np.ndarray, preds: np.ndarray, crossbin_hash: dict[int, str], directed, undirected, label_to_node):
    directionality = Counter()
    wrong_distances = []
    cross_bin_wrong = 0
    system_change_wrong = 0

    for truth0, pred0 in zip(labels, preds):
        true_eg = int(truth0) + 1
        pred_eg = int(pred0) + 1
        if true_eg == pred_eg:
            directionality["correct"] += 1
            continue

        true_hash = crossbin_hash.get(true_eg)
        pred_hash = crossbin_hash.get(pred_eg)
        if true_hash != pred_hash:
            cross_bin_wrong += 1

        true_node = label_to_node.get(true_eg)
        pred_node = label_to_node.get(pred_eg)
        if true_node is not None and pred_node is not None and true_node != pred_node:
            true_system = true_node.split("_", 1)[0]
            pred_system = pred_node.split("_", 1)[0]
            if true_system != pred_system:
                system_change_wrong += 1
            dist = shortest_distance(undirected, true_node, pred_node)
            wrong_distances.append("inf" if dist is None else dist)
            if has_directed_path(directed, true_node, pred_node):
                directionality["descendant_pred_lower_sym"] += 1
            elif has_directed_path(directed, pred_node, true_node):
                directionality["ancestor_pred_higher_sym"] += 1
            else:
                directionality["branch_jump"] += 1
        else:
            directionality["unmapped"] += 1

    wrong_n = len(labels) - int(directionality["correct"])
    counter = Counter(wrong_distances)
    finite = [v for v in wrong_distances if v != "inf"]
    return {
        "wrong_n": int(wrong_n),
        "cross_bin_wrong": int(cross_bin_wrong),
        "cross_bin_fraction_wrong": (float(cross_bin_wrong) / wrong_n) if wrong_n else 0.0,
        "crystal_system_change_fraction_wrong": (float(system_change_wrong) / wrong_n) if wrong_n else 0.0,
        "distance_buckets": {
            "1": int(counter.get(1, 0)),
            "2": int(counter.get(2, 0)),
            "3": int(counter.get(3, 0)),
            "4plus": int(sum(v for k, v in counter.items() if k not in (1, 2, 3, "inf"))),
            "inf": int(counter.get("inf", 0)),
        },
        "pct_wrong_le2": (100.0 * (counter.get(1, 0) + counter.get(2, 0)) / wrong_n) if wrong_n else None,
        "mean_wrong_dist": (sum(finite) / len(finite)) if finite else None,
        "directionality": {key: int(value) for key, value in directionality.items()},
    }


def summarize_channel(labels: np.ndarray, symbols: np.ndarray, alpha: float, heldout_boot: int, heldout_test_frac: float, seed: int):
    n_symbols = int(symbols.max()) + 1 if len(symbols) else 1
    alpha_eff = float(alpha) if alpha > 0 else 1.0 / max(n_symbols, 1)
    W, prior = fit_channel(labels, symbols, n_classes=N_CLASSES, n_symbols=n_symbols, alpha=alpha_eff)
    out = {
        "n_symbols": int(n_symbols),
        "alpha": float(alpha_eff),
    }
    out.update(bayes_topk_exact(W, prior))
    out["heldout"] = bayes_topk_heldout(
        labels,
        symbols,
        N_CLASSES,
        alpha=alpha_eff,
        test_frac=heldout_test_frac,
        n_boot=heldout_boot,
        seed=seed,
    )
    return out


def encode_base3_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    n_features = rows.shape[1]
    weights = np.asarray([3 ** p for p in range(n_features - 1, -1, -1)], dtype=np.int64)
    return np.asarray(rows @ weights, dtype=np.int64)


def sweep_one_rung(
    data,
    labels: np.ndarray,
    rung: str,
    rates: list[float],
    weighting: str,
    alpha: float,
    heldout_boot: int,
    heldout_test_frac: float,
    seed: int,
    crossbin_hash,
    directed,
    undirected,
    label_to_node,
):
    rng = np.random.default_rng(seed)
    six_rows = decode_base3_symbols(np.asarray(data[f"six_symbols_{rung}"], dtype=np.int64), n_features=6)
    rule_rows = reconstruct_rule_rows(data, rung)

    six_weights = weights_for_reflections(SIX_REFLS, weighting)
    rule_weights = weights_for_reflections(RULE_REFLS, weighting)

    out = {
        "rates": rates,
        "weighting": weighting,
        "n_samples": int(len(labels)),
        "channels": {
            "six_family": {},
            "rule_mask_v1": {},
        },
    }

    for rate in rates:
        six_corrupt = apply_impurity_six(six_rows, rate, six_weights, rng)
        rule_corrupt = apply_impurity_rule(rule_rows, rate, rule_weights, rng)

        six_symbols = encode_base3_rows(six_corrupt)
        rule_sigs = [physical_rule_signature(row) for row in rule_corrupt]
        rule_symbols = ids_from_signatures(rule_sigs)

        six_summary = summarize_channel(labels, six_symbols, alpha, heldout_boot, heldout_test_frac, seed)
        six_preds, _ = exact_top1_predictions(labels, six_symbols, N_CLASSES, alpha)
        six_summary["error_signature"] = error_signature(labels, six_preds, crossbin_hash, directed, undirected, label_to_node)

        rule_summary = summarize_channel(labels, rule_symbols, alpha, heldout_boot, heldout_test_frac, seed)
        rule_preds, _ = exact_top1_predictions(labels, rule_symbols, N_CLASSES, alpha)
        rule_summary["error_signature"] = error_signature(labels, rule_preds, crossbin_hash, directed, undirected, label_to_node)
        rule_summary["diagnostics"] = {
            "serial_undet_rate": float(np.mean([any(v == S_UNDET for v in sig[:3]) for sig in rule_sigs])),
            "glide_undet_rate": float(np.mean([any(v == G_UNDET for v in sig[3:6]) for sig in rule_sigs])),
            "centering_undet_rate": float(np.mean([sig[6] == C_UNDET for sig in rule_sigs])),
        }

        key = f"{rate:.4f}"
        out["channels"]["six_family"][key] = six_summary
        out["channels"]["rule_mask_v1"][key] = rule_summary
    return out


def main():
    args = parse_args()
    rates = parse_rates(args.rates)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.npz, allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=np.int64)

    canonical_csv = resolve_repo_path(args.canonical_signatures_csv)
    graph_json = resolve_repo_path(args.graph_json)
    crossbin_hash = load_crossbin_map(str(canonical_csv))
    _, directed, undirected, label_to_node = build_graph(str(graph_json))

    out = {
        "artifact": args.npz,
        "rates": rates,
        "weighting": args.weighting,
        "n_samples": int(len(labels)),
        "canonical_signatures_csv": str(canonical_csv),
        "graph_json": str(graph_json),
    }
    if args.all_rungs:
        rung_names = ["ideal_mask", "structure_factor_like", "broadening_overlap_like", "rruff_like", "background_noise_like"]
        out["rungs"] = {
            rung: sweep_one_rung(
                data,
                labels,
                rung,
                rates,
                args.weighting,
                args.alpha,
                args.heldout_boot,
                args.heldout_test_frac,
                args.seed,
                crossbin_hash,
                directed,
                undirected,
                label_to_node,
            )
            for rung in rung_names
        }
    else:
        out["rung"] = args.rung
        out.update(
            sweep_one_rung(
                data,
                labels,
                args.rung,
                rates,
                args.weighting,
                args.alpha,
                args.heldout_boot,
                args.heldout_test_frac,
                args.seed,
                crossbin_hash,
                directed,
                undirected,
                label_to_node,
            )
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
