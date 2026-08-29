"""
discrete_channel.py
===================

Simulator-conditioned discrete-channel analysis for extinction-group (EG)
inference from powder diffraction.

Transmitted variable : G  in {0, ..., n_classes-1}   (99 extinction groups)
Received  variable   : Mhat, an observed mask over diagnostic reflection
                       families, each family in {present, absent, unresolved}.

This module answers a *different* question from model accuracy: given an
explicitly specified absence-mask channel p(Mhat | G), how many EG bits can
that channel transmit, independent of any neural decoder?

    I(G; Mhat)  -- bits the channel carries under a given prior pi(G)
    H(G | Mhat) -- residual EG entropy after observing the mask
    2^H(G|Mhat) -- effective number of remaining candidates
    C = max_pi I_pi(G; Mhat)  -- channel capacity (Blahut-Arimoto)

Because there are 99 EGs, C <= log2(99) ~= 6.63 bits, by construction.

Everything is in BITS (log base 2). All probabilities are clipped before
log to stay numerically stable. Channels are estimated from simulation
count tables with Dirichlet (additive) smoothing.

Only dependency: numpy.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
MASK_STATES = ("present", "absent", "unresolved")  # 0, 1, 2
_STATE_TO_INT = {s: i for i, s in enumerate(MASK_STATES)}


# --------------------------------------------------------------------------- #
# Mask encoding
# --------------------------------------------------------------------------- #
def encode_mask_vector(mask_vec, n_states: int = 3) -> int:
    """
    Encode a per-family mask vector (e.g. ['present','absent','unresolved'])
    into a single integer symbol so the joint observation can be treated as
    one received letter. Alphabet size is n_states ** n_families.

    WARNING: this grows as 3**J. For more than ~6-8 diagnostic families either
    restrict to the top-J most diagnostic families, or estimate per-family
    channels and use chain-rule bounds instead of the full joint. The caller
    is responsible for keeping J small; we assert a sane bound here.
    """
    ints = [_STATE_TO_INT[m] if isinstance(m, str) else int(m) for m in mask_vec]
    assert all(0 <= v < n_states for v in ints), "mask state out of range"
    assert len(ints) <= 12, (
        f"{len(ints)} families -> {n_states**len(ints)} symbols; "
        "too large for a dense joint channel. Restrict families or use "
        "per-family chain-rule bounds."
    )
    sym = 0
    for v in ints:
        sym = sym * n_states + v
    return sym


# --------------------------------------------------------------------------- #
# Channel estimation
# --------------------------------------------------------------------------- #
def estimate_channel(labels, observations, n_classes, n_symbols,
                     dirichlet_alpha: float = 0.5):
    """
    Estimate p(Mhat | G) from simulation outcomes with Dirichlet smoothing.

    labels       : (N,) int array of G in [0, n_classes)
    observations : (N,) int array of received symbols in [0, n_symbols)
    returns      : W of shape (n_classes, n_symbols), rows sum to 1
                   (W[g, m] = p(Mhat = m | G = g))

    A warning string is returned for any class with zero raw counts (the row
    falls back to the smoothing prior, i.e. uniform over symbols).
    """
    labels = np.asarray(labels, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.int64)
    assert labels.shape == observations.shape, "labels/observations shape mismatch"
    assert labels.min() >= 0 and labels.max() < n_classes, "label out of range"
    assert observations.min() >= 0 and observations.max() < n_symbols, "symbol out of range"

    counts = np.zeros((n_classes, n_symbols), dtype=np.float64)
    np.add.at(counts, (labels, observations), 1.0)

    empty = np.where(counts.sum(axis=1) == 0)[0]
    warning = ""
    if empty.size:
        warning = (f"{empty.size} class(es) had zero counts; their rows use the "
                   f"Dirichlet({dirichlet_alpha}) prior (uniform over symbols).")

    counts += dirichlet_alpha
    W = counts / counts.sum(axis=1, keepdims=True)
    return W, warning


# --------------------------------------------------------------------------- #
# Information quantities (all in bits)
# --------------------------------------------------------------------------- #
def _safe_log2(x):
    return np.log2(np.clip(x, _EPS, None))


def entropy_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(-np.sum(p * _safe_log2(p)))


def mutual_information_bits(W, prior):
    """
    I(G; Mhat) in bits for channel W = p(Mhat|G) under input prior pi(G).

    Computed as sum_{g,m} pi(g) W(g,m) log2[ W(g,m) / p(m) ], which is the
    KL form and is provably >= 0 (no clipping artifacts can make it negative
    beyond ~1e-12, which we floor away).
    """
    W = np.asarray(W, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    joint = prior[:, None] * W                      # p(g, m)
    p_m = joint.sum(axis=0)                          # p(m)
    # term = p(g,m) * log2[ p(g,m) / (p(g) p(m)) ]; zero where p(g,m)=0,
    # which also neutralizes symbols/classes with zero marginal (0*log0 -> 0).
    denom = prior[:, None] * p_m[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(joint) - np.log2(denom)
        term = np.where(joint > _EPS, joint * log_ratio, 0.0)
    mi = float(np.sum(term))
    return max(mi, 0.0)


def conditional_entropy_bits(W, prior):
    """H(G | Mhat) = H(G) - I(G; Mhat)."""
    return entropy_bits(prior) - mutual_information_bits(W, prior)


def effective_candidates(W, prior):
    """2 ** H(G | Mhat): effective number of EGs still in play after the mask."""
    return float(2.0 ** conditional_entropy_bits(W, prior))


# --------------------------------------------------------------------------- #
# Blahut-Arimoto channel capacity
# --------------------------------------------------------------------------- #
def blahut_arimoto_capacity(W, max_iter: int = 5000, tol: float = 1e-10):
    """
    Channel capacity C = max_{pi} I_pi(G; Mhat), in bits, via Blahut-Arimoto.

    W : (n_classes, n_symbols) channel p(Mhat|G).
    Returns (C_bits, capacity_achieving_prior).

    This is the only quantity in the analysis that is a genuine *ceiling*:
    the maximum EG information the absence-mask channel can transmit under the
    best possible class prior. It is bounded above by log2(n_classes).
    """
    W = np.asarray(W, dtype=np.float64)
    n_classes = W.shape[0]
    r = np.full(n_classes, 1.0 / n_classes)         # input prior, uniform start

    C_prev = -np.inf
    for _ in range(max_iter):
        joint = r[:, None] * W
        p_m = joint.sum(axis=0)
        # q(g | m) = r(g) W(g,m) / p(m)
        q = (r[:, None] * W) / np.clip(p_m[None, :], _EPS, None)
        # log r'(g) = sum_m W(g,m) log2 q(g|m)   (then exponentiate + normalize)
        log_r_new = np.sum(W * _safe_log2(q), axis=1)
        log_r_new -= log_r_new.max()                # stabilize before exp2
        r = np.exp2(log_r_new)
        r /= r.sum()

        C = mutual_information_bits(W, r)
        if abs(C - C_prev) < tol:
            break
        C_prev = C
    return float(C), r


# --------------------------------------------------------------------------- #
# Reference channels (for tests and as analytic baselines)
# --------------------------------------------------------------------------- #
def erasure_channel(n_classes: int, erase_prob: float):
    """
    Pure erasure channel: each of n_classes symbols is transmitted perfectly
    with prob (1 - erase_prob) and mapped to a dedicated erasure symbol with
    prob erase_prob. Output alphabet has n_classes + 1 symbols (last = erase).

    Closed form under a uniform prior: I(G;Mhat) = (1 - erase_prob) * log2(n_classes).
    Useful as the canonical model of preferred-orientation fading, where whole
    reflection families are erased rather than corrupted by additive noise.
    """
    W = np.zeros((n_classes, n_classes + 1), dtype=np.float64)
    for g in range(n_classes):
        W[g, g] = 1.0 - erase_prob
        W[g, -1] = erase_prob
    return W


def full_summary(W, prior, run_capacity: bool = True):
    """Convenience: dict of the headline discrete-channel quantities."""
    out = {
        "H_prior_bits": entropy_bits(prior),
        "I_G_Mhat_bits": mutual_information_bits(W, prior),
        "H_G_given_Mhat_bits": conditional_entropy_bits(W, prior),
        "effective_candidates": effective_candidates(W, prior),
        "log2_n_classes": float(np.log2(W.shape[0])),
    }
    if run_capacity:
        C, pi_star = blahut_arimoto_capacity(W)
        out["capacity_bits"] = C
        out["capacity_achieving_prior"] = pi_star
    return out
