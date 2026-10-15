"""Tests for discrete_channel.py -- covers spec requirements 5, 6, 7 plus
capacity sanity. Run: pytest test_discrete_channel.py -q
"""
import numpy as np
import pytest

from discrete_channel import (
    encode_mask_vector, estimate_channel, entropy_bits,
    mutual_information_bits, conditional_entropy_bits, effective_candidates,
    blahut_arimoto_capacity, erasure_channel,
)

N = 99


def _uniform(n):
    return np.full(n, 1.0 / n)


# --- spec test 5: MI is nonnegative ----------------------------------------
def test_mi_nonnegative():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(7), size=N)            # random channel, 7 symbols
    for prior in (_uniform(N), rng.dirichlet(np.ones(N))):
        assert mutual_information_bits(W, prior) >= 0.0


# --- spec test 6: MI <= H(G) -----------------------------------------------
def test_mi_le_entropy():
    rng = np.random.default_rng(1)
    W = rng.dirichlet(np.ones(5), size=N)
    prior = rng.dirichlet(np.ones(N))
    assert mutual_information_bits(W, prior) <= entropy_bits(prior) + 1e-9
    # and conditional entropy stays in [0, H(G)]
    Hc = conditional_entropy_bits(W, prior)
    assert -1e-9 <= Hc <= entropy_bits(prior) + 1e-9


# --- spec test 7: erasure monotonicity -------------------------------------
def test_erasure_monotonic():
    prior = _uniform(N)
    mis = [mutual_information_bits(erasure_channel(N, e), prior)
           for e in np.linspace(0.0, 1.0, 11)]
    diffs = np.diff(mis)
    assert np.all(diffs <= 1e-9), "MI must not increase with erasure prob"
    # closed form check: I = (1-e) log2(N) at e=0 and e=1
    assert mis[0] == pytest.approx(np.log2(N), abs=1e-9)
    assert mis[-1] == pytest.approx(0.0, abs=1e-9)


# --- capacity invariants ----------------------------------------------------
def test_capacity_bounded_by_log2_n():
    rng = np.random.default_rng(2)
    W = rng.dirichlet(np.ones(8), size=N)
    C, pi_star = blahut_arimoto_capacity(W)
    assert C <= np.log2(N) + 1e-6
    assert C >= mutual_information_bits(W, _uniform(N)) - 1e-6   # cap >= any prior
    assert pi_star == pytest.approx(pi_star.clip(0), abs=0) and \
        pi_star.sum() == pytest.approx(1.0)


def test_capacity_noiseless_identity():
    # A noiseless channel with K input symbols mapped 1:1 to K outputs has
    # capacity exactly log2(K).
    K = 16
    W = np.eye(K)
    C, _ = blahut_arimoto_capacity(W)
    assert C == pytest.approx(np.log2(K), abs=1e-6)


# --- estimation + encoding sanity ------------------------------------------
def test_estimate_channel_smoothing_warns_on_empty_class():
    labels = np.array([0, 0, 1, 1])
    obs = np.array([0, 1, 0, 1])
    W, warn = estimate_channel(labels, obs, n_classes=3, n_symbols=2,
                               dirichlet_alpha=0.5)
    assert W.shape == (3, 2)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert "zero counts" in warn                     # class 2 never observed


def test_encode_mask_vector_roundtrip_ordering():
    # distinct vectors must map to distinct symbols
    a = encode_mask_vector(["present", "absent", "unresolved"])
    b = encode_mask_vector(["absent", "present", "unresolved"])
    assert a != b
    assert encode_mask_vector([0, 0, 0]) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
