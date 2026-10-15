"""
How many samples per EG do we need? Simulate a channel matched to the real
Gate-2 one (99 classes, ~600-symbol skewed mask alphabet, H(Mhat)~3.9,
imbalanced prior) with a KNOWN true MI, then sweep samples/class and report:
  - true MI (exact, from the channel matrix)
  - debiased MI estimate (plugin - shuffle null)
  - shuffle-null mean (= the finite-sample bias floor)
  - trust ratio = null / debiased  (want << 1)
for both the full 3^6 alphabet and the 2^6 collapse.
"""
import numpy as np
from discrete_channel import mutual_information_bits
from mi_bias_corrected import mi_debiased

rng = np.random.default_rng(0)


def skewed(n, target_H, tries=40):
    """A skewed distribution over n items with entropy ~ target_H bits."""
    best = None
    for _ in range(tries):
        a = rng.random() * 1.5 + 0.05          # power-law-ish exponent search
        p = 1.0 / (np.arange(1, n + 1) ** a)
        p /= p.sum()
        H = -np.sum(p * np.log2(p))
        if best is None or abs(H - target_H) < abs(best[1] - target_H):
            best = (p, H)
    return best[0]


def build_channel(n_classes, n_symbols, prior_H, mask_H, eps):
    """W[g,m] = (1-eps)*[m==sig(g)] + eps*base[m]; returns prior, W, true MI."""
    prior = skewed(n_classes, prior_H)
    base = skewed(n_symbols, mask_H)
    sig = rng.choice(n_symbols, size=n_classes, p=base)   # common syms common sigs
    W = eps * np.tile(base, (n_classes, 1))
    W[np.arange(n_classes), sig] += (1 - eps)
    W /= W.sum(1, keepdims=True)
    return prior, W, mutual_information_bits(W, prior)


def sample(prior, W, per_class):
    n_classes = len(prior)
    labels = np.repeat(np.arange(n_classes), per_class)
    symbols = np.array([rng.choice(W.shape[1], p=W[g]) for g in labels])
    return labels, symbols


def collapse_to_2state_6fam(symbols):
    """3^6 symbol -> 2^6 by merging unresolved(2)/absent(1) -> 1, present(0)->0."""
    out = np.zeros_like(symbols)
    for j in range(6):
        digit = (symbols // (3 ** j)) % 3
        out += np.where(digit == 0, 0, 1) * (2 ** j)
    return out, 64


def run(label, n_symbols, prior_H, mask_H, eps, collapse=False):
    prior, W, true_mi = build_channel(99, n_symbols, prior_H, mask_H, eps)
    print(f"\n### {label}  (true MI = {true_mi:.3f} bits, "
          f"alphabet={n_symbols if not collapse else '3^6->2^6=64'}) ###")
    print(f"{'per_class':>9} {'N':>7} {'debiased':>9} {'null':>7} "
          f"{'trust(null/sig)':>16}")
    for pc in (10, 50, 100, 300, 1000):
        lab, sym = sample(prior, W, pc)
        if collapse:
            sym, nsym = collapse_to_2state_6fam(sym)
        else:
            nsym = n_symbols
        d = mi_debiased(lab, sym, 99, nsym, n_perm=80, n_boot=0)
        sig = max(d["mi_debiased"], 1e-6)
        ratio = d["mi_null_mean"] / sig
        flag = "  <- trustworthy" if (d["mi_null_mean"] < 0.3 * true_mi
                                      and ratio < 0.5) else ""
        print(f"{pc:>9} {len(lab):>7} {d['mi_debiased']:>9.3f} "
              f"{d['mi_null_mean']:>7.3f} {ratio:>16.2f}{flag}")


if __name__ == "__main__":
    # eps tuned so true MI lands ~2 bits (a plausible mid-range for a corrupted
    # absence channel) on the full 3^6 alphabet.
    print("=" * 70)
    print("FULL 3^6 alphabet (n_symbols=729 nominal, ~600 realized)")
    run("structure-factor-like, full alphabet", 600, 4.7, 3.9, eps=0.55)

    print("\n" + "=" * 70)
    print("COLLAPSED 2^6=64 alphabet (drop 'unresolved')")
    # build on full 3^6 then collapse, so the collapse logic is exercised
    run("structure-factor-like, collapsed", 729, 4.7, 3.9, eps=0.55,
        collapse=True)
