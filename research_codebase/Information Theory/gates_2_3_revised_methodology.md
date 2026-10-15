# Revised Methodology for Gates 2 and 3
### What to do differently, and why — informed by prior information-theory-and-scattering work

*Self-contained design note. Captures the corrections arrived at after the
initial Gate 2/3 pilots and after reviewing the prior literature on information
theory in scattering. Someone picking this up cold should be able to act on it
without the back-history.*

---

## 0. Context in one paragraph

We treat extinction-group (EG) inference as decoding through a noisy physical
channel: the transmitted variable is the EG (one of 99 classes), the received
variable is an absence mask Mhat over a small set of diagnostic reflection
families (each `present` / `absent` / `unresolved`), and the channel p(Mhat|G) is
defined by the measurement physics. **Gate 2** estimates that channel for one or
two conditions and asks whether it carries real EG information (it does: ~2.8–3
bits, robustly). **Gate 3** is the *degradation ladder*: applying physical
corruptions (structure factor → form-factor/thermal → broadening/overlap →
background/noise → preferred orientation → impurity) and measuring how many EG
bits each one costs. The metric throughout is mutual information in bits.

## 1. What the prior work changes

Four imports from the literature reshape the method (full citations in the
related-work note):

- **Information gain = ΔH (Treece et al. 2019; Heinrich et al. 2020).** Our
  `bits_over_prior` is exactly their reflectometry information gain,
  H(prior) − H(posterior), applied to a *categorical* label instead of continuous
  parameters. This gives the metric a precedent — but the categorical setting
  introduces finite-sample MI bias that the Gaussian-posterior reflectometry case
  never faces, which is the source of most of our corrections below.
- **Analytic, sampling-free information (Durant et al. 2021).** Their Fisher
  information is computed from the model and the counts, never from sampling. The
  discrete analog: compute the channel semi-analytically (Wilson statistics) for
  the rungs that admit it, rather than by Monte Carlo. Their "maximum information
  gain" is the continuous analog of our Blahut–Arimoto channel capacity.
- **Marginalization onto a subset (Heinrich et al. 2020).** They marginalize
  information gain onto chosen structural parameters. The analog for Gate 3 is to
  attribute each rung's bit loss to the specific diagnostic families it destroys.
- **Overlap as a nuisance with a prior (Sivia & David 1994).** They handle peak
  overlap in powder data softly rather than by hard assignment. The analog: make
  the present/absent/unresolved call a *probability*, not a hard threshold.

One conceptual boundary from the MaxEnt-crystallography line (Bricogne; Gilmore;
Sakata/Takata): that work uses entropy as a *reconstruction prior*, a different
role from our *channel-information* measure. Do not conflate the two when citing.

## 2. Gate 2 — channel estimation: what to do differently

**Problem we hit.** Raw plug-in MI / capacity on a 99 × ~600 table from 2
samples/class with per-cell smoothing α = 0.5 returned ~0 bits — pure estimator
artifact (the smoothing prior dominated the counts). The channel was fine; the
estimate was starved.

Do this instead:

1. **Never report raw plug-in MI or capacity from a starved table.** Report the
   permutation-**debiased** MI: `MI_debiased = MI_plugin − mean(MI under label
   shuffle)`, which is 0 in expectation under independence by construction. Carry
   both a permutation null (significance) and a bootstrap CI (sampling spread).
   The shuffle null *is* the finite-sample bias floor — watch it, not the raw
   number.
2. **Use the null as the stopping rule, not a fixed N.** Trustworthy ≈ null below
   ~25% of the debiased signal. Empirically that needs **~100 samples/EG minimum,
   ~300/EG for a defensible capacity**; 50/EG is a ballpark only. The exact figure
   scales inversely with the true MI, which is why the measured null — not a fixed
   N — is the criterion.
3. **Generate once, subsample for the curve.** Generation is the cost (and it is
   superlinear for high-symmetry cubic groups due to rejection). Generate the
   largest batch you can afford *once*, save the structures, and subsample to
   build the null-vs-N curve. Do not run separate s10/s50/s100 jobs.
4. **Set smoothing α ≈ 1/n_symbols** (total pseudocount per row ≈ 1), not 0.5.
5. **Collapse the alphabet as a companion.** Drop `unresolved` → present/absent
   (2^F symbols) for a lower-variance estimate; report full and collapsed and
   confirm they agree on the story.
6. **Capacity (Blahut–Arimoto) only after the null is small,** and validate it
   against a *shuffled-capacity* null (capacity on label-permuted data should
   collapse toward 0). BA maximizes over the prior, so it exploits sparse-table
   noise more aggressively than MI does — it is the last quantity to trust, not
   the first.
7. **Compute the analytic rungs instead of sampling them (Durant philosophy).**
   The *ideal* mask is a deterministic function of G, so I(G; M_ideal) =
   H(M_ideal) exactly, with no simulation. The structure-factor and
   form-factor/thermal rungs follow from Wilson statistics: an allowed family
   reads `present` only if its intensity clears threshold, with probability
   exp(−t/⟨I⟩) (acentric) or erfc(√(t/2⟨I⟩)) (centric), ⟨I⟩ set by the
   form-factor and Debye–Waller envelope. This gives W(Mhat|G) in closed form —
   bias-free, and free of the chemistry-sampling caveat that conditions any MC
   estimate. **Use Monte Carlo to validate this on a few EGs, not as the primary
   estimator for these rungs.**
8. **Soft present/absent (Sivia & David).** Replace the hard intensity threshold
   with a per-family probability of presence (from Wilson + noise). This removes
   the threshold as a free parameter and preserves the information sitting right
   at the detection boundary.
9. **Run the exact injectivity check first (free).** With binary present/absent
   over F families there are only 2^F ideal patterns. For F = 6 that is 64 < 99,
   so by pigeonhole **some EGs are forced to collide** — unseparable by the mask,
   regardless of physics. Run `ideal_channel_analysis` on the real EG→family
   absence table to see *which* EGs collide. This is a mask-design ceiling you
   read off the table, and it anchors the top of the Gate 3 waterfall.

## 3. Gate 3 — degradation ladder: what to do differently

**Problem we hit.** Per-rung MI came out bunched in a ~0.06-bit band (2.78–2.84)
and **non-monotone**, with `ideal_mask` sitting *below* the corrupted rungs. Under
a genuine data-processing chain the ideal rung must be the supremum, so this is a
tell: the rungs were built as *parallel* forward constructions from the same
structure (each with its own alphabet and bias floor), not as a nested chain. The
ordering tracked per-rung alphabet/bias, not degradation.

Do this instead:

1. **Make each rung a literal downstream transform of the previous rung's
   realized PATTERN — not a separate simulation, and not a transform of the
   encoded mask symbol.** Realize the clean pattern once per structure; then
   form-factor multiplies the realized intensities by the q-dependent/Debye–Waller
   envelope, broadening convolves that result, background adds to it, and so on.
   Extract the mask only at the end of each stage. This makes
   G → X₀ → X₁ → … → X_K a Markov chain, so the data-processing inequality
   **guarantees** the MI is monotone non-increasing and forces `ideal_mask` to the
   top. (Degradation at the *pattern* level is faithful physics; degradation of
   the lossy mask symbol is not.)
2. **Anchor the top rung at the exact ideal-mask channel** (§2.9), so the
   waterfall descends from H(distinct ideal patterns), a sampling-free supremum.
3. **Use the paired estimator for per-rung losses.** With all rungs computed on
   the *same* structures, bootstrap with shared structure indices so the per-rung
   difference cancels structure-sampling variance. A loss CI that straddles zero
   means that corruption is below resolution — an honest result to report, not a
   failure. (Without pairing, a few-thousandths-of-a-bit loss drowns; with it, the
   CI is ~5× tighter.)
4. **Honor the two DPI exceptions.**
   - *Impurity-interference* injects exogenous peaks at forbidden positions that
     are **not** derivable from the clean pattern, so it is not a downstream
     transform and can *raise* apparent information. Flag it as a **non-DPI step**;
     do not force it into the monotone curve. (If it is currently folded into a
     combined `rruff_like` rung, that rung is not DPI-protected.)
   - *Preferred orientation* is a valid transform **only if per-reflection hkl is
     retained**, so the March–Dollase factor can multiply the realized intensities.
5. **Attribute loss per family (Heinrich's marginalization).** Do not stop at
   total bits-lost per rung. Recompute the MI with each diagnostic family held at
   its clean value (or individually degraded) to get a per-family marginal loss.
   This turns "preferred orientation costs X bits" into "preferred orientation
   costs X bits, concentrated in the serial-reflection families that distinguish
   screw axes" — the mechanistic claim, and a falsifiable prediction: PO is a
   *structured* erasure, so the EGs that lose most should be exactly those whose
   diagnostic absences sit in the suppressed families.
6. **Referee-safe checks before Gate 3 enters any response letter:**
   `ideal_mask` is the maximum of the curve; every reported per-rung loss has a
   paired CI on the correct (non-negative) side; impurity is reported separately
   from the DPI chain.

## 4. Cross-cutting estimator discipline (applies to both gates)

- **Floor vs ceiling.** Real-data model performance gives **model-extracted bits**
  (a lower bound, the ΔH information gain of a specific decoder). Simulation gives
  **simulator-available bits** / capacity (a ceiling). Keep them separate; never
  read model accuracy as a physics limit. This floor/ceiling separation is the
  part the prior reflectometry work does not do for a categorical label, and it is
  where our contribution sits.
- **Apparent zero ≠ real zero.** A measured ~0 (Gate 1 Top-1 ≈ prior; Gate 2
  capacity ≈ 0) can be a real null or an estimation artifact. Distinguish with a
  shuffle null and an N-trend before interpreting.
- **Sub-resolution gap ≠ ordering claim.** When a difference (clean vs degraded,
  or rung-to-rung) is smaller than the differential bias between conditions, the
  raw and debiased estimates can disagree on sign. That is the signal to report a
  bound, not an ordering.
- **Always report against the appropriate null,** and prefer analytic/exact
  quantities (ideal channel, Wilson channel) over sampled ones where they exist;
  use Monte Carlo to validate, not as the default estimator.

## 5. One-line summary

Gate 2 result (channel carries ~2.8–3 bits of EG information) stands and does not
depend on any of the above. What changes is **how we estimate and present it**:
debiased MI with nulls and adequate N for Gate 2; a literal realized-pattern
Markov-chain ladder with per-family loss attribution for Gate 3; analytic channels
where available; and a strict floor/ceiling/exception discipline throughout.
