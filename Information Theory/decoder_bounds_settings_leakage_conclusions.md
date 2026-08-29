# Multi-Decoder Information Bounds, RRUFF Settings, and Leakage: Conclusions & Next Steps

*Synthesis of the working session. Covers (1) what the four decoders do and do not
let us say information-theoretically, (2) the valid instruments for the bits claim,
(3) the RRUFF non-standard-setting correction to the lattice-match path, and (4) the
status and scope of the MP-overlap (leakage) analysis. Written to be dropped into the
project notes and to seed the next round of runs.*

---

## 1. What the multiple decoders actually buy us

The decoders in play: three XRDBench models (CNN11, BiGRU, Transformer) and our
stage2c ViT — four architecturally independent decoders, all evaluated on RRUFF-473.

**The asymmetry that governs everything.** Decoders bound the recoverable information
(and the Bayes accuracy) only *from below*. The best of any set of decoders is still a
lower bound. No collection of decoders can bound the ceiling from above; that bound
comes only from the channel model — our exact discrete absence-channel Bayes
computation (Gate 2). So "more decoders" tightens the floor and characterizes the gap;
it does not and cannot raise the ceiling. The earlier framing ("say more about the
upper bound") was therefore aimed at something unavailable in principle; the genuine
yield is a tighter, better-trusted *floor* and a sharper picture of the decoder-vs-
channel gap.

**Result — small ensemble gain.** Simple probability-space ensembling of the four
decoders lifts Top-1 only modestly (~12% individual → 14.16–14.38% ensemble) and leaves
Top-5 essentially flat (49.89–50.95%). Note this is straightforward ensemble combination,
not a proof of optimal decoder fusion; we have not shown that the best possible fusion
of these decoders stays on the floor, only that simple combination does. A small gain
across architecturally diverse decoders is positive evidence that they are *redundant* —
they fail on the same examples — which is the signature of a shared (channel) limit
rather than idiosyncratic decoder weakness. This is a stronger statement than "external
models fall in a floor band," because it shows that *combining* diverse decoders (at
least by simple ensembling) does not escape the floor.

**RETIRED: the near-saturation reading.** An earlier draft suggested the ensemble
Top-5 (~50%) might be near the channel ceiling. That was a cross-prior artifact and is
now explicitly retired. Putting floor and ceiling on the same prior — and, crucially,
the same surface — kills it (job `732593`, against the CFML `background_noise_like`
benchmark from `3173569`): see §1b.

## 1b. The corrected same-surface picture: the gap is large and it is *not* in classification

The decisive control runs the discrete absence-channel Bayes ceiling and the
continuous-pattern decoder on the **same** simulated surface (`background_noise_like`,
including impurity and PO), at the **same** prior:

- discrete absence-channel ceiling: **Top-1 67.38%, Top-5 85.71%**
- stage2c on the continuous degraded pattern: **Top-1 10.78%, Top-5 41.23%**
- gap: **56.60 Top-1 points, 44.48 Top-5 points**

This answers the obvious objection — *"Gate 2 already includes impurity and PO, so why
do the models do so poorly?"* The resolution is that the loss is **not** at the step
Gate 2 measures. Gate 2 measures absences → EG, and that channel is wide open (67%):
once you have the absence pattern, the symmetry is strongly determined even with
impurity and PO present. The 56-point gap is therefore located **upstream**, in
recovering the discrete absence pattern from the continuous degraded pattern.

A sharper way to state it: Gate 2 models impurity and PO as *discrete absence flips*
(present↔absent at the pattern level), which is **milder** than their full effect on
the continuous pattern. At the discrete-absence level those nuisances do not collapse
the channel (hence 67%); but on the real continuous pattern they degrade background,
intensities, and overlap such that *detecting which reflections are systematically
absent* becomes the hard, lossy step. So "we included impurity and PO" is true but
refers to their gentle discrete representation, not their full continuous-pattern
damage — and the continuous→discrete recovery step, which Gate 2 does not charge for,
is where the bits go.

**Three-way localization (the corrected framing).** The EG-information budget on this
surface decomposes into three steps, and we have now instrumented all three:

1. **absence pattern → EG** — *open*, not the bottleneck. Discrete ceiling 67.38/85.71;
   the symmetry is recoverable once the absences are known.
2. **continuous pattern → absence pattern** — *the dominant lossy step, now measured*.
   Of the 56.60-point Top-1 gap, an explicit two-stage reader localizes 53.21 points here
   (Top-5: 43.43 of 44.48). See the measured decomposition below.
3. **decoder-suboptimality given the recovered absences** — *small, now measured*. Only
   3.38 Top-1 points (1.06 Top-5) — the residual between the two-stage path and end-to-end
   stage2c.

This is a stronger and more physical statement than the two-way floor/ceiling sandwich,
and it replaces the premature "decoder-limited" framing: the dominant limit is
**absence detection on real degraded patterns**, not symmetry classification.

That is the load-bearing decoder result. The important number is not just that
`stage2c` reaches `10.78%` Top-1 on the degraded continuous benchmark. It is that
the *same-surface discrete absence channel* remains at `67.38%` Top-1. The
56.60-point gap therefore cannot be assigned to the absence->EG step, because the
discrete ceiling is precisely the performance of that step when the absence object
is known. What remains is to split that upstream gap directly into absence-recovery
loss versus residual decoder-suboptimality.

That experiment is now done. On the same `473`-example matched benchmark, a
simple 5-fold out-of-fold continuous->absence reader followed by the same
discrete Bayes decoder gives `14.16%` Top-1 and `42.28%` Top-5. Relative to the
same-surface ceiling and to end-to-end `stage2c`, the split is:

- Top-1:
  - total upstream gap `56.60` points
  - absence-recovery loss `53.21` points
  - residual downstream gap `3.38` points
- Top-5:
  - total upstream gap `44.48` points
  - absence-recovery loss `43.43` points
  - residual downstream gap `1.06` points

So the earlier inference is now directly measured on the matched simulated
surface: almost all of the missing performance lives in continuous->absence
recovery rather than in the final absence->EG decision.

**Validity check — the two-stage path tracks end-to-end.** Recovered-absence→EG
(14.16% Top-1) sits just *above* end-to-end stage2c (10.78%), as expected: the explicit
Bayes decoder is optimal given the recovered absences, so it should edge out stage2c's
monolithic map, and the ~3-point residual means stage2c is already nearly as good as the
optimal absence→EG decoder fed same-quality absences. Two different routes to the absence
representation, both bottlenecked at the same place — evidence the decomposition is real,
not an artifact of one reader.

**Absence-space readout (where recovery fails).** The reader's exact absence-symbol
accuracy is 13.74% — recovery of the *full* symbol is hard, independent of EG. Per-family
probe-state recovery, in the fixed order 100/010/001/110/101/011, is
62.8/81.2/71.9/45.5/68.7/50.5%. The weakest recovered channels are the mixed-index
**110 (45.5%)** and **011 (50.5%)** probes; the single-index probes (100/010/001, mean
~72%) recover better than the mixed-index probes (110/101/011, mean ~55%), suggesting
combined low-order absence constraints — which require reading coincident conditions
across more than one axis from a degraded 1D trace — are the harder recovery target.
**Caveat on naming:** the six families are a *probe basis, not a physically disentangled
rule basis*, so these do not map cleanly to named "glide" / "centering" channels; the
honest statement is the descriptive one (mixed-index 110/011 are weakest). Whether this
traces to the same physical conditions as the split-head/rule-mask glide fragility is an
*open question* — it would require expressing both in a common rule basis to test, and is
not asserted here as convergence.

**Per-reader caveat on the attribution.** The 53.21-point absence-recovery loss is the
loss of *this* good-faith 5-fold out-of-fold reader, not a proof that absence recovery is
information-theoretically that lossy; a better reader could recover more, shrinking 53.21
and growing the (tiny) residual. The 13.74% exact-symbol accuracy indicates the step is
genuinely hard rather than the reader weak, but the airtight phrasing is "with an explicit
continuous→absence reader at least as good as ours, 53.21 of 56.60 points are lost in
absence recovery and only 3.38 downstream."

**Independent corroboration from the error structure.** The cross-bin error
decomposition on the RRUFF-473 XRDBench evals says the same thing from the error side:
wrong predictions are essentially all *cross-bin* (CNN11 100.0%, BiGRU 99.0%,
Transformer 99.5%) — i.e. landing in the wrong absence bin (misreading the absences),
not within-bin Laue confusions. Even conditioned on reaching the correct absence bin,
within-bin Laue error is tiny (CNN11 0/60, BiGRU 4/70, Transformer 2/42). So three
independent lines — the discrete ceiling (absences contain the bits), the realized
accuracy (decoders can't extract them from continuous data), and the error topology
(failures are absence misreads) — converge on the same conclusion: the bottleneck is
continuous→discrete absence recovery.

**Ensemble redundancy still stands, in its proper role.** The small simple-ensemble
gain (Top-1 → 14.16–14.38%, Top-5 flat ~50%) remains valid evidence of shared failure
structure — but its role is the *accuracy/agreement* argument, not the bits floor. It
shows there is no easy decoder-only rescue, consistent with the bottleneck being
upstream of the decoder.

One caveat should stay explicit in downstream prose. This localization is airtight
on the matched simulated surface. What remains open is the real-data magnitude:
real RRUFF may be harder still than the `background_noise_like` rung. That does
not weaken the structural conclusion about *where* the loss is; it only means the
size of the same loss on real data may differ.

## 2. The cross-entropy bound: a diagnostic, not a result

Computing the cross-entropy MI lower bound, I(X;Y) ≥ H(Y) − CE(q), gave **negative
values for all four decoders individually, and ensembling did not fix it.** A negative
bound is not a finding about the channel — it is arithmetically the statement
CE(q) > H(Y), i.e. each decoder scores *worse than predicting the class frequencies*.
The cause is the combination of (a) overconfident, uncalibrated posteriors whose
unbounded −log penalty on confidently-wrong examples dominates the average, and (b)
genuine distribution shift — these decoders are mis-prioritized for RRUFF (the
XRDBench models trained on SimXRD; stage2c on its own mix). Both are real facts, and
(b) independently corroborates the through-line that the surviving signal is *rank-like,
not mass-like*: they carry usable ranking information on RRUFF-473, but not usable
probabilistic mass on the true label. But neither makes the negative number a reportable
bound. **The cross-entropy detour's value was diagnostic:
it told us which instruments are valid on which models.**

**Temperature scaling is rejected for the XRDBench models.** Fitting a temperature on
a held-out *RRUFF* split for models trained on SimXRD does not correct calibration on
their own distribution — it tunes a free parameter to paper over a domain gap, and it
lets every decoder peek at the evaluation distribution, inflating the bound. It is also
not uniform across the four (stage2c can be calibrated on its own held-out data; the
visitors cannot be calibrated at home and evaluated on RRUFF with the data we have).
So we do not temperature-scale the foreign models. The honest reading is that an OOD
decoder's raw cross-entropy is simply not a valid instrument for the information bound.

## 3. The valid instrument set (the defensible sandwich)

- **Ceiling (upper bound):** the exact discrete absence-channel Bayes computation
  (Gate 2), on the same surface and prior as the floor — `background_noise_like` at the
  RRUFF-473 marginal: **Top-1 67.38%, Top-5 85.71%** (job `732593`).
- **Floor — Fano is VACUOUS here, not merely weak.** The accuracy-based Fano bound
  returns exactly 0.0000 bits for every decoder and both ensembles. This is the bound
  hitting its floor for a specific reason: at Top-1 error P_e ≈ 0.86–0.89 over the
  populated class count, the penalty term H_b(P_e) + P_e·log(K−1) already exceeds H(Y),
  so the bound is negative and clamped to zero. Fano certifies essentially no MI in this
  (P_e, K) regime; report it as *vacuous*, not as a weak estimate (which would invite
  "did the code break").
- **Floor — operative instrument: shuffle-null-corrected hard-confusion MI.** Raw
  plug-in MI from the 46×46 confusion matrix is badly upward-biased at ~473 examples; a
  permutation (shuffle) null correction is required. Corrected bits:

  | decoder | raw plug-in | shuffle-corrected |
  |---|---:|---:|
  | stage2c | 0.2603 | 0.0883 |
  | BiGRU | 0.5452 | 0.2092 |
  | CNN11 | 0.7984 | **0.3249** |
  | Transformer | 0.1417 | 0.0442 |
  | mean ensemble | 0.6858 | 0.2950 |
  | log ensemble | 0.6168 | 0.2722 |

  The correction roughly halves every value, confirming the raw plug-in was inflated.
  **The tightest calibration-free floor is the best single corrected decoder, not the
  ensemble** — here CNN11 at ~0.325 bits. (Mean-ensemble MI 0.295 sits *below* CNN11:
  averaging blurs structured errors, so the ensemble is not the bits floor; its value is
  the accuracy-redundancy argument.) No temperature scaling was used anywhere in this
  floor.
- **Floor, cross-entropy, stage2c only (optional, pending):** a single
  legitimately-calibrated in-distribution cross-entropy bound — temperature + probability
  floor fit on a held-out split of stage2c's *own* training distribution. One valid
  in-distribution decoder gives a clean bound; the OOD four do not give a better one.

This yields a bits-floor with no parameter tuned against the evaluation distribution,
under the same-surface channel ceiling. But note the sandwich is now secondary to the
**three-way localization (§1b)**: the headline is that the absence→EG channel is open
(67%) while the realized accuracy is 10.78%, so the dominant loss is continuous→discrete
absence recovery, not the classification step the bits floor measures.

**Calibration-free shared-floor evidence (independent of the bound):** the small
ensemble gain (§1); the joint-error overlap vs the independence prediction (decoders
fail together far more than chance independence implies — quantify the excess factor);
and the consensus confusion matrix (averaging over independent decoders cancels
idiosyncratic confusions and leaves the channel's genuine confusability — check that it
matches the cross-bin / Laue-degeneracy decomposition). These carry the
"shared floor, not decoder-specific gap" conclusion on their own.

## 4. RRUFF non-standard settings: correction to the lattice-match path

RRUFF publishes cells in **non-standard settings**; the standard-setting cell is one we
*manufacture* by transforming. The trustworthy lattice is therefore RRUFF's original;
the standardized cell is the suspect, possibly metric-corrupting one. Consequences:

- **Feed the matcher RRUFF's native cell + native centering symbol**, and let S6/Selling
  — which is invariant to setting (a setting change is the legitimate basis change S6
  absorbs) — do the reconciliation against MP's standard cells in S6 space. Drop the
  standardization transform from the lattice-comparison path entirely; it is not just
  unnecessary there but a liability (every transform is a chance to corrupt the metric
  before S6 sees it).
- **The one consistency requirement:** pass RRUFF's cell and RRUFF's centering letter
  *together, both native* — never a native cell with a standardized centering letter or
  vice versa, or `makeprimredcell`'s primitive construction is wrong.
- **Standardization linter (one-liner with the binding):** for each case, S6 distance
  between the RRUFF native cell and our standardized cell should be ~0 (same lattice,
  different setting). Any case where it is not ~0 is a standardization bug, localized to
  that case/setting class — a direct audit of the standardization we are unsure about,
  against the data we trust.
- **Peak self-consistency**, anchored on the native cell: predict peaks from the RRUFF
  native cell and confirm they sit on the observed background-subtracted peaks
  (intensity-agnostic; powder positions constrain the metric, not the axis labeling).
  This validates parsing and internal consistency without MP.

## 5. MP-overlap (leakage) analysis: status and scope

Purpose: separate the two surviving explanations of AlphaDiffract's ~47% RRUFF lead —
*recognition* (MP train–test structural overlap) vs *cleaner input* — given that the
clean synthetic control already **rejected the capability explanation** (AlphaDiffract
collapses to the ~1% prior on uniform-EG synthetic patterns in the same RRUFF geometry,
indistinguishable from the other externals; our ViT clears it ~10×).

Mechanism: regress per-case correctness on MP-match strength for each model; a
**differential slope** (AlphaDiffract steeper than an MP-naive control) is the
recognition signature; flat-but-accurate points to cleaner input; both rising together
is a difficulty confound the differential cancels.

Built and self-tested (synthetic, stub distance + supplied peaks): match strength
separates in-pool from novel cases at 6.1σ, and the regression recovers a steep leaky
slope (3.82 [3.02, 5.27]) vs a flat control (≈0 [−0.33, 0.31]), differential CI
excluding zero. The S6 binding has since been **built** as a minimal `ncdist`-only
`s6dist` extension (job `3173105` smoke validates the rebuilt extension and the
`--max-cases 2` scan path); the real MP-overlap pass over the visually-accepted 49
processed-overlap cases (within RRUFF-209) against the staged MP candidate JSONL is
**scheduled** (`3172927`, production rerun `3173117` chained `afterok:3173105`) but had
not returned countable runtime at last ledger update. Still to confirm before trusting
output: `validate_s6dist.py --cli` axis-swap→~0 and CLI agreement, and `cs6dist` scale.

## 6. Next steps (prioritized)

1. **DONE — same-surface ceiling/floor.** Discrete ceiling 67.38/85.71 vs stage2c
   10.78/41.23 on `background_noise_like` at the RRUFF marginal (job `732593`). Write the
   three-way localization (§1b) into the full writeup as the corrected headline,
   retiring the near-saturation reading explicitly.
2. **DONE — two-stage decomposition.** Explicit continuous→absence reader + same Bayes
   decoder gives recovered-absence→EG 14.16/42.28, splitting the gap into 53.21 (Top-1)
   absence-recovery loss + 3.38 residual. Absence-symbol accuracy 13.74%; weakest probes
   mixed-index 110/011. The internal split is now measured, not inferred. **Still open:**
   (a) *sim-to-real magnitude* — repeat the absence-reader fidelity on real RRUFF patterns
   with established ground-truth absences; (b) *probe-vs-rule basis* — express the
   per-family probe weakness in a common rule basis to test whether 110/011 fragility
   coincides with the split-head/rule-mask glide weakness (currently an open question, not
   convergence).
3. **Report the corrected floor as built:** Fano documented as *vacuous* at this (P_e, K);
   shuffle-null-corrected hard-confusion MI as the operative floor, best single decoder
   CNN11 ~0.325 bits (not the ensemble). Optional: the single calibrated stage2c
   cross-entropy bound on its own held-out split.
4. **Shared-floor evidence numerically:** joint-error-overlap-vs-independence factor and
   the consensus confusion matrix; check the latter against the cross-bin / Laue
   decomposition (which already shows ~99–100% cross-bin).
5. **Drop the four-decoder cross-entropy bound** from the writeup; document the negative
   result as a calibration/OOD diagnostic, not a bound. Do not temperature-scale the
   XRDBench models.
6. **Measurement-design implication** (the constructive payoff): since the bottleneck is
   continuous→discrete absence recovery, the lever for autonomous EG determination is
   better *absence detection* — background subtraction, S/N at systematically-absent
   positions, multiphase deconvolution, or explicitly modeling absence inference — not a
   better classifier head. State this; it is where the information-theory framing pays off.
7. **Leakage:** finish `validate_s6dist.py --cli` (axis-swap→~0, CLI agreement, `cs6dist`
   scale); let the scheduled MP-overlap run (`3173117`) land; rewire to RRUFF native cell +
   centering with the standardization linter and native-anchored peak self-consistency
   gate; confirm stage2c is MP-naive; then finalize the AlphaDiffract footnote (capability
   already excluded by the synthetic control).

**Flagged open item — sim-to-real magnitude.** The localization (bottleneck =
continuous→discrete absence recovery) is established *on the simulated surface*. Whether
absence recovery on *real* RRUFF is even harder is the remaining calibration; it does not
change the structural claim, only the magnitude on real data. Step 2 begins to bound it.

## 7. Honest bottom line

The "decoders tighten the ceiling" goal was unavailable in principle — decoders only
bound the floor. But the session produced a stronger result than the one originally
aimed at. The same-surface, same-prior control shows the discrete absence→EG channel is
**open** (67.38% Top-1) while realized accuracy is **10.78%**, so the dominant loss is
neither symmetry classification (the channel is open) nor primarily decoder
suboptimality (the corrected floor is small but the gap is enormous) — it is the
**recovery of the discrete absence pattern from the real continuous degraded pattern.**
Gate 2 "includes impurity and PO," but only as gentle discrete absence-flips; their full
damage lands on the continuous pattern, in the absence-detection step Gate 2 does not
charge for. Three independent lines — the open discrete ceiling, the low realized
accuracy, and the ~99–100% cross-bin error topology — converge on this localization. The
constructive consequence is a measurement-design prescription: improve absence detection,
not the classifier. Supporting results stand in their proper roles: the small simple-
ensemble gain as shared-failure evidence; Fano as *vacuous* (not weak); shuffle-corrected
hard-confusion MI (CNN11 ~0.325 bits) as the operative calibration-free floor; the RRUFF
native-setting correction removing a bug class via a setting-invariant metric; and the
leakage analysis precisely scoped (capability rejected; recognition-vs-cleaner-input
pending the scheduled real run). The remaining open item is the sim-to-real *magnitude*
of the absence-recovery loss, which step 2 begins to bound. The paragraph we can now
write is both stronger and more honest than the one we started with.
