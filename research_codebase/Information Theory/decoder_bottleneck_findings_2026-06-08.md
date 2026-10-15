# What's actually limiting diffraction→symmetry decoders — findings (2026-06-08)

## The question
PXRD→extinction-group (EG) decoders — ours and every external baseline — plateau around
**~10% Top-1 on real RRUFF**, far below the **~67% absence-channel ceiling** (Bayes-optimal
EG given the present/absent pattern). Where does the loss actually live, and is it the data,
the representation, or the model?

## TL;DR
The bottleneck is **not** the data degradation (preferred orientation / background / noise /
impurity cost almost nothing in a controlled test), **not** representation, scale, Q-range, or
Laue class. It is two **detector/architecture** problems:

1. **Localizing the diagnostic reflections** — knowing *where* to look is worth **~+20 Top-1**, a global reader fundamentally cannot do it, and a real adaptive peak finder **realizes ~80% of it** (§6–§7): 40→86% strong-peak recall, and ≈+18 of the +20 survives when its detected positions feed the reader.
2. **Detecting the weak, mixed-index reflections** to decide present vs absent — the binding wall: even a near-matched-filter wavelet finder detects only **~41–44%** of the diagnostic family reflections (86% of strong but **41% of weak**), so a large part of the gap to the ~66 ceiling is **intrinsic SNR**, not reader weakness. Persists on perfectly clean data.

The lever is therefore a **position-aware, weak-peak-sensitive absence reader**, not cleaning the
simulation or chasing nuisances. This also explains why scale and "fix the generator" don't help.

---

## 1. The loss is in absence *recovery*, not representation or absence→EG
- **Representation is near-sufficient**: given the clean absence pattern (rule-mask), EG is ~90%
  recoverable; given only crystal family (six-family), ~14–28%. So absences carry the EG; family
  does not.
- **Everyone fails at the same step**: cross-bin error fraction is **~100%** across our model,
  XRDBench (CNN/RNN/Transformer), and Schopmans ResNet — errors are *absence misreads*, not Laue
  confusions. Verified **not** an off-by-one (convention checked; within-bin errors that occur are
  genuine Laue partners).
- **Channel decomposition** (matched simulated surface, RRUFF-473 marginal): ceiling **67.4/85.7**
  (given absences) → recovered-absence reader **14.2/42.3** → end-to-end **~10.8/41.2**. The collapse
  is the recovery step.

## 2. Levers that are NOT the bottleneck (ruled out)
- **Higher Q / synchrotron / PDF**: the pattern→absence(→EG) ceiling is **flat (~98% Top-1 from
  q=4 to q=15)** — EG-discriminating systematic absences are *low-index*; extending Q is redundant.
  (Earlier "lab window limits it" was an artifact of a 17-reflection probe set; with all reflections
  the lab window already saturates.) Thermal damping is a non-issue here for the same reason
  (diagnostic absences are low-Q, barely damped).
- **Data scale**: Schopmans trained on **261M** patterns and still lands in the same modest real-data
  band with ~100% cross-bin. Scale feeds *SG-synthetic structure-type recognition*, not *EG-real
  absence recovery* — a clean dissociation.
- **Laue class**: +2 to +4 points only.
- **Coarse representation**: already solved by the rule-mask.

## 3. Cross-model comparison (verified from source / paper)
- **Schopmans / ML4pXRDs**: space-group classifier; **261M** random-crystal patterns; **sqrt**
  preprocessing; models background (GP) + add/mult noise + pseudo-Voigt (Caglioti) + **impurity ≤5%**;
  **no preferred orientation**; **B=0** thermal (DW machinery present but fed an empty dict). Its higher
  RRUFF Top-1 (38.7%, sqrt=F) is **structure-type fingerprinting** (in-distribution recognition), not
  absence physics — proven by (a) sqrt *collapses* it 38.7→15 (upweighting weak/near-absent reflections),
  and (b) ~100% cross-bin errors.
- **Ours**: EG classifier; RRUFF-conditioned; **did** train with PO (500k March-Dollase in the mixed
  final model); **B=0** thermal; models background/noise/impurity.
- Both omit thermal. The only real asymmetries: we model PO (they don't); they use sqrt (we don't).

## 4. Index-first (classical) hits the same wall
Built a full pipeline: CWT peak-finder + **Conograph** (MIT, built on TACC).
- **Peak-finding**: 72% of RRUFF-473 yield enough precise peaks (median 24–25 for mono/ortho; ~100%
  precision); the rest are genuinely peak-poor / low-symmetry.
- **Conograph on full RRUFF-473 with TRUE DIF peaks**: **top-1 = 30%**, **any-candidate = 75%**. The
  true cell is *recoverable* 75% of the time but de Wolff's M **mis-ranks** it (cubic: 0% top-1 / 93%
  any-candidate — always found, never #1). The 30→75 gap is a **selection/FoM weakness**, scoped as a
  coverage-based re-rank (Phase 1+2; see `indexing/COVERAGE_SELECTION_PLAN.md`).
- **Why the mis-rank**: the M-winners are subcells that *genuinely index the peaks* (positions
  underdetermine the cell) or tiny cells that cover by reciprocal-lattice density. Breaking the
  degeneracy requires the **systematic absences** — i.e. classical indexing's subcell problem couples
  right back to the absence problem. Classical and NN reach the same wall from opposite sides.

## 5. The oracle-position experiments — the decisive controls
Gate-2 channel (6 low-index families → EG; ceiling 66.4/85.3). LogReg/RandomForest per family,
features = global PCA of the pattern (no positions) vs local windows at the **geometric** family
positions (oracle positions; reader must still *decide* present/absent).

| reader | Top-1 |
|---|---|
| Ceiling (true absences) | 66 |
| Oracle-position, RandomForest | **35** |
| Oracle-position, LogReg | 24 |
| Global-PCA, RandomForest | **15** |
| Global-PCA, LogReg | 16 |

- **Localization is worth ~+20** (oracle 35 vs global 15, RF). The global reader is **stuck at 15
  regardless of classifier strength** — it cannot exploit a better reader because it doesn't know
  *where* to look. (My earlier +8 was a weak-LogReg artifact; an earlier +28 was a confound where the
  threshold reader got *absent* families labeled for free.)
- **Reader-limited**: LogReg 24 → RandomForest 35 — the present→absent error was substantially a weak
  classifier, not a vanished signal.
- **Degradation is NOT the culprit** (clean vs degraded oracle reader): **CLEAN 38.7** vs
  **DEGRADED 35.0** — adding bg+noise+PO+impurity costs only **−3.7**. The 38.7→66 gap exists on
  *perfectly clean* data.
- **The residual is the weak mixed-index reflections**: mixed-index families (110/101/011) recover
  ~80–85% even clean vs axial ~92%; present→absent suppression ~18% even clean. By crystal system the
  all-families-correct rate is **cubic 61% / triclinic 59%** (equivalent families / no absences) down to
  **orthorhombic 16% / tetragonal 7%** — the systems whose EG-distinguishing absences live on the weak
  mixed-index families.

**Conclusion**: the wall is **(a) localization and (b) detecting weak mixed-index reflections** — both
detector/architecture problems, both present on clean data. Chasing PO/noise/impurity cannot fix them.

## 6. Peak-finding upgrade (Gregoire wavelet, overlap resolution) — now adaptive + benchmarked
Ported `johnmgregoire/PyWaveletPeakID` (Gregoire 2011 CWT + parent/child-ridge overlap resolution) to
Py3, then **made the threshold adaptive / noise-relative** so a single setting generalizes (no
per-pattern tuning — that was the "finicky" problem). Recipe: scale grid set from the Caglioti
resolution (not tuned); per-scale Donoho MAD noise σ[j]; keep peaks with SNR = wt/σ[scale] ≥ k_snr
(scale-invariant). The only global knob is k_snr.

**Full RRUFF-473 benchmark vs DIF truth** (resolvable-cluster matching; 471/471, ~11 s on one Stampede
node). recall_strong = recovery of ≥5%-intensity (indexable) peaks:

| finder | recall_all | recall_strong | precision | med #pk |
|---|---|---|---|---|
| ours (high-precision Ricker+doublet) | 24.9% | 40.4% | 96.2% | 24 |
| **Gregoire adaptive k=3.0** | 64.7% | **86.4%** | 75.2% | 65 |
| Gregoire adaptive k=4.0 | 58.6% | 83.3% | 80.7% | 55 |

The adaptive finder **~doubles strong-peak recall (40→86%)** at a precision cost, with one global
threshold. By crystal system the gain is **largest exactly where it's needed** — recall_strong
triclinic 31→89, monoclinic 35→91, orthorhombic 42→94, tetragonal 42→82, hexagonal 49→80 — the dense,
low-symmetry patterns whose EG-distinguishing absences sit on weak, overlapped mixed-index reflections.
**Cubic is the only tie** (60↔58): few strong well-separated peaks, our finder already gets them.
This delivers the **localization half** of the prize (per §5, localization ≈ +20): accurate positions
for both a position-aware absence reader and autonomous indexing. (Details:
`project-indexing/indexing/PEAKFIND_ADAPTIVE_RESULTS.md`.)

## 7. The decisive test — is the +20 localization lever *realizable* with a real finder? (yes, ~80%)
The §5 oracle used *geometric* family positions (knows the cell). The deployable question: replace
those with positions a **real adaptive peak finder** produces, holding indexing perfect (geometric
position used only as the search anchor), and read present/absent from what the finder actually detects
(`run_peakfinder_reader.py`, both RRUFF-473 rungs). At the anchors the finder detects only **41–44%** of
the family reflections (the weak-recall wall, §6) — yet most of the localization gain survives:

| reader (RandomForest, Top-1) | clean (sf) | degraded (bgnoise) |
|---|---|---|
| global (no localization) | 16.3 | 16.3 |
| oracle (geometric window) | 38.7 `[+22.4]` | 35.0 `[+18.6]` |
| **peak-finder window (realistic)** | 33.8 → **78%** | 29.9 → **73%** |
| **peak-finder compact [found,Δpos,snr]** | 34.9 → **83%** | 32.0 → **84%** |

**~75–85% of the +20 localization lever survives realistic peak-finding** (with LogReg the realistic
reader even *matches/beats* oracle — centering on the actual detected peak localizes better than the
slightly-off geometric position). The **compact detection features beat the raw window** → the explicit
*found/position/SNR* verdict carries the localized information; the raw window just adds overfit noise.
So the practical pipeline (find → index → read absences at predicted positions) reaches **~32–35 Top-1,
≈2× the no-localization reader (16)**, on real RRUFF. The residual to the 66 ceiling is the
detection/intrinsic-SNR wall (only ~44% of family reflections are detectable), per §6. The two
independent lines now close: oracle says localization ≈ +20; a real finder delivers ≈ +18 of it.

---

## Unified picture & where the lever is
- Not the nuisances (−3.7), not scale (261M still fails), not Q (flat), not Laue (+2), not the
  representation (rule-mask solves absence→EG).
- The decoder is gated by **localizing the diagnostic reflections (~+20, global readers can't — and
  §7 shows ~80% of this is *realizable* with a real adaptive peak finder) and detecting the weak
  mixed-index reflections to call present/absent (only ~44% are detectable — intrinsic-SNR wall)**.
- Index-first *can* recover positions (75% with true peaks); a real finder gives 86% strong-peak recall
  and, fed to the reader, realizes ≈+18 of the +20 localization (§7). The remaining limit is detection
  of the weakest reflections, which is also what gates index-first (the absence degeneracy).

### What to build next
1. **Position-aware absence reader — now validated, build the production version.** §7 proves the
   compact *found/Δpos/SNR* features at finder-detected positions retain 83–84% of the +20 (beating the
   raw window). So the production reader is: adaptive finder → index → per-family detection features at
   predicted positions → EG. Expected ~32–35 Top-1 (≈2× global) on real RRUFF; the ceiling to 66 is the
   intrinsic-SNR detection wall, not architecture.
2. **Coverage-based candidate selection for Conograph** (Phase 1+2) — lifts autonomous unit-cell
   determination 30→toward 75 (separate goal: get the cell, not the symmetry).
3. **Hybrid peak finder** — adaptive Gregoire is DONE (§6: 40→86% strong recall, one global
   threshold). Remaining refinement: fuse our finder's 96% precision with Gregoire's recall (use ours
   to gate/clean Gregoire's detections) so indexing isn't fed the ~25% false positives.

### Open question — now partially answered
On clean data the oracle reader gets 38.7 vs the 66 ceiling — is the remaining gap a *better reader*
away (matched-filter/CNN weak-peak detector) or the *intrinsic SNR* of the weakest reflections? The
RandomForest jump (24→35) says meaningful reader headroom remains.

**New evidence (peak-finder, §6) leans toward "intrinsic" for the weakest reflections.** The adaptive
wavelet finder recovers **86% of strong but only 41% of weak (<5%) reflections** on RRUFF-473. The
Ricker CWT is ~a matched filter (near-optimal detection of a known peak shape in noise); that it misses
~59% of weak reflections means those reflections sit **below the detectability floor of an optimal
linear detector** — the signal isn't there to decide on. Since EG-distinguishing absences live on weak
mixed-index reflections, and a *missed* weak reflection is indistinguishable from a *systematic absence*
(a forced false-"absent"), a substantial part of the 38.7→66 gap is intrinsic SNR rather than reader
weakness. Reader headroom remains for the *detectable* reflections (the RF jump); the weakest are
largely irreducible from a single lab pattern. **Decisive next test:** swap the oracle's geometric
family positions for peak-finder positions in the gate-2 reader and measure how much of the +20
localization survives realistic 86%/41% recall — this separates indexing error from intrinsic SNR.
