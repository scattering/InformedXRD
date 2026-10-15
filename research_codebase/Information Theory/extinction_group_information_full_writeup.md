# How Much Symmetry Information Survives a Powder Diffraction Measurement?
### A channel-decoding study of extinction-group inference — combined write-up

*This document is meant to stand on its own. Part 1 explains the underlying
information theory for a reader who has never seen it (a motivated undergraduate
should be able to follow). Part 2 reviews what others in the field have done.
Parts 3–5 present our approach, the external-model evidence, and our results
across three experimental "gates." Part 6 is the synthesis.*

> **Status note.** All results below are current as of this draft. The Gate 3
> story is now settled by the exact Bayes-accuracy ceilings (§5.3): the channel's
> recoverable information is flat across the degradation ladder, which makes a
> small residual non-monotonicity in the MI *waterfall* (still being cleaned up at
> the decision-rule level) scientifically immaterial. The headline floor/ceiling
> comparison (§5.3) is final.

---

## Part 1 — The question, and the information theory behind it

### 1.1 The scientific setup (no prior crystallography assumed)

A **crystal** is matter whose atoms repeat in a regular 3D pattern, like a vast
wallpaper. What a material *does* — conduct, magnetize, catalyze — depends on that
atomic arrangement, so determining structure is the central measurement problem of
materials science, and increasingly a bottleneck: we can make and even
AI-propose candidate materials far faster than we can characterize them.

The workhorse tool is **diffraction**. Shine X-rays or neutrons at the sample;
the regularly spaced planes of atoms reflect the waves so they reinforce only at
special angles (the same physics that makes a CD throw rainbows). For a *powder*
— millions of tiny randomly oriented crystals — the result is a one-dimensional
curve of intensity versus angle: a **barcode**. Powder data is cheap and common,
but compressing 3D structure into a 1D trace discards a great deal, which is why
reading structure back out is hard.

We do not attempt the full structure here, only one informative piece: the
**extinction group (EG)**, one of **99** symmetry categories. The reason this is
readable comes from a clean fact: certain symmetries force certain reflections to
cancel to *exactly zero* — not faint, forbidden. These **systematic absences**
depend on the symmetry, so the *pattern of missing peaks* fingerprints the EG.

A neural network reads the barcode and outputs a probability for each of the 99
groups. Its single best guess is right about **10%** of the time ("Top-1
accuracy"). A recent paper read that 10% as a *fundamental information limit* of
powder diffraction. We suspected this confused two very different things — "our
model tops out at 10%" versus "10% is the most any method could achieve" — and to
separate them we needed to measure *information itself*. That is what information
theory does.

### 1.2 Information theory in one page

**Bits = yes/no questions.** To pin down one of 8 equally likely options, binary
search ("is it ≤ 4?") needs 3 questions, because 8 = 2³. In general, identifying
one of *N* equally likely options costs **log₂(N)** yes/no questions — that count
is the number of **bits**.

**Surprise.** Rare outcomes are more surprising; the surprise of an event of
probability *p* is **log₂(1/p)** bits. A coin flip carries 1 bit; something you
were 99% sure of carries almost none.

**Entropy = average surprise = uncertainty.** For a quantity *G* with various
outcome probabilities, the **entropy** *H(G)* is the average surprise — the
average number of yes/no questions to identify it. If all 99 EGs were equally
common, *H(G)* = log₂(99) ≈ 6.6 bits. They are not equally common, so the real
uncertainty is lower, about **4.7 bits**. That is our starting ignorance about the
EG before looking at any barcode.

**The prior.** That "before looking" guess is the **prior** — the histogram of how
common each group is. A predictor that ignores the barcode and just names the
commonest groups is a *prior-only* baseline. Tellingly, the network's 10% Top-1 is
barely above what this histogram already achieves — the first hint that "10%" is
describing the prior, not the physics.

**Mutual information = bits the data gives you.** If looking at the barcode *X*
drops your uncertainty from *H(G)* to *H(G | X)*, the drop is the **mutual
information** *I(G;X) = H(G) − H(G|X)* — literally how many yes/no questions the
measurement answered. Useless data gives 0; a perfect fingerprint gives the full
4.7 bits.

**The channel, and capacity.** Picture the true EG as a **message** sent down a
**noisy channel** (the measurement) and arriving as the barcode *X*. The noise is
all the physical junk: background, peak overlap, preferred orientation,
contamination. The question of the whole project is *how many bits about G survive
the channel*. A specific method (our network) is one **decoder**; its accuracy is
a **lower bound** on what's recoverable — a better decoder might do more. The
channel **capacity** is the *maximum* any decoder could recover — a genuine
ceiling. **Confusing one decoder's accuracy with the channel capacity is the error
we set out to fix:** "I couldn't open the jar" is not "the jar is unopenable."
One corrective note matters for everything below: the original six-family mask we
simulate is a particular low-index, prototype-reflection reduction of the pattern,
not the canonical extinction-group object itself. It should now be treated as a
**deprecated baseline operationalization**. The measured bits and Bayes ceilings
for that object are still useful, but the operational design we should carry
forward is the orientation-invariant **rule mask** built from rule-level
systematic-absence families in the EXPO/conventional setting.

**Two kinds of "knowing."** This distinction runs through every result below.
*Mass*: putting high probability on the exact correct group (getting #1 right).
*Rank*: pushing the correct group high on a shortlist (e.g. into the top 5), even
if not #1. A model can be good at narrowing 99 options to a believable shortlist
while rarely nailing the single winner.

---

## Part 2 — What others have done

Information theory has touched scattering before, in two distinct lines.

**Information gain for experiment design (the closest precedent, much of it from
OrgA-Center).** Treece, Kienzle, Hoogerheide, Majkrzak, Lösche & Heinrich (2019) measure
the information gain of a neutron-reflectometry experiment as ΔH = H(prior) −
H(posterior) over the model parameters, and use it to optimize experimental
design. Heinrich et al. (2020) extend this to isotopic contrast and introduce
*marginalizing the information gain onto a chosen subset of parameters*. Durant,
Wilkins, Butler & Cooper (2021) take a frequentist route via the **Fisher
information**, computable analytically from the model and the counts (no posterior
sampling), giving a *maximum* information gain and real-time experiment decisions.
Our `bits_over_prior` is exactly this ΔH information gain applied to a *categorical*
label; our channel capacity is the discrete analog of Durant's maximum information
gain; and our per-rung loss attribution (Gate 3, §5.2) borrows Heinrich's marginalization.

**Maximum entropy in crystallography (a different role for entropy).** The
maximum-entropy method reconstructs electron-/nuclear-density maps from diffraction
data by maximizing configurational entropy subject to structure-factor constraints
(Gilmore 1996; Takata & Sakata 1996; International Tables Vol. H ch. 4.8), and
Bricogne's Bayesian theory of the phase problem (1984, 1988, 1991) casts structure
solution as entropy maximization plus likelihood ranking. Most relevant to us,
Sivia & David (1994) treat peak **overlap** in powder data as a nuisance with a
prior — the same overlap that appears as a degradation rung in our ladder, and the
basis for our *soft* present/absent calls. Note the distinction: this line uses
entropy as an *inference prior*, not as a measure of channel information.

**What is new here.** (1) The target is a discrete 99-way categorical, so capacity
is bounded by log₂(99) and the relevant estimator is a discrete-channel mutual
information with finite-sample issues absent from the Gaussian reflectometry case.
(2) We separate *model-extracted bits* (a lower bound) from *simulator-available
bits* (a capacity), and refuse to read model accuracy as a physics limit. (3) We
decompose the measurement into an explicit degradation channel and attribute
information loss to specific physical nuisances — a framing we have not seen in the
diffraction literature.

---

## Part 3 — Our approach: three gates

We separate three quantities and never conflate them: the **prior-only** baseline;
**model-extracted bits** (a lower bound, from a trained classifier on real data);
and **channel capacity** (a ceiling, measurable only in simulation where the true
EG is known). We test in gated stages, cheapest and most decisive first:

- **Gate 1** — real data: does the trained model extract information beyond the
  prior, and of what kind (mass or rank)?
- **Gate 2** — simulation: does the absence-mask channel carry real EG
  information, and how much?
- **Gate 3** — simulation ladder: how much EG information does each physical
  corruption destroy?

---

## Part 4 — Results, Gate 1 (real data) and external models

### 4.1 Gate 1: the surviving information is real but rank-like

On the RRUFF-325 benchmark our model reaches **9.54% Top-1, 43.08% Top-5**. We
scored its predictions two ways.

The **mass test** (`bits_over_prior`: does the model beat the prior at placing
probability on the *true* group, scored by log-loss?) came out **negative**,
about **−0.61 bits**, and still **−0.52** after retuning confidence to its best
possible (oracle) temperature. Negative means that, on average, the calibrated
model is *more* surprised by the true group than the histogram is — it is
confidently wrong on a near-miss group more often than it is right. The strict,
mass-based version of "the model is informative" therefore **fails**, robustly.

The **rank test** tells the opposite story. Top-5 and the average rank of the true
group both beat a **label-shuffle null** (scramble the answer key, re-score, see
what luck gives) — the Top-5 lift over the model's own shuffle null is significant
(one-sided *p* ≈ 0.04 at the calibrated temperature, ≈ 0.007 at oracle
temperature), and the per-pattern information varies substantially across patterns
(KL spread ≈ 1.6 bits), which a flat prior-ranker could not produce. Top-1 alone
does not beat chance.

A baseline subtlety we are careful about: on RRUFF-325 the trivial "always name the
5 commonest groups" strategy already scores ~43.7% Top-5, essentially tying the
model's 43.08%. So the model does **not** beat the dumb static baseline on raw
Top-5; what it beats is the shuffle null that destroys *per-pattern* structure. The
defensible claim is precisely "the model is not a flat prior-ranker" — it adjusts
its shortlist pattern by pattern — and nothing stronger.

**Gate 1 conclusion:** Top-1 ≈ prior does *not* mean zero information. The model
extracts real but **narrow, rank-like** information: it narrows the candidate set
per pattern without reliably nailing the exact group, and its confident mass often
lands on a symmetry-adjacent group. This is a property of extinction-group
ambiguity, not a fundamental wall, and it is exactly why such a classifier is
useful as an upstream *candidate-set constraint* rather than for end-to-end
identification.

### 4.2 External models converge to the same band

Independent groups' models reproduce this picture, which is strong evidence the
floor is a property of the data, not of our architecture.

| model | benchmark | Top-1 | Top-3 | Top-5 | coverage |
|---|---|---|---|---|---|
| ResNet-50 (Schopmann, 250M+ ICSD-biased examples) | RRUFF-473 | 15.0% | 38.1% | 52.9% | 99.6% |
| ResNet-50 | RRUFF-325 | 11.4% | 31.1% | 44.9% | 99.4% |
| ResNet-101 | RRUFF-473 | 10.8% | 25.4% | 31.1% | 99.6% |
| ResNet-101 | RRUFF-325 | 10.8% | 22.2% | 28.6% | 99.4% |

Three points. First, the external-benchmark picture is no longer a single narrow
band: Schopmann's ResNets and the `XRDBench` representatives still occupy the
modest real-data regime, but after fixing the one-based-target bug in our direct
adapter, AlphaDiffract is a substantially stronger baseline on the shared
`RRUFF-473` and exact processed-overlap `209` surfaces. Second, within the
Schopmann line the **bigger** ResNet-101 still does *worse* than ResNet-50; more
capacity and a vast training set do not automatically climb out of the modest
band. Third, coverage is ~99.5%, so the reduced space-group range is not creating
an apples-to-oranges artifact; these models are being compared on nearly the same
task support.

AlphaDiffract is still informative, but no longer as a weak-transfer negative
control. The remaining caution there is provenance rather than performance: the
paper's actual `734` / `240` manifests remain unreleased, so exact apples-to-
apples reproduction of its hidden benchmark is still impossible even though the
released checkpoint is strong on the shared surfaces we can test directly.

The right reading: these are all **lower-bound decoders**, and their convergence
shows model-extracted bits are stable across decoders — strong evidence we are near
the achievable floor for this data, not stuck on one bad model. It does not by
itself prove a capacity ceiling; that is what Gates 2–3 supply. The cross-model
agreement is the empirical *floor*; the simulation gives the *ceiling*.

We now have a sharper version of that claim from a direct decoder-floor suite on
the shared `RRUFF-473` posterior surface. Using our `stage2c` model together
with the three representative `XRDBench` decoders (`BiGRU`, `CNN11`,
`Transformer`), every individual decoder still has a **negative** strict
cross-entropy lower bound. Simple ensembles do not fix that. The best arithmetic-
mean or log-probability ensemble raises Top-1 only to `14.16–14.38%` and leaves
Top-5 essentially unchanged at `49.89–50.95%`, while pairwise shared-error
overlap factors stay close to independence (`~1.01–1.04` in most pairs). That is
what a shared measurement floor looks like: some mild complementarity, but no
large hidden reserve of decoder-specific recoverable information waiting to be
unlocked by a different output combiner.

One more correction matters. Once the discrete absence-channel ceiling is
reweighted to the empirical `RRUFF-473` marginal, the exact Bayes Top-5 ceiling
is not `~49–50%`; it is about `88.7%`, with Top-1 about `63.2%`. So the earlier
numerical resemblance between a `~50%` ensemble Top-5 and the simulation-prior
ceiling was a cross-prior mirage. On a same-prior comparison, the decoders are
still far below the discrete-channel ceiling. The calibration-free floor
instruments reinforce that. Fano from Top-1 collapses to zero for all four
decoders on the `46`-class empirical support, making it too weak to report as a
useful floor; the hard-confusion plug-in MI floor needs finite-sample correction
and, after a permutation-null subtraction, the best single decoder (`CNN11`)
lands at about `0.325` bits. That is still far below the same-prior channel
ceiling. The surviving point is therefore narrower and
cleaner: decoder diversity does not buy much, but that does **not** mean the
current decoders are close to saturating the available absence-channel
information on `RRUFF-473`.

---

## Part 5 — Results, Gates 2 and 3 (simulation)

### 5.1 Gate 2: the absence-mask channel carries ~2–3 bits

We reduce the barcode to an **absence mask** over a small set of diagnostic
reflection families (each `present` / `absent` / `unresolved`) — the part of the
signal that physically encodes symmetry — and estimate the channel p(Mhat | G)
from simulated crystals with known EGs.

The headline result, after combining all symmetry shards: the mask carries roughly
**2.8–3 bits** of EG information, with a small estimator noise floor (shuffle null
≈ 0.18 bits against a ~3-bit signal), stable across sample sizes and insensitive to
smoothing. **The channel is real and robust.**

Getting there taught the central methodological lesson. A first run returned ~0
bits — which looked like "the mask is useless." It was a **measurement artifact**:
estimating a ~600-symbol channel across 99 classes from 2 samples per class let the
smoothing filler dominate the counts. The fix was more samples (~100–300 per class)
and reporting **debiased** mutual information (subtracting the shuffle-null bias
floor), never the raw plug-in value. *Apparent zero was not real zero.*

One structural caveat the simulation can't escape: with six binary
(present/absent) families there are only 2⁶ = 64 possible ideal patterns for 99
groups, so by pigeonhole some EGs are **fundamentally unseparable** by this mask —
a design ceiling distinct from any physics limit, which we read directly off the
extinction table rather than discovering by simulation.

### 5.2 Gate 3: the degradation waterfall

Gate 3 applies physical corruptions cumulatively — each rung a downstream transform
of the previous rung's *realized pattern*, so that true-EG → clean → … → corrupted
forms a Markov chain. By the **data-processing inequality**, information can only
be lost down such a chain, so a correct ladder must show mutual information that
*decreases monotonically* from the ideal rung. We measure each step's loss with a
*paired* estimator (same structures across rungs, so structure-sampling variance
cancels) on debiased mutual information.

Current debiased per-rung mutual information:

| rung | MI (bits) |
|---|---|
| ideal_mask | 2.492 |
| structure_factor_like | 2.126 |
| broadening_overlap_like | 2.114 |
| background_noise_like | 2.016 |
| rruff_like | 2.048 |

Debiased paired per-rung losses (positive = information lost, as DPI requires):

| step | loss (bits) | 95% CI |
|---|---|---|
| ideal → structure_factor | +0.342 | [+0.334, +0.349] |
| structure_factor → broadening | +0.012 | [+0.006, +0.017] |
| broadening → background | +0.094 | [+0.085, +0.103] |
| background → rruff_like | **−0.030** | [−0.035, −0.025] |

The redesign worked where it mattered: **ideal_mask is the maximum rung** (the
anchor the earlier parallel-construction version failed), and the first three steps
are clean, positive, and tightly bounded. Two earlier failure modes — a globally
non-monotone curve, and a spurious negative loss from comparing rungs with
different alphabets — were resolved by (a) rebuilding rungs as literal transforms
of the realized pattern and (b) bootstrapping the *debiased* difference per
resample. One small inversion remains in the MI waterfall (`rruff_like` ~0.030
bits above `background_noise_like`), traced to a decision-rule artifact in the
final rung — attenuation evaluated against a stale noise reference, which can
sharpen rather than blur a borderline present/absent call. It is being cleaned up,
but §5.3 shows it is scientifically immaterial: at the level of recoverable
*accuracy*, all rungs are equivalent to within noise.

The qualitative result is already a finding: **the waterfall is shallow.** The
corruptions cost very little EG information at the absence-mask level (a few
hundredths of a bit per step after the first), because the diagnostic signal lives
in systematic *absences*, which resist the nuisances that wreck continuous-intensity
estimation. The largest single drop is at ideal → structure_factor, where idealized
absences give way to real intensities and weak-but-allowed reflections begin to
read as absent.

### 5.3 The exact accuracy ceiling — and where the real loss sits

Because in simulation we hold the channel matrix p(Mhat|G) itself, we can compute
the **exact Bayes-optimal accuracy**: the most any decoder could achieve reading
the absence mask. Observe mask m, predict the highest-posterior EG; this is correct
with probability max_g p(g|m), so the Top-1 ceiling is Σ_m max_g p(g,m), with Top-k
using the k largest posteriors. This is an exact ceiling for mask-based decoding —
not an information bound — estimated held-out to avoid the optimistic bias of a
plug-in channel.

Two facts about these numbers stand out and carry most of the weight.

**The ceiling is flat across the entire degradation ladder.** Every rung sits at
~19–21% Top-1 and ~48–49% Top-5:

| rung | Top-1 ceiling | Top-3 | Top-5 |
|---|---|---|---|
| ideal_mask | 19.2% | 38.7% | 48.4% |
| structure_factor_like | 20.7% | 39.7% | 49.2% |
| broadening_overlap_like | 20.9% | 39.7% | 49.3% |
| background_noise_like | 20.7% | 39.9% | 49.3% |
| rruff_like | 20.7% | 39.6% | 49.1% |

The corruptions essentially do not move the Bayes ceiling. This is the strongest
form of the robustness result: not "the model is robust to nuisance" (a model
property) but **"the recoverable information itself is robust to nuisance"** (a
channel property, decoder-independent). It also retires the MI-waterfall
non-monotonicity as a non-issue — the rung-to-rung ceiling differences (~1.7 Top-1
points, ideal not even strictly the maximum) are the same noise-floor scale as the
0.03-bit MI inversion, so two independent quantities agree the rungs are
near-equivalent. And clean vs degraded are identical to three significant figures
(19.83% vs 19.86% Top-1; 48.75% vs 48.77% Top-5 in the Gate 2 merged artifact),
which settles the nuisance-ordering question definitively: the degradation we apply
does not measurably reduce recoverable EG information. (The `1/effective_candidates`
heuristic, ~6.8%, is very loose against the true ~20% Top-1 ceiling — report the
exact Bayes accuracy, not the bound. Effective candidates after the mask is ~14.6:
the mask narrows 99 EGs to ~15 effective, and even an oracle names the exact group
only ~1 in 5 times.)

**The same-prior decoder floor does not meet the discrete ceiling.**

| quantity | Top-1 | Top-5 |
|---|---|
| `stage2c` on continuous `background_noise_like` patterns at the `RRUFF-473` marginal | 10.8% | 41.2% |
| same-surface discrete absence-channel Bayes ceiling | 67.4% | 85.7% |

Once the comparison is made on the **same prior and same synthetic surface**, the
near-saturation reading disappears. The initial `Top-1 = 0` result from the
minimal Gate 2 decoder control turned out to include a one-based EG numbering bug
in the scorer; after fixing that bookkeeping, the continuous-pattern `stage2c`
result is still only `10.78%` Top-1 and `41.23%` Top-5. So the same-surface gap
remains very large: `56.60` Top-1 points and `44.48` Top-5 points. This is the
decisive cross-channel correction. The discrete absence representation still
contains substantial EG information, but current decoders lose a large fraction
of it when they have to infer that representation from degraded continuous
patterns. The clean decomposition is now:

1. **absence→EG is not the bottleneck** — the discrete ceiling at
   `67.38% / 85.71%` proves the absence representation itself still strongly
   determines EG on this surface;
2. **continuous→absence is the dominant lossy step, now directly measured** — an
   explicit 5-fold out-of-fold continuous→absence reader, decoded by the *same*
   discrete Bayes decoder used for the ceiling, reaches `14.16% / 42.28%`. Of the
   `56.60`-point Top-1 gap, `53.21` points are absence-recovery loss
   (Top-5: `43.43` of `44.48`);
3. **decoder-suboptimality given the recovered absences is small** — only `3.38`
   Top-1 points (`1.06` Top-5), the residual between the two-stage path and
   end-to-end `stage2c` (`10.78% / 41.23%`). The two-stage path sitting just above
   end-to-end confirms the split is not a one-reader artifact.

The reader's exact absence-symbol accuracy is `13.74%`; per-family probe-state
recovery (order `100/010/001/110/101/011`) is `62.8/81.2/71.9/45.5/68.7/50.5%`, with
the mixed-index `110` and `011` probes weakest. Note the six families are a *probe
basis, not a physically disentangled rule basis*, so these are not cleanly "glide" /
"centering" channels; the honest reading is that combined low-order (mixed-index)
constraints are the hardest to recover. The `53.21`-point attribution is the loss of a
good-faith reader (a better reader could recover more), not an information-theoretic
floor.

That localization is better science than a generic “decoder-limited” story,
because it turns the information-theory result into an experimental prescription:
the main lever is better absence recovery, not just a different classifier head.
That means background handling, S/N at systematically absent positions,
multiphase deconvolution, and explicit absence-inference models are the natural
improvement targets.

This also resolves the earlier tension in the cleanest possible way. The discrete
absence channel itself is not close to exhausted on this matched surface. Once the
absence pattern is in hand, EG inference is comparatively easy. What is directly
established is therefore that the dominant loss lies upstream of that step. We
attribute that primarily to absence recovery on the basis of the cross-bin error
topology, and the explicit two-stage control below now measures that split on
the matched simulated surface.

That two-stage measurement is now available on the same `473`-example matched
benchmark. A simple 5-fold out-of-fold continuous->absence reader achieves only
`13.74%` exact absence-symbol accuracy, though several individual family-state
channels are much easier (`62.8%`, `81.2%`, `71.9%`, `45.5%`, `68.7%`, `50.5%`).
When those recovered absence symbols are passed through the same discrete Bayes
decoder used for the `67.38% / 85.71%` ceiling, the resulting EG accuracy is
`14.16%` Top-1 and `42.28%` Top-5. So the original same-surface gap now
decomposes directly: of the `56.60` missing Top-1 points, about `53.21` are
absence-recovery loss and only `3.38` are residual downstream gap relative to
end-to-end `stage2c`; for Top-5 the split is `43.43` versus `1.06` points.
That is the cleanest localization we have: almost all of the missing performance
on this matched surface is already gone before the final absence→EG decision rule
has a chance to act.

Two caveats define exactly how far the ceiling claim reaches:

1. **Mask, not full pattern.** This is the ceiling for decoding the absence mask.
   The mask is a function of the pattern, so the full-pattern ceiling is ≥ this; the
   mask ceiling is not a universal ceiling for the raw task unless the mask were a
   sufficient statistic. Our final canonical audit shows that this six-family mask
   is not a canonical EG invariant for many groups, so the honest statement is
   narrower: this is the ceiling for this simulated six-family mask channel.
2. **Oracle vs extraction.** The ceiling is an oracle handed the *clean* mask; the
   real models must *infer* present/absent from a noisy 1D trace (a two-stage
   channel: pattern → inferred mask → EG). The corrected Gate 2 control shows that
   this extraction step is currently the dominant loss. So the honest phrasing is
   not "models nearly match an oracle decoding the idealized mask"; it is "the
   oracle mask channel is much stronger than current continuous-pattern decoders,
   which localizes the main deficit to absence reading rather than to the final EG
   decision rule alone."

Both caveats are conditional on simulator fidelity and the chosen mask. The prior
bookkeeping caveat is now resolved for the load-bearing comparison above: the
continuous-pattern decoder and the discrete ceiling were both evaluated on the same
`RRUFF-473` marginal (`46` populated classes). So the remaining gap is no longer a
cross-prior artifact. It is a genuine cross-channel gap between the discrete mask
oracle and the current continuous-pattern decoder.

### 5.4 The right operational object: rule mask, not six prototypes

The canonical audit ended up separating three distinct effects that had been mixed
together.

**1. The original six-prototype mask is a broken operationalization.** Symmetry-
exact, the fixed prototypes `100, 010, 001, 110, 101, 011` are intrinsically
inconsistent for only **9** EGs, all orthorhombic, because their member SGs live in
different axis orientations and the fixed prototypes land on different physical
reflections per setting. In the sampled/windowed artifact this blew up to `57 / 99`
apparently inconsistent EGs, so most of that larger collapse was not symmetry at
all — it was windowing/readout artifact on top of a small symmetry brittleness.

**2. A rule-level mask fixes almost all of that waste.** Replacing the six fixed
HKLs with an orientation-invariant **rule mask** built from rule-level diagnostic
families (axial serials through index 6 to distinguish `2_1/4_1/4_2/4_3/6_1/6_2/6_3`,
zonal glide families including `n` and `d`, and centering parity classes),
canonicalized over axis permutations, gives the correct operational absence object.
In the exact noiseless symmetry calculation:

| mask | distinct patterns | Top-1 ceiling | Top-5 ceiling | effective candidates | inconsistent EGs |
|---|---:|---:|---:|---:|---:|
| six prototypes (deprecated baseline) | 18 | 18.2% | 54.5% | 9.21 | 9 |
| `rule_mask_v1` (new operational mask) | 63 | 63.6% | 99.0% | 1.81 | 0 |
| full canonical absence box | 65 | 65.7% | 99.0% | 1.75 | 0 |

So the rule mask recovers **63 of the 65** absence classes available to a complete
canonical absence signature and removes the orthorhombic setting brittleness
entirely. That is the real "fix the broken object" gain. The six-prototype mask
should therefore be treated as a deprecated baseline, not the main object of the
paper-facing story.

**3. Even a perfect absence object stops at 65/99.** The complete canonical
absence signature is not injective: it separates only **65** of the **99** EGs. This
is the irreducible absence-only limit. It is not a failure of the mask
implementation; it is a structural fact that extinction groups are "absence pattern
+ Laue class", not absences alone.

One design-limit statement remains clean regardless of implementation: any six
binary families can realize at most `2^6 = 64` ideal patterns, so with `99` EGs at
least `35` EGs must collide even in a perfect binary six-family design. The rule
mask shows that the real absence ceiling is only slightly above that in practice:
`65` distinct canonical absence signatures.

### 5.5 The last third is Laue class, not absences

The absence audit also tells us exactly what the better representation must add.
The rule mask gets to `63 / 65` of the canonical absence ceiling, so the absence
representation problem is basically solved on paper. What remains is not "better
absence engineering"; it is **Laue class**.

In the exact noiseless symmetry calculation:
- absence alone (full canonical box): `65 / 99`
- `rule_mask_v1`: `63 / 99`
- `rule_mask_v1 +` perfect Laue oracle: `99 / 99`

So the right operational redesign is:
- **rule mask** = the near-optimal absence object
- **Laue side channel** = the thing that breaks the irreducible absence degeneracy

That also explains why the full-pattern ceiling must exceed the absence-mask
ceiling: peak positions and point-group symmetry carry exactly the information that
absences cannot.

One framing point matters here. These are **noiseless exact ceilings**. The gain of
`rule_mask_v1` over the six prototypes is purchased precisely by the higher-index
serials and glide-diagnostic zonals that real measurement tends to damage first:
`300/400/600`, `420/440`, and their symmetry equivalents. So "`rule_mask_v1` is the
near-optimal absence object" is a statement **in principle**. The realized gain of
that better mask under overlap, background, impurity, and preferred orientation is
the next number to measure, not something we should assume from the noiseless
ceiling.

We have now measured that realized ceiling under the current Gate-3 nuisance
stack. Using the same simulated structures and the same cumulative nuisance
ladder, but swapping the readout from the old six prototypes to
`rule_mask_v1`, the degraded `background_noise_like` rung gives:

- six prototypes:
  - Top-1 `25.0%`
  - Top-5 `52.74%`
- `rule_mask_v1`, `s10`:
  - Top-1 `89.74%`
  - Top-5 `94.92%`
  - shuffle-null Top-1 `87.64%`
  - shuffle-null Top-5 `91.43%`
  - lift over shuffle: `+2.09` Top-1 points, `+3.49` Top-5 points
- `rule_mask_v1`, `s100`:
  - Top-1 `88.86%`
  - Top-5 `94.91%`
  - shuffle-null Top-1 `79.44%`
  - shuffle-null Top-5 `88.28%`
  - lift over shuffle: `+9.42` Top-1 points, `+6.63` Top-5 points

The important correction is that the raw `~89% / ~95%` realized rule-mask
numbers are **not** directly comparable to the binary noiseless absence ceiling
(`63.6%` Top-1, `99.0%` Top-5). The realized channel is an operational ternary
channel (`present/absent/unresolved`), and at `s10` the raw score sat only
slightly above its own label-shuffle null. So the right quantity here is the
lift above shuffle, not the raw Top-1 itself.

The `s100` rerun changes the read materially. The raw score stays high, but it
also separates more clearly from shuffle (`+9.42` Top-1 points, `+6.63` Top-5
points), so the effect is not just tiny-sample inflation. That supports a real
simulator-conditioned gain for the operational rule-mask channel. But it is
still provisional: the current nuisance stack is favorable to the higher-index
rule-mask reflections, and the raw realized score should not yet be written as a
calibrated real-data absence ceiling.

The caveat is important. This should be read as a **simulator-conditioned
realized ceiling**, not yet as a calibrated real-data claim. The current nuisance
stack is still favorable to the higher-index rule-mask reflections. So the result
shows that the representation fix is potentially very large if those diagnostic
reflections survive, but it does not yet prove that the same gain survives under
fully realistic impurity, preferred-orientation, and measurement corruption.

We then took the operational next step and collapsed the realized rule-mask
encoding down to the actual physical-family object. The first coarse collapse
was still wrong: it produced a held-out score below its own null, which exposed
a scorer/alphabet mismatch rather than a property of the mask. After fixing the
held-out/null footing and using the `v3` physical-family collapse, the
completed `s100` artifact gives the first credible per-pattern comparison on
identical scoring:

- `rule_mask`
  - symbols: `372`
  - held-out Top-1 `15.94%`
  - held-out Top-5 `49.18%`
  - null Top-1 `0.97%`
  - null Top-5 `4.75%`
- `six_family`
  - symbols: `136`
  - held-out Top-1 `11.89%`
  - held-out Top-5 `42.11%`
  - null Top-1 `0.93%`
  - null Top-5 `4.74%`

So under the corrected scorer and the lower-cardinality physical-family
encoding, the realized rule mask now does beat six-family on the simulated
degraded channel:

- `+4.05` Top-1 points
- `+7.08` Top-5 points

Matched-fold bootstrap shows that this final-rung gain is statistically clean:

- held-out Top-1 gain `+4.04` points, 95% CI `[+3.16, +5.03]`
- held-out Top-5 gain `+7.16` points, 95% CI `[+5.97, +8.25]`

And the same comparison across the full ladder shows a monotone nuisance tax on
the rule-mask advantage itself:

- `ideal_mask`: Top-1 `+16.79` points, Top-5 `+19.30`
- `structure_factor_like`: Top-1 `+11.17`, Top-5 `+12.85`
- `broadening_overlap_like`: Top-1 `+9.90`, Top-5 `+11.50`
- `rruff_like`: Top-1 `+4.34`, Top-5 `+7.55`
- `background_noise_like`: Top-1 `+4.04`, Top-5 `+7.16`

That is the correct operational readout to keep. It is still strongly
simulator-conditioned, and the diagnostics show the channel is paying a real
nuisance tax:

- serial undetermined rate: `16.95%`
- glide undetermined rate: `55.16%`
- centering undetermined rate: `50.24%`
- high-index missing rate: `44.44%`
- low-index missing rate: `32.59%`

This is why the gain is modest rather than dramatic. The rule mask wins on the
strength of the serial families, which remain readable, while the glide,
centering, and higher-index diagnostics that would push it closer to the
noiseless `63 / 65` ceiling are degraded into `undetermined` or `missing`
states by the end of the ladder.

So the updated conclusion is narrower and cleaner. The noiseless redesign still
matters in principle, and with the right operational readout it can beat the
old six-family mask under the current simulated nuisance model. But the same
simulated channel is still far more favorable than current real powder data, so
this should be read as evidence of feasible representation gain under nuisance,
not as a claim that the real-data bottleneck has been solved.

We also tried the most obvious targeted refinement: a `rule_mask_v2` that fits
glide type from all surviving zonal probes and centering from all surviving
reflections, while dropping the most fragile high-index diagnostics. This did
exactly what it was designed to do at the feature level — it drove glide and
centering `undetermined` nearly to zero — but it failed decisively as a decoder.
The decisive point is that the failure is visible even before nuisance. In the
exact noiseless symmetry calculation:

- `rule_mask_v1`: `63` distinct patterns, Top-1 `63.64%`, Top-5 `98.99%`,
  effective candidates `1.81`
- `rule_mask_v2`: `26` distinct patterns, Top-1 `26.26%`, Top-5 `75.76%`,
  effective candidates `5.38`

So `v2` is not merely a bad degraded-channel estimator. It throws away
discriminative structure in the exact symmetry object itself. On the same
degraded `s100` artifact:

- `v1_rule_mask`: Top-1 `16.18%`, Top-5 `50.39%`
- `v2_rule_mask`: Top-1 `4.70%`, Top-5 `21.35%`

with held-out `v2 - v1` losses of:

- Top-1 `−11.47` points, 95% CI `[−12.29, −10.54]`
- Top-5 `−29.10` points, 95% CI `[−30.56, −27.78]`

and the same sign across the full ladder, including the ideal rung. The lesson
is important: simply forcing more categorical decisions is not enough. `v2`
eliminated `undetermined` by collapsing weak evidence into fitted rule labels,
but in doing so it destroyed discriminative structure at the exact symmetry
level and then failed even harder once absence-corrupting nuisance was added.
So the current best operational absence object remains `rule_mask_v1`, not the
first targeted redundancy redesign.

### 5.6 Missing nuisance can close the floor/ceiling gap

The floor/ceiling gap is now constrained quantitatively. Starting from the clean
Gate 2 mask-channel ceiling (`19.23%` Top-1, `47.87%` Top-5), a simple analytic
corruption model on the mask shows that modest **absence-destroying** nuisance is
already enough to land in the real-data regime:

- impurity-style `absent → present` flips:
  - `5%`: Top-1 `16.93%`, Top-5 `43.21%`
  - `10%`: Top-1 `15.02%`, Top-5 `39.62%`
- preferred-orientation-style `present → absent` erasures:
  - `5%`: Top-1 `17.59%`, Top-5 `45.47%`
  - `10%`: Top-1 `16.14%`, Top-5 `43.20%`

This supports a stronger mechanistic statement than we had before: the simulator's
flat nuisance ladder does not mean nuisance is irrelevant; it means the ladder we
instantiated missed the corruptions that most directly destroy absence information.
In the mask picture, impurity-like `absent → present` corruption bites harder than
preferred-orientation-like `present → absent` erasure at the same rate, which is
physically plausible because the systematic **absence itself** is the diagnostic
signal. The caveat is that this is an analytic flip model on the mask, not a
calibrated physical impurity simulation: it establishes plausibility and slope,
not a fitted real nuisance rate.

We then extended the realized impurity sweep across **all** Gate 3 rungs:

- `ideal_mask`
- `structure_factor_like`
- `broadening_overlap_like`
- `rruff_like`
- `background_noise_like`

Artifact:
- `/work2/ALLOC3/UserE/information_theory_outputs/impurity_sweep_rulemask_allrungs_s100.json`

Here the quantity that matters is the **held-out** Bayes ceiling, not the exact
in-sample channel fit: the exact `rule_mask_v1` fit rises with impurity because
the corrupted symbol alphabet fragments, which is not an operational gain.

The all-rungs result is cleaner than the earlier single-rung read because it
collapses the raw table to a single quantity per rung: the **crossover impurity**
\(p^*(\mathrm{rung})\), defined by linear interpolation of the held-out Top-5
threshold where `rule_mask_v1` and six-family are equal. This is the right
figure-level summary because it answers the operational question directly:
*how much absence-corrupting impurity can the richer rule mask tolerate before the
coarse six-family control becomes the better decoder?*

Using held-out Top-5 ceilings, the interpolated crossover curve is:

| rung | `p*` impurity |
|---|---:|
| `ideal_mask` | `16.9%` |
| `structure_factor_like` | `4.60%` |
| `broadening_overlap_like` | `3.35%` |
| `rruff_like` | `1.32%` |
| `background_noise_like` | `1.27%` |

That curve is the climax of the impurity analysis.

Three points matter.

First, the six-family channel is the **robustness control**. It degrades
monotonically and relatively slowly under the same impurity operator, while the
richer rule mask starts higher but collapses faster. So the issue is not that
all absence channels fail equally; it is specifically that the extra diagnostic
families bought by `rule_mask_v1` are the first ones impurity destroys.

Second, the rung dependence is strongly **multiplicative**, not additive. Moving
from the ideal rung to the late realistic rungs shifts the tolerated impurity by
more than an order of magnitude: from about `16.9%` at `ideal_mask` to about
`1.3%` at `rruff_like` / `background_noise_like`. That is the cleanest evidence
that impurity interacts with existing degradation rather than merely subtracting a
fixed amount of accuracy.

Third, the result must be read with the right caveats. The ceiling being crossed
here is a **held-out Bayes ceiling** for the simulated channel, not a direct
experimental accuracy. And the impurity axis is a **knob-unit nuisance rate** for
an inverse-norm weighted `absent -> present` corruption operator, not a calibrated
weight-fraction estimate for real specimens. So `p*` is best interpreted as a
comparative tolerance curve, not as a literal laboratory impurity percentage.

The same qualitative pattern appears in Top-1, but Top-5 is the cleaner summary
because it is more stable across rungs and more tightly connected to the
rank-based real-data floor.

So the refined conclusion is sharper than before:

- `rule_mask_v1` is genuinely the better absence object when absences remain
  faithful;
- its advantage is **impurity-fragile**;
- and the impurity tolerance shrinks catastrophically as one moves down the
  physical degradation ladder.

That is the measurement-limited mechanism in explicit form: the representation is
sound, but the gain it offers is conditional on phase purity / absence fidelity.

The error structure stays in the same regime across the sweep. For both channels
the cross-bin fraction of wrong calls remains in the same high-`90%` band, so
impurity does not move the failure mode into Laue ambiguity; it keeps it as an
absence-read failure. That is the mechanistic bridge back to the real-model
decomposition.

### 5.7 Real errors are overwhelmingly measurement-limited, not Laue-limited

The exact ceilings above let us ask the sharper real-data question: when a model is
wrong, is it mostly confusing EGs that share the same canonical absence pattern
(a **within-bin**, Laue-limited error), or is it usually jumping to an EG in a
different absence bin (a **cross-bin**, measurement-limited error)? Using the
canonical absence-bin decomposition on the completed `RRUFF-473` XRDBench evals:

| model | Top-1 | Top-5 | wrong | within-bin | cross-bin |
|---|---:|---:|---:|---:|---:|
| CNN11 | 12.68% | 41.86% | 413 | 0 | 413 |
| BiGRU | 13.95% | 43.97% | 407 | 4 | 403 |
| Transformer | 8.46% | 35.10% | 433 | 2 | 431 |

So the wrong-prediction budget is essentially **all cross-bin**:
- CNN11: `100.0%` cross-bin
- BiGRU: `99.0%` cross-bin
- Transformer: `99.5%` cross-bin

This closes the argument. The current real models are not failing because they
reach the absence ceiling and then get stuck on Laue degeneracy. They are failing
much earlier: they are **misreading the absence pattern itself** under measurement
degradation. The realized Laue-limited floor is tiny at current performance.

One caveat is worth stating precisely. The unconditional within-bin count
understates the realized Laue-limited share, because a model that rarely reads the
absence pattern correctly will also rarely have the chance to make a same-bin,
wrong-Laue error. The conditional number is therefore more honest: *among the small
set of cases where the predicted EG lands in the correct absence bin, how often is
it still wrong?* That conditional within-bin error rate is:

- CNN11: `0 / 60 = 0.0%`
- BiGRU: `4 / 70 = 5.7%`
- Transformer: `2 / 42 = 4.8%`

So even conditioned on reaching the correct absence bin, Laue confusion is still a
small residual effect at current performance. The main point survives intact:
Laue degeneracy is not the active bottleneck for these models yet; measurement-
degraded absence reading is.

That in turn fixes the operational priority:
- the rule mask remains the right *in-principle* absence representation
- but the binding real-world constraint is still **measurement-limited absence
  reading**
- improving absence-family design and adding Laue features are both secondary to
  getting the diagnostic reflections through the nuisance channel intact

### 5.8 The split head measures the same decomposition directly

The split-head model already factorizes prediction into a `37`-dimensional
structured code:

- `7` system / crystal-system logits
- `5` lattice-centering logits
- `25` operator bits

So the overlap with the information-theory factorization is not just conceptual.
It is instrumented in the model itself. The system and lattice heads are coarse
categorical side channels; the operator block is the absence-side structured code.
That lets us measure the decomposition directly on `RRUFF-473` from saved logits,
rather than inferring it only from final EG confusions.

For the manuscript `stage2c` checkpoint, the head-level results are:

- system / crystal-system head accuracy: `35.94%`
- lattice-centering head accuracy: `40.38%`
- operator-bit accuracy: `82.17%`
- operator exact-match: `0.63%`

For the stronger `mixed2500k` checkpoint:

- system / crystal-system head accuracy: `35.31%`
- lattice-centering head accuracy: `48.20%`
- operator-bit accuracy: `92.49%`
- operator exact-match: `5.92%`

These raw head accuracies need an immediate caveat: on the `RRUFF-473` targets
the trivial majority baselines are already high.

- system / crystal-system head prior baseline: `45.45%`
- lattice-centering head prior baseline: `66.60%`
- operator-bit mean majority baseline: `94.00%`
- operator exact-code most-common-pattern baseline: `13.74%`

So the head-level readout is best understood as a **diagnostic under train/test
prior shift**, not as a claim that the heads are strong standalone real-data
predictors.

The calibration picture is also informative. For `stage2c`, the system head is
strongly overconfident (mean confidence `73.99%`, ECE `0.386`), the lattice head
is somewhat better calibrated but still weak (`58.88%` confidence, ECE `0.196`),
and the operator groups are much cleaner:

- serial/screw operator group: `88.84%` accuracy, ECE `0.0446`
- glide-like operator group: `78.42%` accuracy, ECE `0.0198`

The weakest individual operator features are concentrated in glide-like families:

- `Pos2_n`: `58.77%`
- `Pos2_a`: `62.58%`
- `Pos3_n`: `63.21%`
- `Pos2_c`: `63.85%`
- `Pos3_ab`: `68.08%`

This is the direct, model-internal version of the same conclusion reached from the
channel analysis. Three independent methods now pick out the same physical
failure mode:

- the simulated rule-mask channel identified glide-like families as the most
  fragile / undetermined,
- the `rule_mask_v2` redesign failed because it forced commitment on corrupted
  glide/centering evidence,
- and the weakest real operator features in saved logits are glide-like.

That convergence is stronger than any one absolute head-accuracy number. The
second key point is the compounding arithmetic of the structured decode: local
bit information survives much better than the assembled symbolic answer. A modest
per-bit drop from `mixed2500k` to `stage2c` (`92.49% -> 82.17%`) compounds into a
nearly order-of-magnitude collapse in exact operator match (`5.92% -> 0.63%`).
So the split head should be understood as a useful structured supervision and
fusion path, not as a robust standalone symbolic decoder. It also upgrades the
information-theory narrative from *inferred* to *directly measured*: the same
physical partitioning appears in the saved logits.

One boundary is important. This pass does not yet prove a one-to-one mapping
between every operator output and the final `rule_mask_v1` categories, and the
grouped glide-like calibration is not catastrophically worse than the grouped
serial calibration. So the strongest safe claim is not “glides are globally
miscalibrated,” but the narrower and defensible statement that the weakest
real-data operator features are concentrated in glide-like families, consistent
with the simulated fragility diagnostics. The raw head accuracies themselves
should always be read against their `RRUFF-473` trivial baselines.

---

## Part 6 — Synthesis and takeaways

The original claim was that ~10% Top-1 marks a fundamental information limit of
powder diffraction. The three gates plus the external evidence replace that with
something more careful and more useful:

- **Gate 1:** Top-1 near the prior does not mean no information; the model extracts
  real but **rank-like** information (narrows the candidate set), not mass-like
  (does not nail the exact group). Confirmed across independent architectures, with
  no benefit from added capacity — so the limit is not architectural.
- **Gate 2:** the absence-mask channel genuinely carries **~2–3 bits** of EG
  information; the apparent "zero" was an estimation artifact, and the original
  six-prototype mask is now demoted to a deprecated baseline.
- **Better absence object:** a rule-level, orientation-invariant mask recovers
  **63 of the 65** canonical absence classes, so the absence representation problem
  is basically solved on paper; the remaining absence-only gap is Laue class.
- **Gate 3:** once built as a proper data-processing chain, the channel is
  **robust**, and the exact Bayes-accuracy ceiling makes this decoder-independent:
  the ceiling is **flat across the whole degradation ladder** (~20% Top-1, ~49%
  Top-5 at every rung), so it is the *information itself*, not just the model, that
  resists measurement nuisance. Clean and degraded ceilings are identical to three
  significant figures, settling the nuisance-ordering question.
- **What binds in practice:** real errors are almost entirely **cross-bin absence
  misreads**, not within-bin Laue confusions, so the dominant current limit is
  still measurement degradation rather than irreducible Laue ambiguity. Laue is
  not yet in contact with these models.

The unifying methodological lesson — the most transferable thing here — is
**apparent zero ≠ real zero, and a sub-resolution gap is not an ordering claim.** It
recurred at every gate: Top-1 ≈ prior read as "no information," capacity ≈ 0 read as
"dead mask," tiny rung gaps read as degradation ordering. Each time, the discipline
of comparing against the right null, separating model-extracted bits (a floor) from
channel capacity (a ceiling), and refusing to interpret differences below the
estimator's resolution turned a misleading number into a correct one.

So the contribution is not "10% is the limit." It is a three-level hierarchy:
**noiseless complete absences nearly determine the EG; Laue class closes the final
third; real measurement collapses the achievable accuracy back toward ~43% Top-5.**
That makes the binding current constraint neither the symmetry mathematics
(essentially solved by rule families + Laue) nor model architecture (capacity does
not help), but **measurement degradation of the diagnostic reflections**, with
impurity-like absence-destroying nuisance the dominant single mechanism we have
identified. That is a more careful and more useful claim than the one we set out to
examine.

---

## References

- Treece, B. W., Kienzle, P. A., Hoogerheide, D. P., Majkrzak, C. F., Lösche, M. &
  Heinrich, F. (2019). *J. Appl. Cryst.* **52**, 47–59. doi:10.1107/S1600576718017016
- Heinrich, F., Kienzle, P. A., Hoogerheide, D. P. & Lösche, M. (2020). *J. Appl.
  Cryst.* **53**, 800–810. doi:10.1107/S1600576720005634
- Durant, J. H., Wilkins, L., Butler, K. & Cooper, J. F. K. (2021). *J. Appl.
  Cryst.* **54**, 1100–1110. doi:10.1107/S160057672100563X
- Durant, J. H., Wilkins, L. & Cooper, J. F. K. (2022). *J. Appl. Cryst.* **55**,
  769–781. doi:10.1107/S1600576722003831
- McCluskey, A. R. et al. (2023). *J. Appl. Cryst.* **56**, 12–17.
  doi:10.1107/S1600576722011426
- Gilmore, C. J. (1996). *Acta Cryst.* **A52**, 561–589.
- Bricogne, G. (1984). *Acta Cryst.* **A40**, 410–445.
- Bricogne, G. (1988). *Acta Cryst.* **A44**. (Bayesian phase problem, multichannel MaxEnt)
- Bricogne, G. (1991). *Acta Cryst.* **A47**, 803–829. (extension to powder data)
- Sivia, D. S. & David, W. I. F. (1994). *Acta Cryst.* **A50**. (structure-factor amplitudes from powder; overlap)
- Sivia, D. S., Hamilton, W. A. & Smith, G. S. (1991). *Physica B* **173**, 121–138.
- Takata, M. & Sakata, M. (1996). *Acta Cryst.* **A52**, 287–290.
- International Tables for Crystallography (2019), Vol. H, ch. 4.8 (MEM for powder diffraction).

*Page numbers for the pre-2000 Acta Cryst references (Bricogne 1988; Sivia & David
1994) should be confirmed against the originals before submission.*
