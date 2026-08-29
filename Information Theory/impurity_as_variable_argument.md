# Impurity as a Controlled Variable: The Argument

*Draft section for "Attention Is Not All You Need for Diffraction." Slots after the
realized-rule-mask result (§5.6) as the mechanistic capstone of the measurement-limited
story. Written to be read by co-authors as the case for the experiment before the
experiment is run.*

---

## 1. What is already established, and the gap it leaves

The paper has, by this point, replaced the original "≈10% Top-1 is a fundamental
information ceiling" claim with a sharper and better-supported one. Model accuracy is
a *lower* bound set by a particular decoder; the genuine ceiling is the channel's
Bayes accuracy. On real degraded RRUFF mixtures that ceiling is modest and rank-like —
the data narrows the extinction-group (EG) candidate set without fixing the group — and
it is reached by five independent architectures (our transformer, Schopmann's ResNet-50/101,
and the XRDBench CNN11, BiGRU, and Transformer), with larger models doing no better.
The limit is therefore not architectural. It is also not symmetric-mathematical: rule-level
absence families recover 63 of 65 distinguishable absence classes, and adding Laue class
closes the rest to 99/99. What remains is *measurement*. The error decomposition makes this
quantitative: roughly 99.5% of real-model errors are cross-bin — misreads of the absence
pattern itself — and essentially none are within-bin Laue confusions. The models are not
losing to the irreducible Laue degeneracy; they are losing because the degraded measurement
does not deliver a clean absence pattern.

That leaves one question unanswered, and it is the question that turns a diagnosis into a
mechanism: **which measurement effect destroys the absence pattern, and how much information
does it cost per unit of that effect?** The degradation ladder to date (broadening/overlap,
structure-factor weighting, background, finite counting) drops reflections to "missing." But
the directional signature of the real-model errors points at something the ladder does not yet
contain. Every model's errors skew toward *lower-symmetry descendants* on the group–subgroup
graph (descendant:ancestor ratios from ≈1.9:1 for CNN11 up to ≈10:1 for BiGRU and the
Transformer). Predicting lower symmetry means inferring *fewer* systematic absences than are
truly present — i.e., forbidden reflections appearing present. The physical process that fills
in forbidden reflections is a second crystalline phase: **impurity.** The descendant skew is the
fingerprint of impurity, and impurity is precisely the nuisance the ladder omits.

## 2. The thesis

We propose to treat impurity not as noise to be minimized but as a **controlled, parameterized
variable**, and to measure the dose–response of recoverable EG information against it. This
inverts the field's standard stance — everyone purifies — and it is the stronger move
scientifically, because it converts the qualitative observation ("errors skew descendant, which
looks like impurity") into a quantitative law ("X% second phase costs Y bits / Z Top-5 points of
EG information") and, crucially, into a **reference axis on which the five real models can be
placed.** The convergent floor stops being a flat band of indistinguishable accuracies and becomes
a mechanistic ranking: how far each architecture sits from the impurity-limited optimum, and in
which direction.

The argument has three parts: that impurity is the right variable (§1, above), that there is a
defensible way to inject it (§3–4), and that doing so yields a model-placement result no single
accuracy number can (§5). The caveats that keep the claim honest are in §6.

## 3. Which impurity? The generative-model choice

"Impurity fraction" is underspecified, and the choice is not cosmetic, because it determines
*which* reflections get corrupted — and the entire mechanism depends on corruption landing on the
diagnostic reflections. Three forks have to be resolved and stated.

**Random injection versus real-phase injection.** The simplest knob flips a fraction of the host's
forbidden reflections to "present," or sprinkles spurious intensity at random 2θ. It gives a clean,
monotone dose parameter, but it is physically wrong in a way that matters here: a real impurity is a
*crystalline phase*, so its peaks fall at *its own* lattice positions, and whether they coincide with
the host's systematic-absence positions depends on the two unit cells. Real-phase corruption of
forbidden reflections is therefore sparser and structured, not proportional to a free parameter. The
mechanism — fewer apparent absences → descendant misread — operates under both models, but the
*shape* of the dose–response differs, and any diffractionist reviewing the work will ask immediately
which was used.

The resolution is a **two-tier design** that makes the result robust rather than a modeling artifact.
The random model is the *analytic knob*: it supplies a clean, interpretable dose axis for the
Bayes-ceiling curve and the optimal-decoder error signature. A small set of *real second phases* —
common ones such as quartz, calcite, a feldspar, and corundum (the last an internal-standard analog) —
at realistic weight fractions then serves as *physical validation*: generate their actual peak lists
and superpose them on the host pattern. If the real-phase points fall on the random-knob curve, the
random model is vindicated as a proxy and we report the clean curve with the real phases as anchors.
If they diverge, the divergence is itself a finding — real phases corrupt less per unit mass because
their peaks mostly miss the host's forbidden positions — and we report both. Either outcome is
publishable; what is not acceptable is choosing one model silently.

**Which forbidden reflections are eligible (within the random model).** Uniform corruption treats a
forbidden (1,0,0) and a forbidden (4,4,0) as equally likely to be filled — cleanest, least physical.
Intensity-weighted corruption makes the fill-in probability scale with how much impurity intensity
lands near a reflection's 2θ, which is higher at low angle. Overlap-gated corruption flips a forbidden
reflection only when an impurity peak falls within a peak-width of it, coupling impurity to the existing
broadening rung. We adopt **intensity-weighted corruption as the primary axis**: it is physical enough
to defend, still a clean monotone parameter, and — decisively — it stresses exactly the *low-order*
reflections that the rule mask (especially the low-order v2 redesign) depends on. That interaction is
the sharp internal test of whether moving the diagnostics to low order genuinely helps or merely trades
one fragility for another. Uniform corruption is retained as a shape sanity check; overlap-gating is
available but not primary, to avoid a second free parameter.

## 4. Parameterize by the physical quantity, not the convenient one

The dose axis must be a **second-phase weight fraction**, not "fraction of reflections flipped." The
corruption rate of forbidden reflections should be the *downstream consequence* of a weight fraction
passed through the intensity model, not the input. This matters for the payoff: it lets the optimal-decoder
error signature be parameterized by something an experimentalist controls and reports, so the conclusion
"this model behaves as if at 8% second phase" is a statement in units a crystallographer can act on.
Parameterizing directly by flip-fraction gives a marginally cleaner curve but strips the effective-impurity
readout of its physical anchor, which is the whole point. The proposed sweep is weight fraction
∈ {0, 2, 5, 10, 20%}.

## 5. What the sweep predicts, and the payoff

At each impurity level we compute three things on the same channel and prior.

**The Bayes ceiling versus impurity.** This is the decoder-independent information content. The
prediction, from the earlier analytic flip-model, is that a few percent second phase drags Top-1 from
the clean ≈20% toward the realized ≈15% regime — now to be confirmed in the full simulator with the
actual rule mask rather than asserted from a toy model. The output is a curve, not a point: the
information cost of impurity, in bits and in Top-k.

**The realized rule-mask accuracy versus impurity, for v1 and v2.** This tests the redesign against the
nuisance that most punishes it. v2 drives glide and centering "undetermined" to zero by fitting the
condition to whatever reflections survive and committing to a category — which on dropout-only channels
buys a large Top-5 gain at a small Top-1 cost. Impurity is the adversarial case for that "always commit"
behavior: a filled-in forbidden reflection is *false evidence*, and the rule-fitter will confidently fit
the wrong condition. If v2's Top-1 holds across the impurity sweep, the low-order redesign is robust and
should be promoted; if it collapses while v1 degrades gracefully, the forced-commitment default needs
softening (retain "present, type undetermined" rather than defaulting to none unless low-order reflections
positively confirm absence). The sweep is thus simultaneously the paper's mechanism experiment and the
decision bench for v2.

**The optimal-decoder error signature versus impurity — the payoff.** For a Bayes-optimal decoder on the
same channel, we compute, at each impurity level, the error signature already measured on the real models:
cross-bin error rate, descendant:ancestor skew, and crystal-system-change rate. This produces a *reference
curve* — the error structure an information-optimal reader would exhibit as impurity rises. We then match
each real model's measured signature onto that curve and read off the **effective impurity level each model
behaves as if it sees.** The convergent floor becomes a mechanistic placement:

- A model whose signature matches the optimal decoder at, say, 8% impurity is behaving near-optimally for a
  sample with that much second phase — its errors are the data's fault, not the architecture's.
- A model whose signature is *worse* than the optimal decoder at every impurity level (more scattered, more
  descendant-skewed than impurity alone explains) is leaving information on the table, and the excess
  quantifies the architectural deficit.
- The descendant-skew spread (CNN11 ≈1.9:1 versus BiGRU/Transformer ≈10:1) becomes interpretable: if the
  optimal-decoder skew at realistic impurity is ≈1.9:1, then CNN11 is near-optimal and the others over-skew
  (an uncertainty/architecture effect, weaker models falling back harder onto the low-symmetry prior); if
  it is ≈10:1, the reverse.
- **AlphaDiffract remains a useful contrast case, but no longer as a weak baseline.** After fixing the
  one-based-target bug in our direct adapter, the released checkpoint is strong on the shared `RRUFF-473`
  and exact processed-overlap `209` surfaces. So its value in this experiment is not “does the weak model
  sit off the curve,” but whether a high-performing model that omitted preferred-orientation handling lands
  on a systematically different part of the impurity-response curve than the other strong baselines.

This is a stronger statement than the floor band because it is a claim about *mechanism and distance from
optimality on a physical axis*, not a claim about which model's accuracy point estimate is highest.

## 6. What keeps the claim honest

Three caveats are load-bearing and must travel with the result.

First, the effective-impurity readout is a **behavioral** match, not a measurement of the actual impurity in
any RRUFF specimen — which varies sample to sample and is not known. The correct phrasing is "model M behaves
as if reading data at impurity level p," never "the data contains impurity p." The match places the model's
*error structure* on a reference curve; it does not assay the mineral.

Second, the optimal-decoder reference must be computed on the **same channel and prior** as the real-model
comparison. An optimal decoder evaluated on a different prior or a different nuisance stack would make the
placement apples-to-oranges. This is a discipline already enforced elsewhere in the paper (the rule-mask
comparison runs v1 and v2 through one identical bound-safe scorer) and it applies here without exception.

Third, the random corruption model is justified *only* to the extent the real-phase validation points
(quartz, calcite, feldspar, corundum) fall near its curve. If they do not, the random curve is reported as an
upper bound on impurity sensitivity and the real-phase points carry the physical claim. The validation is not
optional decoration; it is what converts "we chose a corruption model" into "the corruption model is
justified."

## 7. How it strengthens the paper

The spine of the paper is a four-way separation of limits the field routinely conflates: symmetry mathematics
(solved — rule families plus Laue reach 99/99), model capacity (not binding — architecture moves error
*structure*, not the accuracy floor), Laue degeneracy (real but not yet in contact with the models — ≈0.5% of
errors), and measurement degradation (the actual binding constraint — ≈99.5% of errors). The impurity sweep
supplies the missing quantitative core of the fourth limit: it names the dominant single nuisance, measures its
information cost as a dose–response, and — by placing all five real models on that physical axis — demonstrates
that the convergent floor is an impurity-limited measurement floor, with each architecture's distance from the
information-optimal reader made explicit. The original "10% is fundamental" claim is then replaced by something
both more accurate and more actionable: the extinction-group information is largely present in the data, what
destroys it is impurity-dominated measurement degradation, here is how much per unit second phase, and here is
where each model sits relative to the optimum.
