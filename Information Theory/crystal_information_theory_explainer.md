# Reading Crystal Symmetry Through a Noisy Channel
### What we are attempting, and what we have found so far
*Written for someone who can program but has never seen information theory.*

---

## 1. The puzzle

### A little more background: crystals, diffraction, and why anyone cares

A **crystal** is matter whose atoms are stacked in a pattern that repeats in all
directions, like an enormous 3D wallpaper. Almost all solid inorganic materials —
metals, minerals, battery electrodes, semiconductors — are crystals. *How* the
atoms are arranged determines what the material does: whether it conducts, what
color it is, how strong it is, whether it's magnetic. So if you've made a new
material and want to know what you've got, step one is figuring out its atomic
arrangement. That's the central measurement problem of materials science, and it's
a bottleneck: labs can now *make* candidate materials (and AI can now *propose*
them) far faster than anyone can determine their structures.

The standard tool for determining structure is **diffraction**. You send a beam of
X-rays or neutrons at the sample. The beam bounces off the regularly-spaced planes
of atoms, and because the spacing is regular, the reflected waves reinforce each
other only at specific angles — the same reason a CD throws rainbow stripes under
light. Detect the beam coming out and you get peaks at those special angles. If
your sample is a single perfect crystal you get a 2D constellation of spots; if
it's a *powder* (millions of tiny crystals in random orientations, which is much
easier to prepare) all those spots smear into rings, and you record their
intensity as a 1D curve — the **barcode** from the introduction. Powder data is
cheaper and more common, but squashing 3D information into 1D throws a lot away,
which is exactly why pulling structure back out of it is hard.

### The specific question

We don't try to recover the full atomic arrangement here — just one informative
piece of it, the **extinction group**: which of **99** symmetry categories the
crystal falls into. The reason this is readable at all comes down to a clean
physical fact. When a crystal has certain internal symmetries — say, a screw-like
twist or a glide that shifts-then-reflects the atoms — the scattered waves from
the symmetry-related atoms arrive exactly out of phase along certain directions
and **cancel to zero**. The corresponding peaks are not faint; they are forbidden.
These are **systematic absences**, and which peaks are forbidden depends on which
symmetries are present. So the *pattern of missing peaks* is a fingerprint that
narrows 99 possibilities down toward the right symmetry class. Knowing it shrinks
the search space for the full structure enormously.

A neural network was trained to look at the 1D barcode and guess the extinction
group. (You don't need to know how the network works internally; treat it as a box
that reads the barcode and outputs a probability for each of the 99 groups.) Its
single best guess is right about **10%** of the time — "Top-1 accuracy." A recent
paper looked at that 10% and suggested it might be a *fundamental limit* of powder
diffraction: that no method could do much better, because the data simply doesn't
carry the information.

That claim is what set off this whole project. We suspected it confused two very
different things: "our particular model tops out at 10%" and "10% is the most any
method could ever achieve." To tell those apart we needed a way to measure
*information itself* — not the accuracy of one model, but how much is actually
there to extract. That is exactly what information theory is for.

---

## 2. A crash course in information theory

Information theory was invented by Claude Shannon in 1948 to answer a practical
question: how much data can you push through a noisy telephone wire? The ideas
turn out to apply to *any* situation where you are trying to recover an unknown
answer from a noisy observation — including reading crystal symmetry off a
barcode.

**Bits = yes/no questions.** Suppose I'm thinking of a number from 1 to 8 and you
have to guess it with yes/no questions. The smart strategy is binary search: "Is
it ≤ 4?" cuts the possibilities in half each time. You need 3 questions, because
8 = 2³. In general, pinning down one of *N* equally likely options takes
**log₂(N)** questions. We call that number of questions **bits**. One bit is one
perfect yes/no answer.

**Surprise.** Not all answers are equally likely, and rare answers are more
*surprising*. Shannon measured the surprise of an event of probability *p* as

```
surprise = log2(1 / p)   bits
```

A coin flip (p = 1/2) carries log₂(2) = 1 bit of surprise. Something you were 99%
sure of carries almost no surprise. Something with p = 1/1000 carries ~10 bits —
learning it told you a lot.

**Entropy = average surprise = uncertainty.** If a quantity *G* takes many values
with various probabilities, its **entropy** *H(G)* is the average surprise over
all outcomes — equivalently, the average number of yes/no questions you'd need to
identify it. If all 99 extinction groups were equally common, the entropy would be
log₂(99) ≈ **6.6 bits**. But they are *not* equally common — some symmetries are
far more frequent in real minerals than others — so the real uncertainty is lower,
about **4.7 bits**. That number, 4.7 bits, is our starting ignorance: how confused
we are about the extinction group *before we look at the barcode at all*.

**The prior.** That "before looking" guess has a name: the **prior**. It's just
the histogram of how common each group is. A lazy predictor that ignores the
barcode entirely and always names the most common groups is called a *prior-only*
baseline. Crucially, the neural net's 10% Top-1 is *barely above* what this lazy
histogram already gets. That's the first clue that "10%" might be telling us about
the prior, not about the physics.

**Mutual information = bits the data gives you.** Now suppose you look at the
barcode *X*. If it's informative, your uncertainty about the group drops — from
*H(G)* before, to some smaller *H(G | X)* after. The amount it dropped is the
**mutual information**:

```
I(G ; X) = H(G) - H(G | X)
```

This is the heart of everything. It is, literally, *how many yes/no questions the
measurement answered for you*. If the barcode pins down the group exactly,
*I(G;X)* equals the full 4.7 bits. If the barcode is useless noise, *I(G;X)* = 0.

**The channel.** Here is Shannon's picture, applied to us. Think of the true
extinction group *G* as a **message** someone is trying to send you. It travels
through a **noisy channel** — the physical measurement — and comes out the other
end as the barcode *X*. The "noise" is everything that corrupts the signal:
background hiss, peaks blurring into their neighbors, contamination from other
crystals, and so on. The whole project is to ask: *how many bits about G survive
the trip through this channel?*

**Capacity vs. one decoder — the distinction the paper missed.** A specific
method (like our neural net) is one **decoder**. Its accuracy tells you how many
bits *that decoder* recovered — which is a **lower bound**: a smarter decoder might
recover more. The **capacity** of the channel is the *maximum* bits *any* decoder
could ever recover. Capacity is a real ceiling; one model's accuracy is not.
Saying "our model gets 10%, therefore 10% is the limit" is like saying "I couldn't
open the jar, therefore the jar is unopenable." That confusion is precisely what
we set out to fix.

---

## 3. What we are attempting

Instead of defending "10% is a fundamental limit," we reframe the question as
channel decoding and measure the bits carefully, keeping three things separate:

1. **Prior-only** — what you know from the histogram alone (no barcode).
2. **Model-extracted bits** — what our actual neural net pulls out. A *lower
   bound* on what's possible.
3. **Channel capacity** — the true ceiling. We can only measure this in
   *simulation*, where we generate fake crystals so we know the right answer and
   can compute the channel directly.

We also separate **two kinds of "knowing the answer"**, and this distinction runs
through every result below:

- **Mass:** putting high probability on the *exact* correct group (getting #1
  right).
- **Rank:** pushing the correct group high on a *short list* of candidates (e.g.
  into the top 5), even if it isn't #1.

These are genuinely different skills. A model can be good at narrowing 99 options
down to a believable shortlist while still rarely nailing the single winner.

Finally, we run the experiment in **gated stages**: do the cheapest, most decisive
test first, and only build more machinery if it passes. Gate 1 uses real data and
the model we already have. Gate 2 is a minimal simulation. A future Gate 3 would
be a full physical simulation. No point building the cathedral before checking the
foundation holds.

---

## 4. What we found — Gate 1 (real data)

We scored the model's predictions two ways, matching the two kinds of knowing.

**The mass test (did the model beat the prior at putting probability on the true
group?).** We compare the model's *surprise* at the true answer to the prior's
surprise at the true answer. If the model is genuinely informative, it should be
*less surprised* by the truth than the histogram is. The metric
`bits_over_prior` is exactly this difference, averaged over all test patterns.

The result was **negative**: about **−0.61 bits** (and still −0.52 even after we
retuned the model's confidence to its best possible setting). Negative means that,
on average, the model is *more* surprised by the true group than the dumb
histogram is. The reason shows up in the details: on most patterns the model
confidently piles probability onto a *wrong-but-similar* group, leaving little for
the truth. It is **confidently wrong** more often than it is right. So the
strict, mass-based version of "the model extracts information" **fails**.

**The rank test (did the model narrow the candidate list?).** Here we ask a softer
question: does the true group land in the model's top-5 shortlist more often than
chance? Answer: **yes, modestly but consistently.** Top-5 accuracy is 43.1%, and
the model's average rank for the true group beats chance with a p-value around
0.035–0.04 (and better, ~0.007, after retuning). Top-1, by contrast, does **not**
beat chance.

How do we know "more often than chance" is real and not luck? With a **shuffle
test** (a permutation null), which is the same trick a programmer uses to check
for spurious correlations: scramble the answer key, re-score, and see how well the
model does *by accident*. A model that ignores the barcode scores the same
shuffled or not. Our model beats its own shuffled self on the rank tests — so it
*is* using the barcode.

**An important honesty check on the baseline.** You might think "43% top-5 sounds
great." But on this particular benchmark, the lazy "always name the 5 commonest
groups" strategy *already* scores 43.7% — slightly *higher* than the model. So the
model does **not** beat the trivial baseline on raw top-5 numbers. What it beats is
the **shuffle null**, which specifically destroys per-pattern information. That is
a narrower, more honest claim: the model is **not a flat prior-ranker** — it
genuinely adjusts its shortlist pattern by pattern — but the *amount* of
pattern-specific information is small.

**Gate 1 verdict.** "Top-1 ≈ prior" does **not** mean "the model learned nothing."
The model extracts **real but narrow, rank-like information**: it nudges the true
group up the shortlist using the barcode, even though it rarely nails #1 and even
though its confident mass often lands on a near-miss. The paper's "10% =
fundamental limit" is the wrong description. The right one is: *a little real
information of a specific kind survives, and it lives in the ranking, not in the
top slot.*

---

## 5. What we found — Gate 2 (simulation)

On real data we can only ever get a lower bound, because we only have one decoder
(the model). To estimate the channel's **capacity** — the true ceiling — we need
to know the right answer for every pattern, so we **simulate** crystals.

We boiled the barcode down to the part that physically carries symmetry: an
**absence mask**. For six families of reflections, we record whether each is
`present`, `absent`, or `unresolved` (too weak or too blurred to call). That mask
is the fingerprint; the channel is "true group → observed mask."

**The first result looked like a disaster.** The estimated capacity was about
**0.006 bits** out of a possible 6.6 — essentially zero — and the mask seemed to
narrow 99 groups down to... 98.7. As if the fingerprint told you nothing.

**But it was a measurement artifact, not a real zero — and catching that is the
whole lesson.** We were trying to estimate a table with ~600 possible mask
fingerprints across 99 groups, from only **198 examples** — about **2 per group**.
This is the classic overfitting-in-reverse problem: with a huge space of
possibilities and almost no data, your estimate is dominated by the placeholder
"smoothing" values you add to avoid dividing by zero, and those placeholders carry
no information. We measured the smoothing, not the channel.

We proved it was an artifact, not a dead mask, with three checks:

- **The mask itself varies a lot** (its entropy is ~3.9 bits), so it is *not*
  constant — it's not that every crystal looks the same.
- **The measured information rises as we add data** (1.22 → 1.73 → 2.21 bits as
  the sample count goes 49 → 99 → 198). A genuinely empty channel would stay flat
  near zero; a real one that's merely under-sampled climbs as you feed it.
- **A shuffle test exposes how much is still illusion.** Here's the subtle part:
  even with the answer key *scrambled*, this tiny dataset scores ~1.9 bits, purely
  because random coincidences look like structure when you have 600 columns and 2
  rows per category. So of the apparent 2.2 bits, only about **0.3 bits** is real
  signal above chance — and because we're so starved, even that 0.3 is an
  *under*-estimate of the truth.

**Gate 2 verdict.** The machinery works end to end, and the absence mask **does**
carry real group information (it beats its shuffle null). But the *number* is not
yet trustworthy — it's mostly finite-sample noise. The fix is **not** to redesign
the mask; it's to **generate far more simulated crystals** (hundreds per group,
not two) and use gentler smoothing, then re-measure. Simulation is cheap, so this
is straightforward — it just hasn't been run yet.

---

## 6. The big picture

The same trap appeared at both gates, and it's worth naming: **an apparent zero is
not the same as a real zero.** At Gate 1, "Top-1 ≈ prior" looked like "no
information," but the information was real and hiding in the ranking. At Gate 2,
"capacity ≈ 0" looked like "useless mask," but the zero was an artifact of too
little data. Information theory is what lets us tell a true zero from an apparent
one — through lower bounds, shuffle nulls, capacity, and bias correction.

So where do we stand? The claim we can actually defend is deliberately modest, and
much stronger for being honest:

> The model is **not** a flat prior-ranker. On a per-pattern basis it narrows the
> extinction-group candidate set, and that effect is statistically real (it beats
> a label-shuffle null on top-5 and on average rank). But the surviving
> information is **narrow and rank-like**: the full calibrated model does **not**
> beat the prior at placing probability on the exact group, even at its
> best-tuned confidence. This is consistent with genuine physical ambiguity that
> permits narrowing the neighborhood of possible symmetries but not pinning down
> the single answer — and it is *not* evidence for a universal "10% ceiling" of
> powder diffraction.

**Next steps.** Re-run Gate 2 with hundreds of simulated crystals per group to get
a trustworthy capacity number, then build the full physical "ladder" — adding
realistic corruptions one at a time (blurring, background, contamination,
preferred orientation) to see *which* ones destroy the symmetry information and
how fast. That will turn today's qualitative story ("real but narrow") into a
quantitative map of where the bits go.

---

## Mini-glossary

- **Bit** — one yes/no question's worth of information; log₂(N) bits pin down one
  of N equally likely options.
- **Surprise** — log₂(1/p) for an event of probability *p*; rare = surprising.
- **Entropy H(G)** — average surprise of *G*; your uncertainty before measuring.
- **Prior** — the "before looking" histogram of how common each answer is.
- **Mutual information I(G;X)** — bits of uncertainty the measurement removes;
  = H(G) − H(G|X).
- **Channel** — the noisy process turning the true answer *G* into the
  observation *X*.
- **Capacity** — the maximum bits *any* decoder could recover; a true ceiling.
- **Lower bound** — what one specific method recovers; a floor, not a ceiling.
- **Shuffle (permutation) null** — scramble the answer key and re-score, to see
  how much of your result is luck.
- **Systematic absence** — a peak that symmetry forces to be exactly zero; the
  pattern of absences fingerprints the extinction group.
