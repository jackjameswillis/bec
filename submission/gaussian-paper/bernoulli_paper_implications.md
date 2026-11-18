# What the new literature does to the Bernoulli paper

Date: 2026-11-18
Context: written after the literature sweep recorded in `literature.bib` /
`literature_notes.md`. This note is about the *companion* paper
(`../bernoulli-paper/total.tex`), not about the Gaussian draft.

Read the Bernoulli paper's current text before acting on any of this; section
references below are to `total.tex` as of this date.

---

## 1. The structural relation is already there

The Bernoulli paper promises the Gaussian paper. From its Discussion (§7):

> A potential means of removing this issue is to simply take the inertial
> interpretation of continuous parameters [Helwegen et al.], and to optimize
> them with a normal search distribution. Sampling from the distribution then
> becomes simple floating point addition, and quantization with the sign
> function.

So the pair is coherent as a pair, and the Gaussian paper resolves a limitation
the Bernoulli paper states explicitly (§7: "there are no continuous
parameters", the resulting bias-optimisation and mixed-precision problem).

**Implication.** Say this out loud in both introductions. It makes the two
papers read as a programme rather than two papers that happen to share numbers
and architecture.

---

## 2. Where the literature bites

### 2.1 The origin story is now occupied ground

The Introduction hangs the paper on:

> Notably, in their work they [Meng et al.] suggest that a REINFORCE gradient
> estimate could be used to learn these expectations, but do not explore this
> idea any further.

True when written; not true now. Several groups have taken that exact
invitation in 2025–2026, mostly for LLM fine-tuning:

- `shang2025qzo` — QZO: zeroth-order fine-tuning of quantised networks.
- `yang2025fogzo` — FOGZO: STE gradient augmented with zeroth-order
  information; the clearest statement of the bias/variance trade.
- `xu2026qes` — QES: full-parameter fine-tuning in quantised space with
  accumulated error feedback.
- `rakotoarivony2026esc`, `pietron2026evofinetune` — ES for quantisation
  calibration and post-quantisation repair.

The abstract's framing ("an alternative approach ... could be a
gradient-approximation algorithm such as an evolutionary strategy") reads as a
proposal for unclaimed territory. It needs a hedge: *explored since, but as
fine-tuning of already-quantised models, not as from-scratch training with a
distribution over binary networks.*

### 2.2 Okada 2021 is an unaddressed prior-art risk

`okada2021es` applies ES directly to binary network parameters (and the series
continues with GA, PSO and differential evolution on the same problem). The
Bernoulli paper — which is ES directly on binary parameters — does not cite
him. With an EC audience at EvoStar, this is the most likely
"didn't-cite-the-obvious-ancestor" note a reviewer could write.

Room to differentiate, on the evidence available:

- His search is a generic mutation/recombination ES over bit strings; ours is a
  logit-parameterised Bernoulli with an analytically derived score function
  (Eq. 12–13 of the current draft), so the update is the exact gradient of the
  expected-fitness objective rather than a heuristic move.
- No fitness shaping, no population-scaling analysis, no entropy diagnostics,
  no STE comparison, smaller networks.
- He reports {−1,+1} beating {0,1}, which is a free corroborating citation for
  the sign map in §2.1.

### 2.3 Discrete NES is adjacent and uncited

`amin2024discrete_nes` (Discrete Natural Evolution Strategies, 2024). The
Discussion claims the paper deviates from "much of the prior work on
evolutionary strategies that learn the mean of uni-variate normal
distributions"; there is now a discrete-case NES in the literature and it is
not mentioned.

**Status: metadata only.** I have read the title and metadata, not the method.
Do not cite it for a specific claim until someone reads it (see §5).

### 2.4 A theory anchor independent of Meng

`soudry2014expectation` — Expectation Backpropagation. A mean-field/EP posterior
over binary weights produces per-unit activation probabilities of probit form,
Φ(μ/σ). The Bernoulli posterior the paper parameterises is the same object; so
the Bernoulli ES is the score-function cousin of that posterior, and the
Gaussian ES is its probit parameterisation.

This is a stronger lineage than "we took up Meng's unexplored suggestion",
because it does not depend on one 2020 paper's aside.

---

## 3. New light on the paper's results — the part worth building on

The quantised-ZO literature has a crisp account of why optimisation stalls on
discrete parameters. QES §5 splits the trajectory into an ideal continuous
update plus *accumulated quantisation loss*, with two failure modes:

1. If the update is smaller than the grid spacing, quantisation cancels it
   exactly → stagnation.
2. If quantisation is stochastic, the errors random-walk → a noise floor.

The Bernoulli paper reports the empirical shadow of this, but the mechanism in
its own parameterisation is different, and neither paper currently says so.

**Bernoulli.** The score is `x − s`. As a coordinate commits, `s → 0` or `1`
and the sampled `x` matches it almost surely, so `x − s → 0`. The
per-coordinate update *vanishes by construction*. Committed coordinates stop
moving.

**Gaussian.** The score is `ε/σ`, magnitude ~1 regardless of how committed θ
is. Committed coordinates keep receiving full-magnitude updates; what decays is
the probability that the perturbation flips the bit at all, `Φ(−|θ|/σ)`.

So the two searches fail differently:

- Bernoulli: frequent perturbations that stop carrying signal.
- Gaussian: full-strength signal that fires ever more rarely.

### Consequences for the Bernoulli paper

**Testable prediction.** Entropy curves and per-coordinate update magnitudes
should have visibly different decay shapes between the two algorithms. This is
a cheap addition to the pair and it gives the comparison teeth.

**Re-reads two existing results.** Why the output layer saturates first: it has
the fewest coordinates and sits closest to the fitness signal, so it commits
earliest — and then, because its score has collapsed to zero, it *freezes*
there. What the draft calls layer-by-layer convergence may be partly a lock-in
artefact rather than convergence. That would also explain the volatile
validation accuracy that persists after the mode network looks converged: the
output layer is committed and cannot adapt to what the hidden layers do
afterwards.

This converts an observation (Fig. 3) into a mechanism, and it is a reason for
the pair to exist beyond "same thing, two parameterisations".

---

## 4. Smaller points

**Population scaling.** EGGROLL's (`sarkar2025eggroll`) high-dimensional
Gaussian-ES theory supports the direction of the population-size result, but
the caveat survives: the sweep holds *iterations* fixed, so larger P costs more
compute. The newer literature (MeZO, QES) uses *small* populations, which cuts
against a naive "bigger is better" reading. State the equal-iterations
qualification explicitly.

**Helwegen.** "Latent weights do not exist" is an argument against reading
latent magnitude as an importance proxy. In this pair the latent *is* the
search state, so the critique does not bite — the papers can say so rather
than inherit it.

**The STE comparison needs a compute number.** Parity with Adam-STE
(Table 1: 97.87 vs 98.02 test) is the paper's headline comparison, but STE is a
biased, low-variance, one-backward-pass-per-step estimator while the ES is a
high-variance, P-forward-passes-per-step estimator. The Discussion mentions
compute cost; the new literature supplies precise vocabulary for it. One
sentence quantifying the multiple would pre-empt the obvious objection.

**The real differentiator.** Not memory (QZO/QES/MeZO all beat the ES there) but
*no backward pass at all* — black-box fitness, arbitrary non-differentiable
objectives. Keep that as the selling point; the memory-efficiency framing puts
this work in a fight it loses.

---

## 5. Open items before editing any text

1. **Read Okada 2021 properly** (IJSRS-CE 9(1):31–35) and confirm what the ES
   operator actually is, so §2.2's differentiation is accurate rather than
   convenient. The journal page was accessible via a browser fetch; the PDF
   should be too.
2. **Read Discrete NES** (`amin2024discrete_nes`, arXiv 2404.00208) and decide
   whether it is a citation, a contrast, or a threat to the novelty claim.
3. **Decide the hedge wording** for the abstract and Introduction, given
   §2.1 — the smallest edit that stays honest without inviting the reviewer to
   ask why the paper exists.
4. **Decide whether the two papers share a related-work section** or each carry
   their own. Given QES, the Gaussian paper's related work is the harder one and
   should probably be written first.
