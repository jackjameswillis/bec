# Literature map — Gaussian ES paper

Companion to `literature.bib` (45 entries, compiles clean with `bibtex`).
Scope: literature found 2026-11-18 for "A Gaussian Evolutionary Strategy for
Binary Weight Neural Network Optimization". `refs.bib` already holds the 18
citations the draft inherits from the Bernoulli paper; this file tracks what
was found *beyond* those, plus notes on how each group bears on the argument.

Sources used: OpenAlex (keyword sweeps), arXiv (API + listing search),
Crossref (DOI verification), journal pages. Every entry's metadata was pulled
from Crossref/arXiv rather than typed, and the whole file was compiled with
bibtex to check it parses.

---

## A. Binary neural networks: training and optimisation (8)

The baseline literature: what BNN training looks like, and what is wrong with it.

- `qin2020bnnsurvey` — the standard survey; good for the "BNNs are memory-cheap,
  training is the hard part" framing.
- `liu2020real2binary` — the standard practical recipe; useful as the "what a
  well-tuned STE pipeline looks like" reference.
- `chen2021bnnbn` — BNN without batch norm; connects to the `scale` layer choice
  in `bes/mlp.py`.
- `shan2023sgdat` — a recent optimiser variant for BNNs; shows the field is
  still iterating on optimiser design.
- `yin2019ste` — the standard theoretical account of STE bias.
- `shekhovtsov2021reintroducing` — recasts STE as a principled stochastic-binary
  estimator. Closest thing to "STE is an estimator, not a hack".
- `ichikawa2025highdim` — 2025 learning-dynamics analysis of quantised models
  under STE. Useful if we want to say anything about dynamics.
- `colombo2024localbinary` — a fully binary, gradient-free BNN training rule
  (local binary error signals, integer hidden metaplastic weights). This is the
  nearest "no floating-point gradients" competitor and is biologically framed.
- `huang2026surge` — 2026, learnable gradient compensation for BNNs. Evidence
  the STE-bias thread is still active.

## B. Low-precision training at scale (5)

Why the problem is current, not a toy.

- `gupta2015limitedprecision` — stochastic rounding / limited precision training;
  the origin of "quantisation error accumulates like a random walk".
- `li2016ternary`, `zhu2017trainedternary` — the ternary line, still the
  practical middle ground.
- `wang2023bitnet`, `ma2024bitnet158` — 1-bit and 1.58-bit LLMs. This is what
  makes "can you train on discrete weights directly?" a live question again,
  and it is the frame QES/QZO sit in.

## C. Evolution strategies and neuroevolution (7)

Where the estimator comes from.

- `hansen2016cmaes` — the canonical ES reference; establishes the
  "distribution over parameters" language we use.
- `liu2019trustregion` — ES as a trust-region/surrogate method; relevant if a
  reviewer asks why we do not reuse samples across generations.
- `cui2018esgd` — ES mixed with SGD; the hybrid corner of the space.
- `jaderberg2017pbt` — population-based training; the "populations are a
  practical tool" citation.
- `galvan2021neuroevosurvey` — neuroevolution survey; venue-appropriate.
- `amin2024discrete_nes` — Natural ES for *discrete* parameters. Directly
  adjacent: a Gaussian-in-continuous-space estimator for a discrete problem,
  which is precisely our construction.
- `qiu2025esscale` — ES for LLM fine-tuning at scale, 2025. Same group that
  wrote QES; shows the ES-at-scale programme is ongoing.

## D. Prior art: evolutionary training of binary / quantised weights (4)

The papers that mean we cannot claim novelty for "use an EA on binary weights".

- `okada2021es` — ES applied directly to binary network parameters; finds
  {−1,+1} far better than {0,1}, and that sufficient hidden width matters.
  This is the direct ancestor of our Bernoulli paper, not of this one.
- `okada2021ga` — the GA twin of the same study (part of a four-paper series
  incl. PSO and differential evolution).
- `shibuya2025binaryflip` — 2025 ICASSP: stochastic bit-flip optimisation for
  BNN training. Closest recent "discrete move" method.
- `he2025probabilistic` — 2025 Neurocomputing: probabilistic optimiser for BNNs.

None of these use a continuous latent with sign-quantisation-at-evaluation, and
none connect to the STE optimiser stack. That is the gap our paper occupies.

## E. Zeroth-order and ES fine-tuning of quantised models (6)

**The critical section.** This is the literature that arrived independently at
"perturb, quantise, estimate" in the last 18 months.

- `xu2026qes` — **Quantized Evolution Strategies** (Xu, Schweighofer,
  Miikkulainen, Qiu; arXiv 2602.03120, Feb 2026). Full-parameter fine-tuning
  directly in quantised space, with (i) accumulated error feedback to keep a
  high-precision update signal and (ii) stateless seed replay for memory.
  §5 "Temporal Equivalence to Continuous Optimization" explicitly names the
  Gaussian-smoothed objective J_σ and the unbiased estimator, then argues
  unbiasedness is not sufficient on the grid because of *accumulated
  quantisation loss*. Our paper is the complementary answer to exactly that
  problem: keep the search variable continuous and quantise only for
  evaluation, so the accumulated-loss term never exists.
- `shang2025qzo` — QZO: zeroth-order fine-tuning of quantised NNs by perturbing
  the quantisation scale rather than the weights.
- `yang2025fogzo` — FOGZO: combines STE (biased, cheap) with ZO (unbiased,
  expensive); the clearest statement of the bias/variance trade this paper
  lives in.
- `malladi2023mezo` — MeZO; the reference point for "forward passes only,
  memory ≈ inference".
- `rakotoarivony2026esc` — ES used to calibrate activation scales for low-bit
  speech models. ES as a *calibration* tool rather than a trainer.
- `pietron2026evofinetune` — ES to improve post-quantisation accuracy by
  optimising the discrete assignment. Again ES post-hoc, not as the trainer.

Pattern: ES/ZO on quantised models is being used for (a) fine-tuning, (b)
calibration, (c) repair. Nobody is treating it as the *primary training
algorithm with an unbiased gradient of a smoothed objective and a full Adam
stack*, which is our claim. But the space is filling fast — QES is eight months
old and from a well-known group.

## F. Gradient estimation theory (9)

What the estimator actually is, and what is known about it.

- `mohamed2020mcgrad` — the taxonomy paper (score function vs pathwise vs
  measure-valued). The right citation for "our estimator is the score-function
  estimator of a Gaussian, applied to a quantised forward pass".
- `nesterov2015randomgradfree` — Gaussian smoothing gives an unbiased gradient
  of the smoothed function; the theory that justifies calling J_σ smooth.
- `berahas2021comparison`, `berahas2019linear` — careful finite-difference vs
  Gaussian-smoothing comparisons. **Note the 2019 result: linear interpolation
  gives better gradients than Gaussian smoothing** in derivative-free
  optimisation. That is a genuine vulnerability for a paper whose σ is the only
  smoothing knob, and worth pre-empting.
- `spall1992spsa`, `spall1997onemeasurement` — SPSA. Our per-sample estimator
  is a one-sided simultaneous-perturbation estimator; SPSA's variance analysis
  is the natural place to point for "why P samples, why this 1/σ scaling".
- `jang2017gumbel`, `maddison2017concrete` — relaxation-based alternatives to
  score-function estimation for discrete variables. The obvious "why not
  Gumbel-softmax?" reviewer question, and our answer is that we want an
  unbiased estimator of the *true* discrete objective, evaluated on real binary
  networks, not a relaxed surrogate.
- `baydin2022forward` — forward gradients: unbiased gradient from directional
  derivatives along random directions. Structurally the same trick, applied to
  a differentiable network; worth one sentence in related work.

## G. Perturbation-based and biologically plausible learning (5)

Ancestry that matters more for a bio-inspired venue than "ES for RL".

- `jabri1992weightperturbation` — weight perturbation learning for analog VLSI.
  Our update *is* a weight-perturbation rule with a quantised forward pass.
- `werfel2005learningcurves` — node perturbation as stochastic gradient descent;
  gives the variance/learning-curve theory for perturbation rules.
- `fiete2006gradient` — dynamic perturbation of conductances in spiking nets.
- `soudry2014expectation` — Expectation Backpropagation. **Key technical
  resonance**: the mean-field/EP posterior over binary weights yields per-unit
  activation probabilities of probit form Φ(μ/σ) — exactly the
  Bernoulli(Φ(θ_j/σ)) distribution induced by our Gaussian latent. This is the
  principled reason a Gaussian latent is the natural parameterisation, and it
  links our algorithm to a Bayesian rather than purely evolutionary lineage.
- `baldassi2016fewbits` — few-bit synaptic precision suffices; supports the
  premise that discrete weights are not a lossy compromise.

## H. Tooling (1)

- `bannink2020larq` — Larq, for the STE baseline's reproducibility.

---

## What I did not find (honest gaps)

- **No controlled comparison of latent-continuous ES vs direct-Bernoulli ES on
  the same architecture and budget.** Nothing in the sweep does this. Our two
  papers together can, and that is arguably the strongest single claim
  available to this paper.
- **No population-scaling study for ES on binary/quantised networks.** EGGROLL
  (already in `refs.bib`) gives high-dimensional Gaussian-ES theory; nobody has
  swept P on a BNN objective.
- **No measurement of "commitment" as |θ|/σ growth.** Helwegen et al. argue
  latent magnitude carries no information ("latent weights do not exist"); our
  framework can answer that directly, because in our construction the latent
  *is* the search variable and commitment is a measurable dynamical quantity.
  I found no paper doing this.
- **Citation-graph expansion was not done.** Semantic Scholar and OpenAlex both
  rate-limited me out (429) partway through, so coverage is keyword-based. A
  forward/backward crawl from `helwegen2019latent`, `meng2020bayesian`,
  `xu2026qes`, `colombo2024localbinary` would harden it.

## Three positioning consequences

1. **QES is the paper to beat in the related-work section.** It is the closest
   work, it is recent, and it states the smoothed-objective argument we rely on
   — but it stays on the discrete grid and pays for it with error feedback,
   whereas we stay continuous and pay with variance instead. Say that
   explicitly rather than hoping the reviewer does not notice.
2. **The Bernoulli/Gaussian pair is the contribution**, not either paper alone:
   same black-box fitness, same architecture, same budget, two parameterisations
   of the same Bernoulli family (logit vs probit). That is a clean experimental
   design and it makes the "induced p_j = Φ(θ_j/σ)" remark load-bearing rather
   than decorative.
3. **The unbiasedness claim needs a variance footnote.** The literature (FOGZO,
   Berahas et al., SPSA) is clear that unbiased estimators of this type are
   expensive; the honest framing is "unbiased and Adam-compatible, at a
   variance cost that population size buys down", which is exactly what the P
   sweep measures.
