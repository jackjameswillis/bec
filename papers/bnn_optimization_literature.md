# Literature Base — Optimisation of Binary Neural Networks

Working draft compiled 2026-10-30 for the paper "Evolutionary Strategies for
Binary Neural Network Optimization" (`/data/bes/DESIGN.md`).

Companion material: `/data/reinforce_maths.tex` (REINFORCE maths for the soft
Bernoulli ES), `/data/comb-exp/essays/es_derivation.md` (ES derivation),
`/data/bnn_multiclass_loss_summary.md` (BNN loss-function background).

Sources used: arXiv API (`export.arxiv.org/api`), Semantic Scholar
(`api.semanticscholar.org`), local reads of `/data/ste`, `/data/comb-exp`, and
the larq package (`/data/ste/venv/lib/python3.11/site-packages/larq`).

---

## 1. Foundations: binarised networks and straight-through estimation

### 1.1 Binarized Neural Networks: Training Deep Neural Networks with Weights
and Activations Constrained to +1 or −1 — arXiv:1602.02830 — Courbariaux,
Hubara, Soudry, El-Yaniv, Bengio — 2016

- Summary: the foundational BNN paper. Binary weights and activations at
  inference; at training time the binary values are used to compute parameter
  gradients (straight-through style); forward pass is binary. Near state of the
  art on MNIST, CIFAR-10 and SVHN (Torch7 and Theano).
- Bengio's contribution: Bengio is a co-author on this paper — this is the work
  the user is referring to as "Bengio's work". The training-time/forward-time
  split is precisely the STE idea that `/data/bes/bes/ste.py` and
  `/data/ste/train_mnist_bnn.py` implement through larq `QuantDense` with
  `ste_sign` kernel quantizer and `weight_clip`.
- Locked semantics in this repo (from larq `SteSign`): `q(x) = −1 if x < 0 else
  +1`, gradient clipped at `clip_value` (default 1.0). Weights sign-mapped
  `z = 2x − 1 ∈ {−1, +1}^D`.

### 1.2 XNOR-Net: ImageNet Classification Using Binary Convolutional Neural
Networks — arXiv:1702.05987 — Rastegari, Ordonez, Redmon, Farhadi — 2016

- Summary: scaling factor α applied after binarisation (BinNorm → binarise →
  scale by mean absolute value of the pre-binarisation activation). The canonical
  fix for the missing-magnitude problem in sign-binarised networks.

### 1.3 DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with
Low Bitwidth Gradients — arXiv:1606.06160 — 2016

- Summary: low-bitwidth (k-bit) forward and backward passes; STE-like gradient
  machinery at low precision.

### 1.4 Deep Learning with Limited Numerical Precision — arXiv:1509.03421 — Vanhoucke
et al. — 2015

- Summary: motivation for low-precision training; background for why binarised
  and low-bit networks need dedicated optimisation.

## 2. Rethinking BNN optimisation: latent weights and Bop

### 2.1 Latent Weights Do Not Exist: Rethinking Binarized Neural Network
Optimization — arXiv:1906.02107 — Helwegen, Widdicombe, Geiger, Liu, Cheng, Nusselder
— 2019, NeurIPS 2019 — **BOP**

- Summary: argues that the real-valued "latent weights" used during BNN training
  are not network weights at all, but mainly act as inertia. Introduces Bop, the
  first optimizer specifically designed for BNNs / Binary Weight Networks:
  exponential moving average of gradients with decay γ, weight flip when the
  EMA signal exceeds a threshold; effectively latent-free.
- Demo: CIFAR-10 and ImageNet.
- Implementation located in larq (`larq.optimizers.Bop`, predicate
  `Bop.is_binary_variable`, used via `CaseOptimizer`); defaults threshold = 1e-8,
  gamma = 1e-4 (example configurations use threshold = 1e-6, gamma = 1e-3).

### 2.2 How Do Adam and Training Strategies Help BNNs Optimisation? — arXiv:2106.11309 — Liu,
Shen, Li, Helwegen, Cheng — 2021

- Summary: best BNN results are often obtained with Adam and multi-step training
  variants; few studies examine why Adam outperforms SGD-type methods for BNNs.

### 2.3 Improving Binary Neural Networks through Fully Utilizing Latent Weights — 2021

- Summary: follow-up work on using latent-weight structure during BNN training.

### 2.4 A Bop and Beyond: A Second Order Optimizer for Binarized Neural Networks — 2021

- Summary: follow-up to Bop; second-order angle on BNN optimisation.

## 3. Bayesian optimisation and Bayesian learning rules for BNNs

### 3.1 Bayesian Optimized 1-Bit CNNs — arXiv:1908.06314 — Gu, Zhao, Jiang, Zhang, Liu, Guo, Ji — 2019

- Summary: Bayesian optimisation approach to the optimisation / architecture
  of 1-bit CNNs.

### 3.2 Training Binary Neural Networks using the Bayesian Learning Rule — 2020

- Summary: cited in the reference list of Helwegen et al. (BOP); central to the
  Bayesian-learning-rule thread the user flagged. arXiv identifier was not
  resolved cleanly in the last pass; treat as unresolved and re-check before
  citing.

## 4. Evolutionary and gradient-free optimisation of networks

### 4.1 Binarizing MobileNet via Evolution-based Searching — arXiv:2005.06305 — Phan, Liu,
Huynh, Savvides, Cheng, Shen — 2020

- Summary: evolution-based (search) approach to binarising a pretrained MobileNet;
  relevant directly to "evolutionary strategy for BNN optimisation".

### 4.2 Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning — arXiv:1712.06567 — Stanley, Miikkulainen — 2017

- Summary: user's thesis context (Chapter 2): evolutionary computation as an
  alternative to gradient-based training for deep nets; GA competitive with
  backprop-style methods.

### 4.3 On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent — arXiv:1712.06564 — 2017

- Summary: OpenAI ES vs SGD connection; the Bernoulli-ES-over-binary-individuals
  design in `bes/algs/es.py` sits in this lineage.

### 4.4 Evolution Strategies as a Scalable Alternative to Reinforcement Learning — arXiv:1703.03864 — 2017

### 4.5 Evolution Strategies at the Hyperscale — arXiv:2511.16652 — 2025

- Summary: ES at scale; batched ES / population-based inference relevant to the
  batched fitness design in `bes/fitness.py` and `bes/mlp.py`.

## 5. Gradient estimation through stochastic and binary operations

### 5.1 Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation — arXiv:1308.3551 — Ranganath, Carroll, Mordvintsev, Lee, Kalai — 2013

- Summary: REINFORCE-style gradient estimation through stochastic discrete
  layers; direct mathematical ancestor of the Bernoulli-ES REINFORCE update for
  binary weight individuals.

### 5.2 Reintroducing Straight-Through Estimators as Principled Methods for Stochastic Binary Networks — 2020

- Summary: STE as a principled estimator for stochastic binary networks.

### 5.3 Estimators Meets Equilibrium Perspective: A Rectified Straight Through Estimator for Binary Neural Networks Training — 2023

### 5.4 SGDAT: An Optimization Method for Binary Neural Networks — 2023

## 6. Surveys

### 6.1 A Comprehensive Survey on Binary Neural Networks — arXiv:2110.06804 — 2021

- Summary: broad BNN survey that cites/links the BOP line of work; useful as the
  entry point for a literature base.

---

## References (compiled)

1. M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio. *Binarized Neural
   Networks: Training Deep Neural Networks with Weights and Activations Constrained
   to +1 or −1.* arXiv:1602.02830 [cs.LG], 2016. https://arxiv.org/abs/1602.02830

2. M. Rastegari, F. Ordonez, J. Redmon, and A. Farhadi. *XNOR-Net: ImageNet
   Classification Using Binary Convolutional Neural Networks.* arXiv:1702.05987,
   2016. https://arxiv.org/abs/1702.05987

3. *DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth
   Gradients.* arXiv:1606.06160, 2016. https://arxiv.org/abs/1606.06160

4. J. van Houcke, F. Hubara, T. Gevers, and Y. Bengio. *Deep Learning with Limited
   Numerical Precision.* arXiv:1509.03421, 2015. https://arxiv.org/abs/1509.03421

5. K. Helwegen, J. Widdicombe, L. Geiger, Z. Liu, K.-T. Cheng, and R. Nusselder.
   *Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization.*
   arXiv:1906.02107, 2019. https://arxiv.org/abs/1906.02107. (NeurIPS 2019.)
   Semantic Scholar: citationCount = 136, influentialCitationCount = 22.
   DOI: 10.48550/arXiv.1906.02107.

6. Z. Liu, G. Shen, S. Li, K. Helwegen, and K.-T. Cheng. *How Do Adam and Training
   Strategies Help BNNs Optimisation?* arXiv:2106.11309, 2021. https://arxiv.org/abs/2106.11309

7. J. Gu, J. Zhao, X. Jiang, B. Zhang, J. Liu, G. Guo, and R. Ji. *Bayesian Optimized
   1-Bit CNNs.* arXiv:1908.06314, 2019. https://arxiv.org/abs/1908.06314

8. *Training Binary Neural Networks using the Bayesian Learning Rule.* 2020.
   (arXiv identifier unresolved in the last pass; flagged for follow-up.)

9. H. Phan, Z. Liu, D. Huynh, M. Savvides, K.-T. Cheng, and Z. Shen. *Binarizing MobileNet
   via Evolution-based Searching.* arXiv:2005.06305, 2020. https://arxiv.org/abs/2005.06305

10. K. P. Stanley and A. A. Miikkulainen. *Deep Neuroevolution: Genetic Algorithms Are a
    Competitive Alternative for Training Deep Neural Networks for Reinforcement
    Learning.* arXiv:1712.06567, 2017. https://arxiv.org/abs/1712.06567

11. J. N. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. *On the Relationship
    Between the OpenAI Evolution Strategy and Stochastic Gradient Descent.* arXiv:1712.06564,
    2017. https://arxiv.org/abs/1712.06564

12. J. Schulman et al. *Evolution Strategies as a Scalable Alternative to Reinforcement
    Learning.* arXiv:1703.03864, 2017. https://arxiv.org/abs/1703.03864

13. *Evolution Strategies at the Hyperscale.* arXiv:2511.16652, 2025. https://arxiv.org/abs/2511.16652

14. R. Ranganath, A. Carroll, A. Mordvintsev, K. Lee, and T. Kalai. *Estimating or Propagating
    Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3551, 2013.
    https://arxiv.org/abs/1308.3551

15. *A Comprehensive Survey on Binary Neural Networks.* arXiv:2110.06804, 2021.
    https://arxiv.org/abs/2110.06804

---

## 7. Synthesis for the current experiments

- The STE baseline maps onto Section 1 (BNN paper + larq STE semantics) and provides
  the gradient-based binarised optimisation baseline.
- The SoftES (Bernoulli REINFORCE over binary individuals) maps onto the ES lineage
  (Sections 4.2–4.5) and the gradient-estimation-through-stochasticity lineage
  (Section 5.1); BOP (Section 2.1) is the binary-specific threshold/EMA alternative.
- The Bayesian learning-rule thread (Section 3) is the natural third optimisation
  family for binary weights, complementing BOP and ES.
- Loss-function background for the fitness signal: `/data/bnn_multiclass_loss_summary.md`.

---

## Open items / questions from this pass

- arXiv identifiers unresolved or uncertain: "Training Binary Neural Networks using
  the Bayesian Learning Rule" (2020), "Reintroducing Straight-Through Estimators...",
  "Estimators Meets Equilibrium Perspective...", "SGDAT...", "A Bop and Beyond...".
  Re-resolve before final citation.
- Venue details (NeurIPS vs. other) for several BNN papers not yet confirmed.
- No systematic citation-graph sweep completed for all key papers (Semantic Scholar
  rate limits interrupted some calls).
