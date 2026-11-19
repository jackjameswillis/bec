"""Run the soft Bernoulli ES on MNIST via the batched BNN fitness and write a
history JSON to results/.

Usage:
  /path/to/python experiments/run_es.py [--population 100] [--lr 0.1]
      [--iterations 300]       [--fitness loss|accuracy] [--loss ce|hinge]
      [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--table-rows 1024]
      [--antithetic] [--sigma 0.1] [--shaping none|standardize|rank] [--bias-lr LR]
      [--adam] [--natural] [--grad-chunk K] [--pop-chunk K] [--diagnostics]
      [--save-theta] [--out PATH] [--hidden 64] [--layers 1]
      [--scale fixed|he|none] [--activation sign|relu] [--chunk 8192]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from bes import mlp
from bes.algs.es import BES
from experiments._common import (
    load_splits, make_fitness, make_shapes, run_search, validate_args,
)
from experiments._names import es_stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--p-init", type=float, default=0.5)
    parser.add_argument("--fitness", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--loss", choices=["ce", "hinge"], default="ce",
                        help="loss minimised by loss-based fitness (squared hinge if 'hinge')")
    parser.add_argument("--fitness-batch", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="full-val eval cadence (0 = only the final iteration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-rows", type=int, default=1024,
                        help="perturbation table rows; 0 disables the table "
                             "(legacy per-generation RNG)")
    parser.add_argument("--antithetic", action="store_true",
                        help="pair each perturbation row u with 1-u for the "
                             "second half of the population; default off")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Gaussian perturbation scale for the continuous "
                             "layer biases (ES-learned mean, fixed scale); "
                             "0 fixes biases at zero (weights-only search)")
    parser.add_argument("--shaping", choices=["none", "standardize", "rank"],
                        default="none",
                        help="fitness shaping for the REINFORCE advantage: "
                             "'none' = raw f - mean(f) (unbiased), "
                             "'standardize' = population z-score, "
                             "'rank' = centered ordinal ranks in [-0.5, 0.5] "
                             "(Salimans et al. 1703.03864)")
    parser.add_argument("--bias-lr", type=float, default=None,
                        help="separate learning rate for the continuous bias "
                             "coordinates (default: same as --lr)")
    parser.add_argument("--adam", action="store_true",
                        help="preprocess population-summed gradients with a "
                             "per-coordinate Adam rule (default: raw update)")
    parser.add_argument("--grad-chunk", type=int, default=0,
                        help="chunk the population reduction into this many rows "
                             "in BES.update (0 = full shot; memory lever for "
                             "large populations; only changes fp summation order)")
    parser.add_argument("--natural", action="store_true",
                        help="divide the weight-bit gradient by the diagonal "
                             "Bernoulli Fisher s(1-s) (eps-damped natural "
                             "gradient; default: raw REINFORCE gradient)")
    parser.add_argument("--pop-chunk", type=int, default=0,
                        help="chunk the population axis of fitness evaluation "
                             "into this many rows (0 = full population; memory "
                             "lever for large populations; results unchanged)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="hidden layer width (default 64)")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of hidden layers (default 1)")
    parser.add_argument("--scale", choices=["fixed", "he", "none"], default="fixed",
                        help="pre-activation scaling: 'fixed' = 1/sqrt(in_dim) "
                             "per layer, 'he' = sqrt(2/in_dim) per layer "
                             "(He-style, variance-stable for binary weights "
                             "with relu hidden units), 'none' = unscaled logits")
    parser.add_argument("--activation", choices=["sign", "relu"], default="sign",
                        help="hidden activation: 'sign' (binary) or 'relu'")
    parser.add_argument("--chunk", type=int, default=8192,
                        help="population_metrics sample chunk (GPU memory lever "
                             "for wide nets; does not change results)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="record convergence diagnostics to the JSON: "
                             "per-iteration entropy/committed-fraction/fitness "
                             "stats of the sampling distribution, p-histogram "
                             "snapshots, and per-eval full-val accuracy of the "
                             "current population's best/worst individuals plus "
                             "the best-so-far sample (default off)")
    parser.add_argument("--save-theta", action="store_true",
                        help="dump the search-distribution parameters to a "
                             "<out-stem>_theta.npz next to the JSON: theta "
                             "(float32) at every snapshot step plus the final "
                             "iteration (~1.1 MB per snapshot at D=269k)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    validate_args(parser, args)

    tf.keras.utils.set_random_seed(args.seed)
    started = time.time()

    (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = load_splits()
    hidden_units, shapes = make_shapes(args.hidden, args.layers)
    num_bits = mlp.num_bits(hidden_units=hidden_units)
    num_bias = mlp.num_bias(hidden_units=hidden_units)
    fitness_fn = make_fitness(args, x_fit, y_fit, shapes)
    es = BES(num_bits, args.population, p_init=args.p_init,
             table_rows=args.table_rows if args.table_rows > 0 else None,
             antithetic=args.antithetic, seed=args.seed,
             num_bias=num_bias, sigma=args.sigma, shaping=args.shaping,
             bias_lr=args.bias_lr, adam=args.adam, natural=args.natural)

    layer_sizes = [int(i * o) for i, o in shapes]
    assert sum(layer_sizes) == num_bits
    out_stem = es_stem(args)
    run_search(es, args, fitness_fn, x_val, y_val, x_test, y_test,
               shapes, layer_sizes, method="es", bias=True,
               extra_diag=False, out_stem=out_stem, started=started)

if __name__ == "__main__":
    main()
