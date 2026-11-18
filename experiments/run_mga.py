"""Run the microbial GA on MNIST via the batched BNN fitness and write a
history JSON to results/.

Usage:
  /path/to/python experiments/run_mga.py [--population 100] [--deme 2]
      [--crossrate 0.5] [--mutation-rate 0.001] [--bias-std 0.01]
      [--iterations 300] [--fitness loss|accuracy] [--loss ce|hinge]
      [--fitness-batch 4096] [--eval-every 10] [--seed 42] [--out PATH]
      [--scale fixed|he|none] [--activation sign|relu] [--hidden 64]
      [--layers 1] [--chunk 8192] [--pop-chunk 0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from bes import mlp
from bes.algs.mga import MicrobialGA
from experiments._common import (
    final_metrics, load_splits, make_fitness, make_shapes, metrics,
    validate_args,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--deme", type=int, default=2,
                        help="ring deme size; >= population means a fully "
                             "random matching each generation")
    parser.add_argument("--crossrate", type=float, default=0.5,
                        help="per-gene probability that the tournament loser "
                             "copies the winner's gene (1 = full clone)")
    parser.add_argument("--mutation-rate", type=float, default=1e-3,
                        help="per-weight-bit flip probability applied to losers")
    parser.add_argument("--bias-std", type=float, default=0.01,
                        help="std of Gaussian noise added to loser biases")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--fitness", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--loss", choices=["ce", "hinge"], default="ce",
                        help="loss minimised by loss-based fitness (squared hinge if 'hinge')")
    parser.add_argument("--fitness-batch", type=int, default=4096)
    parser.add_argument("--scale", choices=["fixed", "he", "none"], default="fixed",
                        help="pre-activation scaling: 'fixed' = 1/sqrt(in_dim) "
                             "per layer, 'he' = sqrt(2/in_dim) per layer "
                             "(He-style, variance-stable for binary weights "
                             "with relu hidden units), 'none' = unscaled logits")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="full-val eval cadence (0 = only the final iteration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64,
                        help="hidden layer width (default 64)")
    parser.add_argument("--layers", type=int, default=1,
                        help="hidden depth (default 1)")
    parser.add_argument("--activation", choices=["sign", "relu"], default="sign",
                        help="hidden activation: 'sign' = binary units, "
                             "'relu' = dense relu (mgab-style)")
    parser.add_argument("--chunk", type=int, default=8192,
                        help="population_metrics sample chunk (GPU memory lever "
                             "for wide nets; does not change results)")
    parser.add_argument("--pop-chunk", type=int, default=0,
                        help="chunk the population axis of fitness evaluation "
                             "into this many rows (0 = full population; memory "
                             "lever for large populations; results unchanged)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    validate_args(parser, args, table=False, grad_chunk=False)
    if args.population % 2:
        parser.error("--population must be even for ring pairing")

    tf.keras.utils.set_random_seed(args.seed)
    started = time.time()

    (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = load_splits()
    hidden_units, shapes = make_shapes(args.hidden, args.layers)
    fitness_fn = make_fitness(args, x_fit, y_fit, shapes)
    mga = MicrobialGA(mlp.num_bits(hidden_units=hidden_units), args.population,
                      num_bias=mlp.num_bias(hidden_units=hidden_units), deme_size=args.deme,
                      crossrate=args.crossrate, mutation_rate=args.mutation_rate,
                      bias_std=args.bias_std, seed=args.seed)

    history = []
    def log_eval(step):
        best = metrics(mga.best_bits(), mga.best_bias(), x_val, y_val, args, shapes)
        print(f"[{step:4d}] pop fitness mean/best {float(mga.fitness_mean()):+.4f}/"
              f"{float(mga.fitness_best()):+.4f} | best net val loss "
              f"{best['loss']:.4f} acc {best['accuracy']:.4f}")
        history.append(dict(step=step, best_val_loss=best["loss"],
                            best_val_acc=best["accuracy"]))

    for step in range(1, args.iterations + 1):
        mga.step(fitness_fn)
        if step == args.iterations or (
                args.eval_every > 0 and step % args.eval_every == 0):
            log_eval(step)

    final = final_metrics(mga, x_val, y_val, x_test, y_test, args, shapes)
    record = {
        "method": "mga",
        "config": dict(vars(args)),
        "history": history,
        "test": final["test"],
        "val": final["val"],
        "wall_time_s": time.time() - started,
    }
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / (
        f"mga_{args.fitness}_{args.loss}_h{args.hidden}"
        f"{'x' + str(args.layers) if args.layers != 1 else ''}"
        f"{'_' + args.scale if args.scale != 'fixed' else ''}"
        f"{'_' + args.activation if args.activation != 'sign' else ''}"
        f"_p{args.population}_d{args.deme}_cr{args.crossrate}"
        f"_mr{args.mutation_rate}_bstd{args.bias_std}_seed{args.seed}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"Best-net test accuracy: {final['test']['accuracy']:.4f} | wrote {out}")


if __name__ == "__main__":
    main()
