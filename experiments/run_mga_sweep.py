"""Reproduce the mgab "Arc 1" deme-size x crossrate sweep on MNIST.

Grid over ring deme sizes and tournament crossrate at fixed population,
fitness batch, mutation and bias-std settings; each cell runs 3 seeds for
``--iterations`` steps of ``MicrobialGA.step(fitness_fn)`` with one shared
cycling mini-batch per iteration (mgab ``score_mode='batch'`` protocol).

Every ``--eval-every`` steps the whole population is scored on the full val
split (best/mean accuracy and loss), mirroring the mgab per-individual
full-test monitoring at reduced frequency for CPU budgets; the best-fitness
individual is additionally evaluated on val/test at the end of each run.

Usage:
  /path/to/python experiments/run_mga_sweep.py [--demelist 2,4,8,16,32,64,128,256]
      [--crossrates 1.0] [--seeds 0,1,2] [--population 256]
      [--fitness-batch 512] [--iterations 2000] [--mutation-rate 1e-4]
      [--bias-std 1.0] [--eval-every 10] [--eval-chunk 2048]
      [--fitness loss|accuracy] [--loss ce|hinge]
      [--out-dir results/mga_sweep_exact] [--resume]
"""

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._resume import config_mismatches, format_mismatches

import tensorflow as tf

from bes import data, mlp
from bes.algs.mga import MicrobialGA
from bes.fitness import BNNFitness


def expected_config(args, deme, crossrate, seed):
    return {
        "population": args.population,
        "deme": deme,
        "crossrate": crossrate,
        "seed": seed,
        "batch": args.fitness_batch,
        "iterations": args.iterations,
        "mutation_rate": args.mutation_rate,
        "bias_std": args.bias_std,
        "eval_every": args.eval_every,
        "hidden": args.hidden,
        "fitness": args.fitness,
        "loss": args.loss,
        "schedule": "cycle",
        "scale": args.scale,
        "activation": args.activation,
    }


def run_cell(args, deme, crossrate, seed, x_fit, y_fit, x_val, y_val, x_test, y_test,
             num_bits, num_bias, shapes):
    """One (deme, crossrate, seed) run; returns its record dict."""
    tf.keras.utils.set_random_seed(seed)
    started = time.time()
    fitness_fn = BNNFitness(x_fit, y_fit, kind=args.fitness, loss=args.loss,
                            batch_size=args.fitness_batch, seed=seed + 1,
                            shapes=shapes, schedule="cycle", scale=args.scale,
                            activation=args.activation)
    mga = MicrobialGA(num_bits, args.population, num_bias=num_bias, deme_size=deme,
                      crossrate=crossrate, mutation_rate=args.mutation_rate,
                      bias_std=args.bias_std, seed=seed)

    history = []
    def record(step):
        val_loss, val_acc = mlp.population_metrics(
            mga.population, x_val, y_val, chunk_size=args.eval_chunk, shapes=shapes,
            loss=args.loss, bias=mga.bias, scale=args.scale, activation=args.activation,
        )
        loss, acc = val_loss.numpy(), val_acc.numpy()
        history.append(dict(step=step, best_val_acc=float(acc.max()),
                            mean_val_acc=float(acc.mean()),
                            best_val_loss=float(loss.min()),
                            mean_val_loss=float(loss.mean())))

    for step in range(1, args.iterations + 1):
        mga.step(fitness_fn)
        if step == args.iterations or (
                args.eval_every > 0 and step % args.eval_every == 0):
            record(step)
            print(f"    seed={seed} it={step}/{args.iterations} "
                  f"best_val={history[-1]['best_val_acc']:.4f} "
                  f"mean_val={history[-1]['mean_val_acc']:.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)

    val_loss, val_acc = mlp.population_metrics(tf.reshape(mga.best_bits(), (1, -1)),
                                              x_val, y_val, shapes=shapes, loss=args.loss,
                                              bias=tf.reshape(mga.best_bias(), (1, -1)),
                                              scale=args.scale, activation=args.activation)
    test_loss, test_acc = mlp.population_metrics(tf.reshape(mga.best_bits(), (1, -1)),
                                                x_test, y_test, shapes=shapes, loss=args.loss,
                                                bias=tf.reshape(mga.best_bias(), (1, -1)),
                                                scale=args.scale, activation=args.activation)
    return {
        "method": "mga_sweep",
        "config": {"population": args.population, "deme": deme, "crossrate": crossrate,
                   "seed": seed, "batch": args.fitness_batch, "iterations": args.iterations,
                   "mutation_rate": args.mutation_rate, "bias_std": args.bias_std,
                   "eval_every": args.eval_every, "eval_chunk": args.eval_chunk,
                   "hidden": args.hidden, "fitness": args.fitness, "loss": args.loss,
                   "schedule": "cycle", "scale": args.scale, "activation": args.activation},
        "history": history,
        "val": {"best_net_loss": float(val_loss[0]), "best_net_acc": float(val_acc[0])},
        "test": {"best_net_loss": float(test_loss[0]), "best_net_acc": float(test_acc[0])},
        "wall_time_s": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demelist", type=str, default="2,4,8,16,32,64,128,256")
    parser.add_argument("--crossrates", type=str, default="1.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--fitness-batch", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--mutation-rate", type=float, default=1e-4)
    parser.add_argument("--bias-std", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="dense full-set population monitoring cadence")
    parser.add_argument("--eval-chunk", type=int, default=2048,
                        help="population_metrics sample chunk (memory at pop 256)")
    parser.add_argument("--scale", choices=["fixed", "he", "none"], default="none",
                        help="pre-activation scaling: 'fixed' = 1/sqrt(in_dim) per "
                             "layer, 'he' = sqrt(2/in_dim) per layer (He-style, "
                             "variance-stable for binary weights with relu hidden "
                             "units), 'none' = unscaled logits (mgab r='linear')")
    parser.add_argument("--activation", choices=["sign", "relu"], default="relu",
                        help="hidden activation: 'sign' (binary) or 'relu' (mgab parity)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="hidden layer width (default 64; 1024 for wide-net probes)")
    parser.add_argument("--fitness", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--loss", choices=["ce", "hinge"], default="ce")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="skip per-seed JSONs whose cached config matches "
                             "the requested recipe")
    parser.add_argument("--rerun-stale", action="store_true",
                        help="overwrite cached per-seed JSONs whose config does "
                             "not match the requested recipe")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.population <= 0:
        parser.error("--population must be positive")
    if args.fitness_batch <= 0:
        parser.error("--fitness-batch must be positive")
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative (0 = off)")
    if args.eval_chunk <= 0:
        parser.error("--eval-chunk must be positive")
    if args.hidden <= 0:
        parser.error("--hidden must be positive")
    demes = [int(d) for d in args.demelist.split(",") if d.strip()]
    crossrates = [float(c) for c in args.crossrates.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "mga_sweep_exact"
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = data.load_data()
    shapes = mlp.weight_shapes(hidden_units=args.hidden)
    num_bits, num_bias = mlp.num_bits(hidden_units=args.hidden), mlp.num_bias(hidden_units=args.hidden)
    if args.population % 2:
        raise SystemExit("population must be even for the ring pairing")

    def cell_tag(deme, cr):
        tag = f"p{args.population}_d{deme}_b{args.fitness_batch}_cr{cr}"
        if args.hidden != 64:
            tag += f"_h{args.hidden}"
        return tag

    summary = {}
    for cr in crossrates:
        for deme in demes:
            tag = cell_tag(deme, cr)
            cell_dir = out_dir / tag
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {tag} seeds={seeds} iters={args.iterations} "
                  f"mr={args.mutation_rate} bstd={args.bias_std} ===", flush=True)
            finals, peaks, tests = [], [], []
            for seed in seeds:
                path = cell_dir / f"{tag}_s{seed}.json"
                rec = None
                if args.resume and path.exists():
                    try:
                        rec = json.loads(path.read_text())
                    except (OSError, ValueError) as exc:
                        if not args.rerun_stale:
                            raise SystemExit(
                                f"unreadable resume cache {path}: {exc}")
                        print(f"    seed={seed} unreadable cache; rerunning",
                              flush=True)
                        rec = None
                    if rec is not None and not isinstance(rec, dict):
                        if not args.rerun_stale:
                            raise SystemExit(
                                f"invalid resume cache {path}: expected object")
                        print(f"    seed={seed} invalid cache; rerunning",
                              flush=True)
                        rec = None
                    if rec is not None:
                        stale = config_mismatches(
                            rec.get("config"),
                            expected_config(args, deme, cr, seed))
                        if stale:
                            detail = format_mismatches(stale)
                            if not args.rerun_stale:
                                raise SystemExit(
                                    f"stale resume cache for {path}:\n{detail}\n"
                                    "delete it or pass --rerun-stale")
                            print(f"    seed={seed} stale cache; rerunning\n"
                                  f"{detail}", flush=True)
                            rec = None
                        else:
                            print(f"    seed={seed} resume {path.name}",
                                  flush=True)
                if rec is None:
                    rec = run_cell(args, deme, cr, seed, x_fit, y_fit, x_val, y_val,
                                   x_test, y_test, num_bits, num_bias, shapes)
                    path.write_text(json.dumps(rec, indent=2))
                finals.append(rec["history"][-1]["best_val_acc"])
                peaks.append(max(h["best_val_acc"] for h in rec["history"]))
                tests.append(rec["test"]["best_net_acc"])
                print(f"    seed={seed} final best val {finals[-1]:.4f} "
                      f"| test {rec['test']['best_net_acc']:.4f} | wrote {path.name}",
                      flush=True)
            summary[tag] = {
                "deme": deme, "crossrate": cr,
                "final_best_val_acc": float(np.mean(finals)),
                "peak_best_val_acc": float(np.mean(peaks)),
                "final_best_val_acc_seeds": [float(v) for v in finals],
                "peak_best_val_acc_seeds": [float(v) for v in peaks],
                "test_best_net_acc": float(np.mean(tests)),
                "test_best_net_acc_seeds": [float(v) for v in tests],
            }
            print(f"--- {tag}: final mean {np.mean(finals):.4f} "
                  f"| peak mean {np.mean(peaks):.4f} ---", flush=True)

    summary["_config"] = {k: v for k, v in vars(args).items()
                          if k not in ("demelist", "crossrates", "seeds")}
    tag = "+".join(str(c).replace(".", "p") for c in crossrates)
    (out_dir / f"summary_cr{tag}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY: mean-over-seeds best val accuracy (peak | final) ===")
    demes_hdr = "cr\\deme " + "".join(f"{d:>10}" for d in demes)
    print(demes_hdr + "   [peak]")
    for cr in crossrates:
        row_p, row_f = f"{cr:>6}   ", f"{cr:>6}   "
        for deme in demes:
            s = summary[cell_tag(deme, cr)]
            row_p += f"{s['peak_best_val_acc']:10.4f}"
            row_f += f"{s['final_best_val_acc']:10.4f}"
        print(row_p + "   [peak]")
        print(row_f + "   [final]")
    print(f"wrote {out_dir / ('summary_cr' + '+'.join(str(c).replace('.', 'p') for c in crossrates) + '.json')}")


if __name__ == "__main__":
    main()
