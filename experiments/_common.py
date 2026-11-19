"""Shared plumbing for the experiment runners."""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from bes import data, mlp
from bes.fitness import BNNFitness

ROOT = Path(__file__).resolve().parents[1]


def load_splits():
    return data.load_data()


def make_shapes(hidden, layers):
    hidden_units = tuple([int(hidden)] * int(layers))
    return hidden_units, mlp.weight_shapes(hidden_units=hidden_units)


def make_fitness(args, x_fit, y_fit, shapes):
    return BNNFitness(
        x_fit, y_fit, kind=args.fitness, loss=args.loss,
        batch_size=args.fitness_batch, seed=args.seed + 1, shapes=shapes,
        chunk_size=args.chunk, scale=args.scale, activation=args.activation,
        pop_chunk=args.pop_chunk if args.pop_chunk > 0 else None,
    )


def metrics(bits, bias, x, y, args, shapes, chunk_size=8192):
    loss, acc = mlp.population_metrics(
        tf.reshape(bits, (1, -1)), x, y, chunk_size=chunk_size,
        shapes=shapes, loss=args.loss, bias=None if bias is None else tf.reshape(bias, (1, -1)),
        scale=args.scale, activation=args.activation,
    )
    return {"loss": float(loss[0]), "accuracy": float(acc[0])}


def final_metrics(algo, x_val, y_val, x_test, y_test, args, shapes):
    """Full val/test metrics of an algorithm's mode/best individual."""
    if hasattr(algo, "mode_bits"):
        bits = algo.mode_bits()
        bias = algo.mode_bias() if hasattr(algo, "mode_bias") else None
    else:
        bits, bias = algo.best_bits(), algo.best_bias()
    return {
        "val": metrics(bits, bias, x_val, y_val, args, shapes),
        "test": metrics(bits, bias, x_test, y_test, args, shapes),
    }


def snapshot_steps(iterations):
    return {s for s in (1, 10, 50, 100, 500, 1000, 2000, 5000, 10000)
            if s <= iterations}


def dump_theta(dumps, step, theta):
    dumps[f"theta_it{step:05d}"] = theta.numpy().astype(np.float32)


def validate_args(parser, args, table=True, grad_chunk=True, pop_chunk=True):
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.population <= 0:
        parser.error("--population must be positive")
    if args.fitness_batch <= 0:
        parser.error("--fitness-batch must be positive")
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative (0 = off)")
    if args.chunk <= 0:
        parser.error("--chunk must be positive")
    if args.hidden <= 0:
        parser.error("--hidden must be positive")
    if args.layers < 0:
        parser.error("--layers must be non-negative")
    if table and args.table_rows < 0:
        parser.error("--table-rows must be non-negative")
    if grad_chunk and args.grad_chunk < 0:
        parser.error("--grad-chunk must be non-negative")
    if pop_chunk and args.pop_chunk < 0:
        parser.error("--pop-chunk must be non-negative")


def probability_histogram(probs, bins=32):
    idx = tf.clip_by_value(tf.cast(probs * float(bins), tf.int32), 0, bins - 1)
    counts = tf.math.bincount(idx, minlength=bins, maxlength=bins)
    return [int(c) for c in counts]


def unpack_diagnostics(values, layer_sizes, extra=False):
    """Dict + f_argmax/f_argmin/f_max from a packed diagnostics tensor."""
    names = ["entropy_bits", "committed_frac"]
    if extra:
        names.append("flip_frac")
    names += ["theta_std", "f_mean", "f_std", "f_min", "f_max",
              "f_argmax", "f_argmin"]
    out = {name: float(values[i]) for i, name in enumerate(names)}
    pos = len(names)
    if layer_sizes:
        per_layer = ["entropy_bits_layer", "committed_frac_layer"]
        if extra:
            per_layer.append("flip_frac_layer")
        for _ in layer_sizes:
            for name in per_layer:
                out.setdefault(name, []).append(float(values[pos]))
                pos += 1
    out["f_argmax"] = int(out["f_argmax"])
    out["f_argmin"] = int(out["f_argmin"])
    return out


def default_output_path(stem):
    return ROOT / "results" / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"


def run_search(algo, args, fitness_fn, x_val, y_val, x_test, y_test,
               shapes, layer_sizes, *, method, bias, extra_diag, out_stem,
               started):
    """Sample/evaluate/log/update loop shared by BES and GaussianES."""
    history, diag_history, p_snapshots = [], [], []
    theta_dumps = {}
    snapshots = snapshot_steps(args.iterations)
    grad_chunk = args.grad_chunk if args.grad_chunk > 0 else None
    bestsofar_f, bestsofar_bits, bestsofar_bias = float("-inf"), None, None

    def log_eval(step):
        mode = metrics(algo.mode_bits(), algo.mode_bias() if bias else None,
                       x_val, y_val, args, shapes)
        entry = dict(step=step, mode_val_loss=mode["loss"],
                     mode_val_acc=mode["accuracy"])
        if args.diagnostics:
            fs = algo.fitness_stats()
            pop_bias = algo.bias if bias else None
            best = metrics(algo.population[fs["f_argmax"]],
                           None if pop_bias is None else pop_bias[fs["f_argmax"]],
                           x_val, y_val, args, shapes)
            worst = metrics(algo.population[fs["f_argmin"]],
                            None if pop_bias is None else pop_bias[fs["f_argmin"]],
                            x_val, y_val, args, shapes)
            sofar = metrics(bestsofar_bits, bestsofar_bias, x_val, y_val,
                            args, shapes)
            entry.update(best_val_loss=best["loss"],
                         best_val_acc=best["accuracy"],
                         worst_val_loss=worst["loss"],
                         worst_val_acc=worst["accuracy"],
                         bestsofar_val_loss=sofar["loss"],
                         bestsofar_val_acc=sofar["accuracy"])
        print(f"[{step:4d}] pop fitness mean/best {float(algo.fitness_mean()):+.4f}/"
              f"{float(algo.fitness_best()):+.4f} | mode net val loss "
              f"{mode['loss']:.4f} acc {mode['accuracy']:.4f}")
        history.append(entry)

    for step in range(1, args.iterations + 1):
        algo.sample()
        algo.evaluate(fitness_fn)
        if args.diagnostics:
            diag = unpack_diagnostics(
                algo.diagnostics_packed(layer_sizes).numpy(), layer_sizes,
                extra=extra_diag)
            diag["it"] = step
            diag_history.append(diag)
            if diag["f_max"] > bestsofar_f:
                bestsofar_f = diag["f_max"]
                bestsofar_bits = algo.population[diag["f_argmax"]]
                if bias:
                    bestsofar_bias = algo.bias[diag["f_argmax"]]
            if step in snapshots:
                p_snapshots.append(dict(
                    step=step, hist=probability_histogram(algo.probs())))
        if args.save_theta and step in snapshots:
            dump_theta(theta_dumps, step, algo.theta)
        algo.update(args.lr, grad_chunk=grad_chunk)
        if step == args.iterations or (
                args.eval_every > 0 and step % args.eval_every == 0):
            log_eval(step)

    final = final_metrics(algo, x_val, y_val, x_test, y_test, args, shapes)
    record = {
        "method": method,
        "config": dict(vars(args)),
        "history": history,
        "test": final["test"],
        "val": final["val"],
        "wall_time_s": time.time() - started,
    }
    if args.diagnostics:
        record["diagnostics"] = diag_history
        record["p_snapshots"] = p_snapshots
    out = Path(args.out) if args.out else default_output_path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"Mode-net test accuracy: {final['test']['accuracy']:.4f} | wrote {out}")
    if args.save_theta:
        if args.iterations not in snapshots:
            dump_theta(theta_dumps, args.iterations, algo.theta)
        theta_path = out.with_name(out.stem + "_theta.npz")
        np.savez_compressed(theta_path, **theta_dumps)
        print(f"saved theta snapshots to {theta_path}")
    return out
