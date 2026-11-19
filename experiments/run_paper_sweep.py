"""Paper sweep: population / depth / fitness-shaping / batch-size effects of
BES on MNIST, one factor at a time around a shared base recipe.

Base recipe (best stable recorded ES configuration on this baseline):
h1024x1, relu hidden units, he pre-activation scale, loss-based fitness on
categorical cross-entropy, sigma=0 (weights-only search, biases frozen at
zero), standardize shaping, no antithetic pairing, no perturbation table,
fitness batch 4096, and lr = --lr-scale / population (the REINFORCE
estimator is population-summed, E[g] ~ P, so lr ~ 1/P holds the effective
step size constant across population sizes while gradient SNR grows
~ sqrt(P)).

Arms (each varies one factor, everything else at base):
  pop    population in --pops at h1024x1
  depth  h{--depth-hidden}x{--depths} at population --sweep-pop
  shape  shaping in --shapings at h1024x1 and population --sweep-pop
  batch  fitness batch in --batches at h1024x1 and population --sweep-pop

Cells whose full configuration coincides (the top population cell is also
the shaping standardize cell and the batch 4096 cell) share one run tag and
execute once. Each run is a subprocess of experiments/run_es.py (one fresh
TF process per run, with recorded-working GPU memory levers per
architecture) writing a per-seed JSON under <out-dir>/runs/ plus a per-run
stdout log. --resume skips finished run JSONs; one failed run does not
abort the sweep.

Usage:
  /path/to/python experiments/run_paper_sweep.py [--seeds 42,43,44]
      [--iterations 1000] [--eval-every 10] [--pops 128,256,512,1024]
      [--lr-scale 100.0] [--sweep-pop 0] [--depth-hidden 256]
      [--depths 1,2,3] [--shapings none,standardize,rank]
      [--batches 256,1024,4096] [--arms pop,depth,shape,batch]
      [--chunk 0] [--grad-chunk 0] [--pop-chunk 0]
      [--out-dir results/paper_sweep] [--resume] [--dry-run]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from experiments._resume import config_mismatches, format_mismatches

GPU_LEVERS = {
    (1024, 1): dict(chunk=512, grad_chunk=128, pop_chunk=128),
    (256, 1): dict(chunk=512, grad_chunk=64, pop_chunk=128),
    (256, 2): dict(chunk=512, grad_chunk=64, pop_chunk=128),
    (256, 3): dict(chunk=512, grad_chunk=32, pop_chunk=128),
}


def parse_ints(text):
    return [int(t) for t in text.split(",") if t.strip()]


def load_record(path):
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(rec, dict) and "val" in rec and rec.get("history"):
        return rec
    return None


def expected_config(spec, args, seed):
    return {
        "population": spec["pop"],
        "lr": spec["lr"],
        "iterations": args.iterations,
        "p_init": 0.5,
        "fitness": "loss",
        "loss": "ce",
        "fitness_batch": spec["batch"],
        "eval_every": args.eval_every,
        "seed": seed,
        "table_rows": 0,
        "antithetic": False,
        "sigma": 0.0,
        "shaping": spec["shaping"],
        "bias_lr": None,
        "adam": False,
        "grad_chunk": spec["grad_chunk"],
        "natural": False,
        "pop_chunk": spec["pop_chunk"],
        "hidden": spec["hidden"],
        "layers": spec["layers"],
        "scale": "he",
        "activation": "relu",
        "chunk": spec["chunk"],
    }


def cell_summary(recs, seeds, spec):
    done = [(s, r) for s, r in zip(seeds, recs) if r is not None]
    out = dict(config={k: spec[k] for k in
                       ("hidden", "layers", "pop", "lr", "batch", "shaping",
                        "chunk", "grad_chunk", "pop_chunk")},
               net_forwards_per_iteration=spec["pop"] * spec["batch"],
               seeds=[s for s, _ in done],
               failed_seeds=[s for s, r in zip(seeds, recs) if r is None])
    if not done:
        return out
    finals = [r["history"][-1]["mode_val_acc"] for _, r in done]
    peaks = [max(h["mode_val_acc"] for h in r["history"]) for _, r in done]
    peak_steps = [max(r["history"], key=lambda h: h["mode_val_acc"])["step"]
                  for _, r in done]
    val_losses = [r["history"][-1]["mode_val_loss"] for _, r in done]
    tests = [r["test"]["accuracy"] for _, r in done]
    test_losses = [r["test"]["loss"] for _, r in done]
    vals = [r["val"]["accuracy"] for _, r in done]
    walls = [r["wall_time_s"] for _, r in done]
    for name, series in (("final_val_acc", finals), ("peak_val_acc", peaks),
                         ("peak_val_step", peak_steps),
                         ("final_val_loss", val_losses),
                         ("final_test_acc", tests),
                         ("final_test_loss", test_losses),
                         ("final_fullval_acc", vals), ("wall_time_s", walls)):
        out[name] = [float(v) for v in series]
        out[name + "_mean"] = float(statistics.mean(series))
        out[name + "_std"] = (float(statistics.stdev(series))
                              if len(series) > 1 else 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="full-val eval cadence (0 = only the final iteration)")
    parser.add_argument("--pops", type=str, default="128,256,512,1024")
    parser.add_argument("--lr-scale", type=float, default=100.0,
                        help="lr = lr-scale / population (step-size rule for "
                             "the population-summed estimator)")
    parser.add_argument("--sweep-pop", type=int, default=0,
                        help="population for the depth/shape/batch arms "
                             "(0 = max of --pops)")
    parser.add_argument("--depth-hidden", type=int, default=256)
    parser.add_argument("--base-hidden", type=int, default=1024,
                        help="hidden width for the pop/shape/batch arms")
    parser.add_argument("--depths", type=str, default="1,2,3")
    parser.add_argument("--shapings", type=str, default="none,standardize,rank")
    parser.add_argument("--batches", type=str, default="256,1024,4096")
    parser.add_argument("--arms", type=str, default="pop,depth,shape,batch")
    parser.add_argument("--chunk", type=int, default=0,
                        help="override the auto sample-chunk lever (0 = auto)")
    parser.add_argument("--grad-chunk", type=int, default=0,
                        help="override the auto grad-chunk lever (0 = auto)")
    parser.add_argument("--pop-chunk", type=int, default=0,
                        help="override the auto pop-chunk lever (0 = auto)")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="interpreter for the run_es.py subprocesses")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="skip per-seed JSONs whose cached config matches "
                             "the requested recipe")
    parser.add_argument("--rerun-stale", action="store_true",
                        help="overwrite cached per-seed JSONs whose config does "
                             "not match the requested recipe")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative (0 = off)")
    if args.chunk < 0 or args.grad_chunk < 0 or args.pop_chunk < 0:
        parser.error("chunk levers must be non-negative (0 = auto)")
    if args.lr_scale <= 0:
        parser.error("--lr-scale must be positive")

    seeds = parse_ints(args.seeds)
    pops = parse_ints(args.pops)
    depths = parse_ints(args.depths)
    shapings = [s for s in args.shapings.split(",") if s.strip()]
    batches = parse_ints(args.batches)
    arms = [a for a in args.arms.split(",") if a.strip()]
    if not seeds or not pops:
        parser.error("--seeds and --pops must be non-empty")
    sweep_pop = args.sweep_pop if args.sweep_pop > 0 else max(pops)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "paper_sweep"
    args.out_dir = str(out_dir)
    runs_dir = out_dir / "runs"
    if not args.dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)

    specs, cells = {}, []

    def register(arm, label, hidden, layers, pop, batch, shaping):
        lr = args.lr_scale / pop
        levers = dict(GPU_LEVERS.get((hidden, layers),
                                     dict(chunk=512, grad_chunk=32,
                                          pop_chunk=128)))
        if args.chunk > 0:
            levers["chunk"] = args.chunk
        if args.grad_chunk > 0:
            levers["grad_chunk"] = args.grad_chunk
        if args.pop_chunk > 0:
            levers["pop_chunk"] = args.pop_chunk
        tag = (f"es_h{hidden}x{layers}_he_relu_p{pop}_lr{lr:.6g}"
               f"_b{batch}_sh{shaping}")
        if tag not in specs:
            specs[tag] = dict(tag=tag, hidden=hidden, layers=layers, pop=pop,
                              lr=lr, batch=batch, shaping=shaping, **levers)
        cells.append(dict(arm=arm, label=label, tag=tag))

    if "pop" in arms:
        for pop in pops:
            register("pop", f"p{pop}", args.base_hidden, 1, pop, 4096,
                     "standardize")
    if "depth" in arms:
        for layers in depths:
            register("depth", f"h{args.depth_hidden}x{layers}",
                     args.depth_hidden, layers, sweep_pop, 4096, "standardize")
    if "shape" in arms:
        for shaping in shapings:
            register("shape", f"sh_{shaping}", args.base_hidden, 1, sweep_pop,
                     4096, shaping)
    if "batch" in arms:
        for batch in batches:
            register("batch", f"b{batch}", args.base_hidden, 1, sweep_pop,
                     batch, "standardize")

    plan = [(cell, seed) for cell in cells for seed in seeds]
    print(f"plan: {len(cells)} cells x {len(seeds)} seeds = {len(plan)} runs "
          f"({len(specs)} unique configs, sweep_pop={sweep_pop}, "
          f"lr_scale={args.lr_scale})", flush=True)
    for cell in cells:
        spec = specs[cell["tag"]]
        print(f"  [{cell['arm']:>5}] {cell['label']:>8} -> {spec['tag']} "
              f"(lr {spec['lr']:.6g}, chunk {spec['chunk']}, "
              f"grad-chunk {spec['grad_chunk']}, pop-chunk {spec['pop_chunk']})",
              flush=True)
    if args.dry_run:
        return

    recs, failed = {}, []
    total = len(plan)
    for i, (cell, seed) in enumerate(plan, start=1):
        spec = specs[cell["tag"]]
        key = (cell["tag"], seed)
        if key in recs:
            continue
        out = runs_dir / f"{spec['tag']}_s{seed}.json"
        label = f"[{i}/{total}] {cell['arm']}/{cell['label']} seed {seed}"
        cached = load_record(out) if args.resume else None
        if cached is not None:
            stale = config_mismatches(
                cached.get("config"), expected_config(spec, args, seed))
            if stale:
                detail = format_mismatches(stale)
                if not args.rerun_stale:
                    raise SystemExit(
                        f"stale resume cache for {out}:\n{detail}\n"
                        "delete it or pass --rerun-stale")
                print(f"WARNING: stale resume cache for {out}; rerunning\n"
                      f"{detail}", flush=True)
            else:
                recs[key] = cached
                print(f"{label} RESUME val "
                      f"{cached['history'][-1]['mode_val_acc']:.4f} test "
                      f"{cached['test']['accuracy']:.4f}", flush=True)
                continue
        log = out.with_suffix(".log")
        print(f"{label} START (log {log})", flush=True)
        started = time.time()
        cmd = [args.python, str(HERE / "run_es.py"),
               "--hidden", str(spec["hidden"]),
               "--layers", str(spec["layers"]),
               "--scale", "he", "--activation", "relu",
               "--fitness", "loss", "--loss", "ce",
               "--population", str(spec["pop"]), "--lr", repr(spec["lr"]),
               "--fitness-batch", str(spec["batch"]), "--sigma", "0",
               "--shaping", spec["shaping"], "--table-rows", "0",
               "--iterations", str(args.iterations),
               "--eval-every", str(args.eval_every), "--seed", str(seed),
               "--chunk", str(spec["chunk"]),
               "--grad-chunk", str(spec["grad_chunk"]),
               "--pop-chunk", str(spec["pop_chunk"]), "--out", str(out)]
        with log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = log.read_text(errors="replace").splitlines()[-20:]
            print(f"{label} FAILED (rc {proc.returncode}):", flush=True)
            print("\n".join(tail), flush=True)
            failed.append(key)
            recs[key] = None
            continue
        rec = load_record(out)
        if rec is None:
            print(f"{label} FAILED (run finished but {out.name} unreadable)",
                  flush=True)
            failed.append(key)
            recs[key] = None
            continue
        recs[key] = rec
        print(f"{label} DONE val {rec['history'][-1]['mode_val_acc']:.4f} "
              f"test {rec['test']['accuracy']:.4f} "
              f"({time.time() - started:.0f}s incl. startup)", flush=True)

    summary = {"_config": dict(vars(args), sweep_pop=sweep_pop,
                               base_recipe=dict(hidden=args.base_hidden,
                                                layers=1,
                                                scale="he", activation="relu",
                                                fitness="loss", loss="ce",
                                                sigma=0.0, table_rows=0,
                                                lr_rule="lr_scale/pop"))}
    for arm in dict.fromkeys(cell["arm"] for cell in cells):
        section = {}
        for cell in (c for c in cells if c["arm"] == arm):
            section[cell["label"]] = cell_summary(
                [recs.get((cell["tag"], s)) for s in seeds], seeds,
                specs[cell["tag"]])
        summary[arm] = section

    print("\n=== SUMMARY (mean±std over seeds) ===", flush=True)
    for arm, section in summary.items():
        if arm.startswith("_"):
            continue
        print(f"--- {arm} ---", flush=True)
        print(f"{'cell':>10} {'final_val':>17} {'peak_val':>17} "
              f"{'final_test':>17} {'wall_s':>8}", flush=True)
        for label, s in section.items():
            if not s.get("seeds"):
                print(f"{label:>10} {'FAILED':>17}", flush=True)
                continue
            row = f"{label:>10} "
            for key in ("final_val_acc", "peak_val_acc", "final_test_acc"):
                row += f"{s[key + '_mean']:.4f}±{s[key + '_std']:.4f} "
            row += f"{s['wall_time_s_mean']:8.0f}"
            print(row, flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path} ({len(failed)} failed runs)", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
