"""Run the STE baseline on MNIST and write a history JSON to results/.

Usage:
  /path/to/python experiments/run_ste.py [--epochs 25] [--batch-size 512]
      [--learning-rate 0.001] [--hidden 64] [--layers 1] [--scale none] [--eval-every N] [--optimizer sgd|adam] [--no-input-quant] [--use-bias] [--seed 42] [--out PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from bes import data, ste


class EvalCallback(tf.keras.callbacks.Callback):
    """Full-validation eval every N training batches, printed ES-style
    (``[step] val loss acc``) and collected for the history JSON."""

    def __init__(self, x_val, y_val, eval_every, batch_size):
        super().__init__()
        self.x_val, self.y_val = x_val, y_val
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.step = 0
        self.evals = []

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.eval_every > 0 and self.step % self.eval_every == 0:
            loss, acc = self.model.evaluate(self.x_val, self.y_val,
                                            batch_size=self.batch_size, verbose=0)
            self.evals.append({"step": self.step, "val_loss": float(loss),
                               "val_acc": float(acc)})
            print(f"[{self.step:5d}] val loss {loss:.4f} acc {acc:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=ste.HIDDEN_UNITS,
                        help="hidden-layer width (default: mlp.HIDDEN_UNITS = 64)")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of hidden QuantDense layers (default 1)")
    parser.add_argument("--scale", choices=["fixed", "he", "none"], default="none",
                        help="per-layer pre-activation scale mirroring bes/mlp._scale "
                             "(default 'none' = the original recipe)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="evaluate the full val set every N training batches "
                             "and print ES-style stats (0 = off, per-epoch only)")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                        help="optimizer (default sgd, momentum 0, as in the original script)")
    parser.add_argument("--no-input-quant", action="store_true",
                        help="skip the ste_sign input binarisation: the first layer "
                             "takes the raw normalised pixels (full precision)")
    parser.add_argument("--use-bias", action="store_true",
                        help="include biases after each per-layer pre-activation "
                             "scale, matching the ES forward; default no biases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative (0 = off)")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    started = time.time()

    (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = data.load_data()
    eval_cb = EvalCallback(x_val, y_val, args.eval_every, args.batch_size)
    model = ste.build_model(args.learning_rate, use_bias=args.use_bias,
                            hidden_units=args.hidden, hidden_layers=args.layers,
                            scale=args.scale, optimizer=args.optimizer,
                            input_quant=not args.no_input_quant)
    history = ste.train(model, x_fit, y_fit, x_val, y_val,
                        batch_size=args.batch_size, epochs=args.epochs,
                        callbacks=[eval_cb] if args.eval_every else None)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)

    record = {
        "method": "ste",
        "config": dict(vars(args)),
        "history": {key: [float(v) for v in vals] for key, vals in history.history.items()},
        "evals": eval_cb.evals,
        "test": {"loss": float(test_loss), "accuracy": float(test_acc)},
        "wall_time_s": time.time() - started,
    }
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / (
        f"ste_h{args.hidden}{'x' + str(args.layers) if args.layers != 1 else ''}"
        f"{'_' + args.scale if args.scale != 'none' else ''}"
        f"{'_' + args.optimizer if args.optimizer != 'sgd' else ''}"
        f"{'_noiq' if args.no_input_quant else ''}"
        f"{'_bias' if args.use_bias else ''}_seed{args.seed}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"Test accuracy: {test_acc:.4f} | wrote {out}")


if __name__ == "__main__":
    main()
