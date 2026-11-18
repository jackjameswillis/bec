"""Default output-stem compatibility tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._names import es_stem, ges_stem, ste_stem


def main():
    es = SimpleNamespace(
        fitness="loss", loss="ce", hidden=256, layers=2, scale="he",
        activation="relu", population=2048, lr=0.048828125, sigma=0.0,
        bias_lr=None, seed=42, iterations=200, table_rows=0,
        antithetic=False, shaping="rank", adam=False, natural=False,
        diagnostics=True,
    )
    stem = es_stem(es)
    assert stem == (
        "es_loss_ce_h256x2_he_relu_p2048_lr0.048828125_s0.0"
        "_seed42_it200_t0_shrank_diag"
    ), stem
    assert "_s0.0" in stem and "_it200" in stem

    ges = SimpleNamespace(
        fitness="loss", loss="ce", hidden=64, layers=1, scale="fixed",
        activation="sign", population=100, lr=0.1, sigma=1.0, init_std=1.0,
        seed=7, iterations=300, table_rows=1024, antithetic=True,
        shaping="standardize", adam=True, diagnostics=True,
    )
    stem = ges_stem(ges)
    assert stem == (
        "ges_loss_ce_h64x1_fixed_sign_p100_lr0.1_s1.0_is1.0"
        "_seed7_it300_t1024_anti_shstandardize_adam_diag"
    ), stem
    assert "_s1.0" in stem and "_it300" in stem and "_is1.0" in stem

    ste = SimpleNamespace(
        hidden=256, layers=2, scale="he", optimizer="adam",
        no_input_quant=True, use_bias=True, learning_rate=0.001,
        epochs=100, batch_size=512, seed=42,
    )
    stem = ste_stem(ste, "20261118_120000")
    assert stem == (
        "ste_h256x2_he_adam_noiq_bias_lr0.001_e100_b512_seed42_20261118_120000"
    ), stem
    ste.learning_rate = 0.01
    assert ste_stem(ste, "20261118_120000") != stem
    print("default output stems OK")


if __name__ == "__main__":
    main()
