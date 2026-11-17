import json
from pathlib import Path

from nca.train import build_parser, train


def test_training_smoke(tmp_path: Path) -> None:
    """A few iterations on a tiny grid run end to end and write the expected artefacts."""
    args = build_parser().parse_args([
        "--target", "lizard", "--mode", "regenerating", "--size", "16", "--padding", "4", "--hidden", "16",
        "--batch-size", "4", "--pool-size", "16", "--damage-n", "1", "--iters", "6", "--min-steps", "4",
        "--max-steps", "6", "--device", "cpu", "--no-compile", "--out-dir", str(tmp_path)])
    out = train(args)
    assert (out / "model.pt").exists() and (out / "loss.csv").exists()
    assert json.loads((out / "summary.json").read_text())["iters"] == 6
