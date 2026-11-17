"""The browser runner (docs/nca.js) must reproduce the PyTorch rollout numerically."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from js_reference import make_reference  # noqa: E402

from nca.export import model_from_json, model_to_json  # noqa: E402
from nca.model import NCA  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _random_spec(kind: str) -> dict:
    """Small random (not zero-initialised) model so the update is non-trivial without trained weights."""
    import torch

    torch.manual_seed(0)
    growing = kind == "growing"
    model = NCA(channels=8, hidden=16, fire_rate=0.5, laplacian=not growing, alive_masking=growing,
                padding="zeros" if growing else "circular")
    with torch.no_grad():
        model.w2.weight.normal_(0, 0.05)
        model.w2.bias.normal_(0, 0.02)
    return model_to_json(model, f"random-{kind}", 20, 20)


@pytest.mark.parametrize("kind", ["growing", "texture"])
def test_js_matches_pytorch_on_random_models(kind: str, tmp_path: Path) -> None:
    spec = _random_spec(kind)
    (tmp_path / "model.json").write_text(json.dumps(spec))
    (tmp_path / "ref.json").write_text(json.dumps(make_reference(spec)))
    out = subprocess.run([NODE, str(ROOT / "scripts" / "verify_js.mjs"), str(tmp_path / "model.json"),
                          str(tmp_path / "ref.json"), "1e-4"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_json_roundtrip_is_exact() -> None:
    import torch

    spec = _random_spec("texture")
    m = model_from_json(spec)
    assert torch.equal(m.w1.weight.detach().flatten(), torch.from_numpy(
        __import__("nca.export", fromlist=["unb64"]).unb64(spec["w1"])))


@pytest.mark.parametrize("model_json", sorted((ROOT / "docs" / "models").glob("*.json")))
def test_js_matches_pytorch_on_trained_models(model_json: Path, tmp_path: Path) -> None:
    if model_json.name == "index.json":
        pytest.skip("index")
    spec = json.loads(model_json.read_text())
    (tmp_path / "ref.json").write_text(json.dumps(make_reference(spec)))
    out = subprocess.run([NODE, str(ROOT / "scripts" / "verify_js.mjs"), str(model_json),
                          str(tmp_path / "ref.json"), "1e-3"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
