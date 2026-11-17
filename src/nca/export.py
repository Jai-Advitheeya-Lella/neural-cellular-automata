"""Export trained growing-NCA weights to compact JSON for the browser runner (``docs/nca.js``).

Weights are stored as base64-encoded little-endian float32 so the JavaScript side
reconstructs bit-identical parameters.

    python -m nca.export --runs runs --out docs/models
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from nca.model import NCA


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def model_to_json(model: NCA, name: str, height: int, width: int) -> dict:
    """Serialise the update MLP.

    ``w1`` is ``(hidden, F*C)`` with inputs ordered [c0_id, c0_dx, c0_dy, (c0_lap,) c1_id, ...]; ``F`` is 3,
    or 4 for the texture model (extra Laplacian filter).
    """
    w1 = model.w1.weight.detach().cpu().numpy()[:, :, 0, 0]
    b1 = model.w1.bias.detach().cpu().numpy()
    w2 = model.w2.weight.detach().cpu().numpy()[:, :, 0, 0]
    b2 = (model.w2.bias.detach().cpu().numpy() if model.w2.bias is not None
          else np.zeros(model.channels, dtype=np.float32))
    return {
        "name": name, "kind": "growing" if model.alive_masking else "texture",
        "channels": model.channels, "hidden": int(w1.shape[0]), "fire_rate": model.fire_rate,
        "laplacian": model.laplacian, "padding": model.padding, "alive_masking": model.alive_masking,
        "alive_threshold": 0.1, "height": height, "width": width,
        "w1": b64(w1), "b1": b64(b1), "w2": b64(w2), "b2": b64(b2),
    }


def unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


def model_from_json(spec: dict) -> NCA:
    """Rebuild the PyTorch model from an exported spec (inverse of :func:`model_to_json`)."""
    model = NCA(channels=spec["channels"], hidden=spec["hidden"], fire_rate=spec["fire_rate"],
                laplacian=spec["laplacian"], alive_masking=spec["alive_masking"], padding=spec["padding"])
    n_in = (4 if spec["laplacian"] else 3) * spec["channels"]
    with torch.no_grad():
        model.w1.weight.copy_(torch.from_numpy(unb64(spec["w1"])).view(spec["hidden"], n_in, 1, 1))
        model.w1.bias.copy_(torch.from_numpy(unb64(spec["b1"])))
        model.w2.weight.copy_(torch.from_numpy(unb64(spec["w2"])).view(spec["channels"], spec["hidden"], 1, 1))
        model.w2.bias.copy_(torch.from_numpy(unb64(spec["b2"])))
    return model.eval()


def load_model(path: Path) -> tuple[NCA, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NCA(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("args", {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs/models"))
    p.add_argument("--texture-size", type=int, default=80, help="grid side for texture models in the demo")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for ckpt in sorted(args.runs.glob("*/model.pt")):
        run = ckpt.parent.name
        model, targs = load_model(ckpt)
        if run.startswith("texture_"):
            name, side, label = run, args.texture_size, f"texture: {run.removeprefix('texture_')}"
        else:
            target, mode = run.rsplit("_", 1)
            name = target if mode == "regenerating" else f"{target}-{mode}"
            side = targs.get("size", 40) + 2 * targs.get("padding", 16)
            label = f"{target}" + ("" if mode == "regenerating" else f" ({mode}-trained)")
        spec = model_to_json(model, name, side, side)
        (args.out / f"{name}.json").write_text(json.dumps(spec))
        index.append({"name": name, "label": label, "kind": spec["kind"]})
        print(f"exported {name} ({side}x{side}, {spec['kind']})")
    # Demo order: lizard first, other patterns alphabetically, textures last.
    index.sort(key=lambda m: (m["kind"] == "texture", m["name"] != "lizard", m["name"]))
    (args.out / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
