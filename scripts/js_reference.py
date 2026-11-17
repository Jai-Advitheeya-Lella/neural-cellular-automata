"""Write a PyTorch reference rollout for ``scripts/verify_js.mjs``.

Given an exported model JSON, run the model in PyTorch from a fixed initial state with *explicit*
per-step update masks (so no random numbers differ between the two implementations) and record
the state at a few checkpoints.

    python scripts/js_reference.py docs/models/lizard.json /tmp/ref.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from nca.export import model_from_json
from nca.model import make_seed

CHECKPOINTS = (1, 10, 30, 64)


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(np.ascontiguousarray(t.numpy(), dtype="<f4").tobytes()).decode("ascii")


def cell_major(x: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) -> (H, W, C), the layout used by the JavaScript runner."""
    return x[0].permute(1, 2, 0).contiguous()


def make_reference(spec: dict, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = model_from_json(spec)
    c, h, w = spec["channels"], spec["height"], spec["width"]
    x = make_seed(1, c, h, w) if spec["kind"] == "growing" else torch.rand(1, c, h, w) - 0.5
    init = cell_major(x)
    steps = max(CHECKPOINTS)
    masks = torch.rand(steps, 1, 1, h, w) <= spec["fire_rate"]
    states = {}
    with torch.no_grad():
        for t in range(steps):
            x = model(x, update_mask=masks[t])
            if t + 1 in CHECKPOINTS:
                states[str(t + 1)] = b64(cell_major(x))
    mask_bytes = masks.reshape(steps, h * w).to(torch.uint8).numpy()
    return {"init": b64(init), "masks": base64.b64encode(mask_bytes.tobytes()).decode("ascii"),
            "steps": steps, "states": states}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(make_reference(json.loads(args.model.read_text()))))


if __name__ == "__main__":
    main()
