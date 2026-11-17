// Compare the JavaScript runner with a PyTorch reference rollout.
//
//   python scripts/js_reference.py docs/models/lizard.json /tmp/ref.json
//   node scripts/verify_js.mjs docs/models/lizard.json /tmp/ref.json [tolerance]
//
// Exits with status 1 if the maximum absolute difference at any checkpoint exceeds the tolerance.
import { readFileSync } from "node:fs";
import { NCA, decodeFloat32 } from "../docs/nca.js";

const [modelPath, refPath, tolArg] = process.argv.slice(2);
const tol = tolArg ? parseFloat(tolArg) : 1e-3;
const spec = JSON.parse(readFileSync(modelPath, "utf8"));
const ref = JSON.parse(readFileSync(refPath, "utf8"));

const nca = new NCA(spec);
nca.loadState(decodeFloat32(ref.init));
const nCells = spec.height * spec.width;
const masks = new Uint8Array(Buffer.from(ref.masks, "base64"));

let worst = 0;
const report = {};
for (let t = 0; t < ref.steps; t++) {
  nca.step(masks.subarray(t * nCells, (t + 1) * nCells));
  const key = String(t + 1);
  if (ref.states[key]) {
    const expected = decodeFloat32(ref.states[key]);
    let maxDiff = 0;
    for (let i = 0; i < expected.length; i++) maxDiff = Math.max(maxDiff, Math.abs(expected[i] - nca.state[i]));
    report[key] = maxDiff;
    worst = Math.max(worst, maxDiff);
  }
}
console.log(JSON.stringify({ model: spec.name, max_abs_diff_by_step: report, worst }));
process.exit(worst <= tol ? 0 : 1);
