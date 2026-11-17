// Neural Cellular Automaton runner (CPU, plain JavaScript, no dependencies).
//
// Mirrors src/nca/model.py: fixed identity + Sobel (+ Laplacian) perception, a 1x1-conv MLP
// (F*C -> hidden -> C, ReLU), stochastic per-cell updates, and for the growing model alpha-based
// alive masking. Zero padding for growing models, circular (toroidal) padding for textures.
// State is stored cell-major, (H, W, C), in a Float32Array.
//
// Works in the browser and in Node (tests/test_js_parity.py runs it against PyTorch).

const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1].map((v) => v / 8);
const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1].map((v) => v / 8);
const LAPLACIAN = [1, 2, 1, 2, -12, 2, 1, 2, 1].map((v) => v / 16);

export function decodeFloat32(b64) {
  let bytes;
  if (typeof atob === "function") {
    const bin = atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
    bytes = new Uint8Array(Buffer.from(b64, "base64"));
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

export class NCA {
  /**
   * @param {object} spec  JSON produced by `python -m nca.export`.
   */
  constructor(spec) {
    this.name = spec.name;
    this.kind = spec.kind || "growing";
    this.C = spec.channels;
    this.hidden = spec.hidden;
    this.H = spec.height;
    this.W = spec.width;
    this.fireRate = spec.fire_rate;
    this.aliveMasking = spec.alive_masking !== false;
    this.aliveThreshold = spec.alive_threshold ?? 0.1;
    this.laplacian = !!spec.laplacian;
    this.circular = spec.padding === "circular";
    this.F = this.laplacian ? 4 : 3;
    this.w1 = decodeFloat32(spec.w1); // (hidden, F*C)
    this.b1 = decodeFloat32(spec.b1);
    this.w2 = decodeFloat32(spec.w2); // (C, hidden)
    this.b2 = decodeFloat32(spec.b2);
    const n = this.H * this.W;
    this.state = new Float32Array(n * this.C);
    this.next = new Float32Array(n * this.C);
    this.preAlive = new Uint8Array(n);
    this.postAlive = new Uint8Array(n);
    this.nonzero = new Uint8Array(n);
    this.percept = new Float64Array(this.F * this.C);
    this.hid = new Float64Array(this.hidden);
    this.kx = SOBEL_X.slice();
    this.ky = SOBEL_Y.slice();
    this.angle = 0;
    this.stepCount = 0;
    this.nbr = this._neighbours();
    // A growing-model cell whose whole 3x3 neighbourhood is zero perceives 0, so dx = W2 relu(b1) + b2.
    this.dxZero = new Float64Array(this.C);
    for (let c = 0; c < this.C; c++) {
      let s = this.b2[c];
      for (let j = 0; j < this.hidden; j++) s += this.w2[c * this.hidden + j] * Math.max(0, this.b1[j]);
      this.dxZero[c] = s;
    }
    this.reset();
  }

  /** Index of the 3x3 neighbours of every cell (row-major k = (dy+1)*3 + dx+1); -1 outside a zero-padded grid. */
  _neighbours() {
    const { H, W } = this;
    const nbr = new Int32Array(H * W * 9);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            let yy = y + dy, xx = x + dx, v;
            if (this.circular) {
              yy = (yy + H) % H; xx = (xx + W) % W; v = yy * W + xx;
            } else {
              v = yy < 0 || yy >= H || xx < 0 || xx >= W ? -1 : yy * W + xx;
            }
            nbr[(y * W + x) * 9 + (dy + 1) * 3 + (dx + 1)] = v;
          }
        }
      }
    }
    return nbr;
  }

  /** Growing model: a single live cell in the centre. Texture model: uniform noise in [-0.5, 0.5]. */
  reset() {
    if (this.kind === "texture") {
      for (let i = 0; i < this.state.length; i++) this.state[i] = Math.random() - 0.5;
    } else {
      this.state.fill(0);
      const i = ((this.H >> 1) * this.W + (this.W >> 1)) * this.C;
      for (let c = 3; c < this.C; c++) this.state[i + c] = 1;
    }
    this.stepCount = 0;
  }

  /** Overwrite the state (cell-major Float32Array of length H*W*C). */
  loadState(arr) {
    this.state.set(arr);
    this.stepCount = 0;
  }

  /** Rotate the perception frame (radians); the grown pattern rotates with it. */
  setAngle(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    for (let k = 0; k < 9; k++) {
      this.kx[k] = c * SOBEL_X[k] - s * SOBEL_Y[k];
      this.ky[k] = s * SOBEL_X[k] + c * SOBEL_Y[k];
    }
    this.angle = angle;
  }

  _aliveMask(buf, out) {
    const { H, W, C, nbr } = this;
    const thr = this.aliveThreshold;
    for (let i = 0; i < H * W; i++) {
      let m = -Infinity;
      for (let k = 0; k < 9; k++) {
        const j = nbr[i * 9 + k];
        if (j < 0) continue;
        const a = buf[j * C + 3];
        if (a > m) m = a;
      }
      out[i] = m > thr ? 1 : 0;
    }
  }

  /**
   * Advance one step.
   * @param {Uint8Array|null} fireMask  Optional (H*W) mask of cells that update;
   *   if omitted each cell fires with probability `fireRate`.
   */
  step(fireMask = null) {
    const { H, W, C, F, hidden, w1, b1, w2, b2, state, next, percept, hid, kx, ky, nbr } = this;
    const nIn = F * C;
    const sparse = this.aliveMasking && !this.circular;
    if (this.aliveMasking) this._aliveMask(state, this.preAlive);

    const nz = this.nonzero;
    if (sparse) {
      // Mark cells with any non-zero channel; perception is exactly zero if the 3x3 block is all zero.
      for (let i = 0; i < H * W; i++) {
        let any = 0;
        const o = i * C;
        for (let c = 0; c < C; c++) if (state[o + c] !== 0) { any = 1; break; }
        nz[i] = any;
      }
    }

    for (let cell = 0; cell < H * W; cell++) {
      const o = cell * C;
      const fire = fireMask ? fireMask[cell] : (Math.random() <= this.fireRate ? 1 : 0);
      if (!fire) {
        for (let c = 0; c < C; c++) next[o + c] = state[o + c];
        continue;
      }
      if (sparse) {
        let active = 0;
        for (let k = 0; k < 9; k++) { const j = nbr[cell * 9 + k]; if (j >= 0 && nz[j]) { active = 1; break; } }
        if (!active) {
          for (let c = 0; c < C; c++) next[o + c] = state[o + c] + this.dxZero[c];
          continue;
        }
      }
      // Perception: identity, Sobel-x, Sobel-y (and Laplacian) of every channel.
      for (let c = 0; c < C; c++) {
        let gx = 0, gy = 0, lap = 0;
        for (let k = 0; k < 9; k++) {
          const j = nbr[cell * 9 + k];
          if (j < 0) continue;
          const v = state[j * C + c];
          gx += kx[k] * v;
          gy += ky[k] * v;
          if (F === 4) lap += LAPLACIAN[k] * v;
        }
        percept[F * c] = state[o + c];
        percept[F * c + 1] = gx;
        percept[F * c + 2] = gy;
        if (F === 4) percept[F * c + 3] = lap;
      }
      // Hidden layer with ReLU.
      for (let j = 0; j < hidden; j++) {
        let s = b1[j];
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) s += w1[row + i] * percept[i];
        hid[j] = s > 0 ? s : 0;
      }
      // Output layer (residual update).
      for (let c = 0; c < C; c++) {
        let s = b2[c];
        const row = c * hidden;
        for (let j = 0; j < hidden; j++) {
          const h = hid[j];
          if (h !== 0) s += w2[row + j] * h;
        }
        next[o + c] = state[o + c] + s;
      }
    }

    if (this.aliveMasking) {
      // Keep cells that are alive both before and after the update.
      this._aliveMask(next, this.postAlive);
      for (let i = 0; i < H * W; i++) {
        const o = i * C;
        if (this.preAlive[i] && this.postAlive[i]) {
          for (let c = 0; c < C; c++) state[o + c] = next[o + c];
        } else {
          for (let c = 0; c < C; c++) state[o + c] = 0;
        }
      }
    } else {
      state.set(next);
    }
    this.stepCount++;
  }

  /** Destroy every cell within `radius` cells of (cx, cy): zero for growing models, fresh noise for textures. */
  damage(cx, cy, radius) {
    const { H, W, C, state } = this;
    const r2 = radius * radius;
    for (let y = Math.max(0, Math.floor(cy - radius)); y <= Math.min(H - 1, Math.ceil(cy + radius)); y++) {
      for (let x = Math.max(0, Math.floor(cx - radius)); x <= Math.min(W - 1, Math.ceil(cx + radius)); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 > r2) continue;
        const o = (y * W + x) * C;
        if (this.kind === "texture") for (let c = 0; c < C; c++) state[o + c] = Math.random() - 0.5;
        else state.fill(0, o, o + C);
      }
    }
  }

  /** Write the state as an RGBA byte buffer (length H*W*4): premultiplied RGBA over white (growing) or RGB+0.5 (texture). */
  toRGBA(out) {
    const { H, W, C, state } = this;
    const tex = this.kind === "texture";
    for (let i = 0; i < H * W; i++) {
      const o = i * C;
      const a = tex ? 0 : Math.min(1, Math.max(0, state[o + 3]));
      for (let k = 0; k < 3; k++) {
        const v = tex ? state[o + k] + 0.5 : 1 - a + state[o + k];
        out[4 * i + k] = Math.round(255 * Math.min(1, Math.max(0, v)));
      }
      out[4 * i + 3] = 255;
    }
    return out;
  }
}
