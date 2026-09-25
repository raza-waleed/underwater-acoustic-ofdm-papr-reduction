/* PAPR lab: the methods from my M.Eng. thesis, running live in the browser.
 *
 * A dependency free classic script, so it also works when the report is opened
 * straight from disk. PaprLab.mount(root, data) builds three views:
 *   Transmitter  PAPR reduction: repeated clipping and filtering, clipping,
 *                rooting companding, selected mapping, partial transmit sequences
 *   Receiver     frequentative decision feedback (FFB) with the learned amplifier
 *   Channel      the BELLHOP 2 km channel and the guard interval it needs
 * On load the engine recomputes the cases the Python reference exported
 * (data.reference) and reports how many match.
 */
(function () {
  'use strict';

  /* ================================================================ FFT (radix 2, in place) */
  const plans = {};
  function plan(n) {
    if (plans[n]) return plans[n];
    const bits = Math.round(Math.log2(n));
    if ((1 << bits) !== n) throw new Error('FFT size must be a power of two: ' + n);
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0, v = i;
      for (let b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; }
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos(2 * Math.PI * k / n); sin[k] = Math.sin(2 * Math.PI * k / n); }
    return (plans[n] = { rev, cos, sin });
  }

  /* forward: exp(-j 2 pi k n / N); inverse: exp(+j ...) and 1/N, like numpy */
  function fft(re, im, inverse) {
    const n = re.length, p = plan(n), rev = p.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    const sgn = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = p.cos[k * step], wi = sgn * p.sin[k * step];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
    if (inverse) for (let q = 0; q < n; q++) { re[q] /= n; im[q] /= n; }
  }

  /* ================================================================ random numbers */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = rand();
      while (u < 1e-300) u = rand();
      const v = rand(), m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
  }

  /* ================================================================ Gray QAM, same mapping as paprlab.ofdm */
  function grayToBinary(g) { let b = g; for (let s = g >> 1; s; s >>= 1) b ^= s; return b; }
  function qam(order) {
    const m = Math.round(Math.sqrt(order)), k = Math.round(Math.log2(order));
    return { order, m, k, half: k / 2, norm: Math.sqrt(2 * (order - 1) / 3) };
  }
  function modulate(bits, q) {
    const n = bits.length / q.k, re = new Float64Array(n), im = new Float64Array(n);
    for (let s = 0; s < n; s++) {
      let gi = 0, gq = 0;
      const o = s * q.k;
      for (let b = 0; b < q.half; b++) gi = (gi << 1) | bits[o + b];
      for (let b = 0; b < q.half; b++) gq = (gq << 1) | bits[o + q.half + b];
      re[s] = (2 * grayToBinary(gi) - (q.m - 1)) / q.norm;
      im[s] = (2 * grayToBinary(gq) - (q.m - 1)) / q.norm;
    }
    return { re, im };
  }
  function level(v, q) { return Math.min(q.m - 1, Math.max(0, Math.round((v * q.norm + (q.m - 1)) / 2))); }
  function countBitErrors(re, im, bits, q) {
    let errors = 0;
    for (let s = 0; s < re.length; s++) {
      const li = level(re[s], q), lq = level(im[s], q);
      const gi = li ^ (li >> 1), gq = lq ^ (lq >> 1), o = s * q.k;
      for (let b = 0; b < q.half; b++) {
        if (((gi >> (q.half - 1 - b)) & 1) !== bits[o + b]) errors++;
        if (((gq >> (q.half - 1 - b)) & 1) !== bits[o + q.half + b]) errors++;
      }
    }
    return errors;
  }
  function sliceInto(re, im, q, outRe, outIm) {
    for (let s = 0; s < re.length; s++) {
      outRe[s] = (2 * level(re[s], q) - (q.m - 1)) / q.norm;
      outIm[s] = (2 * level(im[s], q) - (q.m - 1)) / q.norm;
    }
  }
  function randomBits(rand, n) { const b = new Uint8Array(n); for (let i = 0; i < n; i++) b[i] = rand() < 0.5 ? 0 : 1; return b; }

  /* ================================================================ OFDM (paprlab.ofdm.to_time / to_freq) */
  function toTime(Xr, Xi, size) {
    const N = Xr.length, h = N >> 1, re = new Float64Array(size), im = new Float64Array(size);
    for (let k = 0; k < h; k++) { re[k] = Xr[k]; im[k] = Xi[k]; }
    for (let k = h; k < N; k++) { re[size - N + k] = Xr[k]; im[size - N + k] = Xi[k]; }
    fft(re, im, true);
    const s = size / Math.sqrt(N);
    for (let k = 0; k < size; k++) { re[k] *= s; im[k] *= s; }
    return { re, im };
  }
  function toFreq(xr, xi, N) {
    const size = xr.length, h = N >> 1, re = Float64Array.from(xr), im = Float64Array.from(xi);
    fft(re, im, false);
    const s = Math.sqrt(N) / size, Xr = new Float64Array(N), Xi = new Float64Array(N);
    for (let k = 0; k < h; k++) { Xr[k] = re[k] * s; Xi[k] = im[k] * s; }
    for (let k = h; k < N; k++) { Xr[k] = re[size - N + k] * s; Xi[k] = im[size - N + k] * s; }
    return { re: Xr, im: Xi };
  }
  function paprDb(re, im) {
    let mx = 0, sum = 0;
    for (let i = 0; i < re.length; i++) { const p = re[i] * re[i] + im[i] * im[i]; sum += p; if (p > mx) mx = p; }
    return 10 * Math.log10(mx / (sum / re.length));
  }

  /* ================================================================ transmitter methods */
  function clipInPlace(re, im, ratio, fixedLevel) {
    let lim = fixedLevel;
    if (lim == null) {
      let s = 0;
      for (let i = 0; i < re.length; i++) { const a = Math.hypot(re[i], im[i]); s += a * a; }
      lim = Math.sqrt(ratio * s / re.length);
    }
    for (let i = 0; i < re.length; i++) {
      const a = Math.hypot(re[i], im[i]);
      if (a > lim) { const f = lim / a; re[i] *= f; im[i] *= f; }
    }
  }
  function bandLimitInPlace(re, im, N) {
    const size = re.length, h = N >> 1;
    fft(re, im, false);
    for (let k = h; k < size - (N - h); k++) { re[k] = 0; im[k] = 0; }
    fft(re, im, true);
  }
  function rcfInPlace(re, im, N, ratio, passes, onPass) {
    for (let p = 0; p < passes; p++) {
      clipInPlace(re, im, ratio);
      bandLimitInPlace(re, im, N);
      if (onPass) onPass(p, paprDb(re, im));
    }
  }
  function rootInPlace(re, im, R) {
    for (let i = 0; i < re.length; i++) {
      const a = Math.hypot(re[i], im[i]);
      if (a > 0) { const f = Math.pow(a, R - 1); re[i] *= f; im[i] *= f; }
    }
  }
  const JPOW = [[1, 0], [0, 1], [-1, 0], [0, -1]];
  function slmBest(X, phaseIndex, size) {
    const N = X.re.length;
    let best = null, bestP = Infinity, bestU = 0;
    for (let u = 0; u < phaseIndex.length; u++) {
      const pr = new Float64Array(N), pi = new Float64Array(N), row = phaseIndex[u];
      for (let k = 0; k < N; k++) {
        const w = JPOW[row[k]];
        pr[k] = X.re[k] * w[0] - X.im[k] * w[1];
        pi[k] = X.re[k] * w[1] + X.im[k] * w[0];
      }
      const x = toTime(pr, pi, size), p = paprDb(x.re, x.im);
      if (p < bestP) { bestP = p; best = x; bestU = u; }
    }
    return { x: best, papr: bestP, index: bestU };
  }
  function ptsPhaseSet(W) { return W === 2 ? [[1, 0], [-1, 0]] : [[1, 0], [-1, 0], [0, 1], [0, -1]]; }
  function ptsBest(X, V, W, size) {
    const N = X.re.length, width = N / V, parts = [];
    for (let v = 0; v < V; v++) {
      const pr = new Float64Array(N), pi = new Float64Array(N);
      for (let k = v * width; k < (v + 1) * width; k++) { pr[k] = X.re[k]; pi[k] = X.im[k]; }
      parts.push(toTime(pr, pi, size));
    }
    const set = ptsPhaseSet(W), combos = Math.pow(W, V - 1);
    const cr = new Float64Array(size), ci = new Float64Array(size);
    let bestP = Infinity, bestC = 0;
    for (let c = 0; c < combos; c++) {
      cr.set(parts[0].re); ci.set(parts[0].im);
      for (let v = 1; v < V; v++) {
        const digit = Math.floor(c / Math.pow(W, V - 1 - v)) % W, w = set[digit], pv = parts[v];
        for (let t = 0; t < size; t++) {
          cr[t] += pv.re[t] * w[0] - pv.im[t] * w[1];
          ci[t] += pv.re[t] * w[1] + pv.im[t] * w[0];
        }
      }
      const p = paprDb(cr, ci);
      if (p < bestP) { bestP = p; bestC = c; }
    }
    const xr = Float64Array.from(parts[0].re), xi = Float64Array.from(parts[0].im);
    for (let v = 1; v < V; v++) {
      const w = set[Math.floor(bestC / Math.pow(W, V - 1 - v)) % W], pv = parts[v];
      for (let t = 0; t < size; t++) { xr[t] += pv.re[t] * w[0] - pv.im[t] * w[1]; xi[t] += pv.re[t] * w[1] + pv.im[t] * w[0]; }
    }
    return { x: { re: xr, im: xi }, papr: bestP, index: bestC };
  }

  /* ================================================================ amplifier and the learned network */
  function rappInPlace(re, im, sat, p) {
    for (let i = 0; i < re.length; i++) {
      const a = Math.hypot(re[i], im[i]);
      const g = Math.pow(1 + Math.pow(a / sat, 2 * p), -1 / (2 * p));
      re[i] *= g; im[i] *= g;
    }
  }
  function Network(w) {
    this.I = w.inputs; this.H = w.hidden;
    this.w1 = Float64Array.from([].concat.apply([], w.w1)); this.b1 = Float64Array.from(w.b1);
    this.w2 = Float64Array.from([].concat.apply([], w.w2)); this.b2 = Float64Array.from(w.b2);
    this.inLo = w.in_range[0]; this.inHi = w.in_range[1]; this.outLo = w.out_range[0]; this.outHi = w.out_range[1];
  }
  Network.prototype.predict = function (flat) {
    const I = this.I, H = this.H, n = flat.length, pad = (I - (n % I)) % I;
    const out = new Float64Array(n), buf = new Float64Array(I), s = new Float64Array(H);
    for (let start = 0; start < n + pad; start += I) {
      for (let i = 0; i < I; i++) {
        const idx = start + i, v = idx < n ? flat[idx] : flat[n - 1];
        buf[i] = 2 * (v - this.inLo) / (this.inHi - this.inLo) - 1;
      }
      for (let h = 0; h < H; h++) {
        let z = this.b1[h];
        for (let i = 0; i < I; i++) z += this.w1[h * I + i] * buf[i];
        s[h] = 1 / (1 + Math.exp(-z));
      }
      for (let o = 0; o < I; o++) {
        let y = this.b2[o];
        for (let h = 0; h < H; h++) y += this.w2[o * H + h] * s[h];
        if (start + o < n) out[start + o] = (y + 1) * (this.outHi - this.outLo) / 2 + this.outLo;
      }
    }
    return out;
  };
  /* apply a transmitter model to a batch of symbols (concatenated, as numpy does) */
  function modelBatch(model, xs) {
    const size = xs[0].re.length, n = xs.length * size, out = [];
    if (model.kind === 'exact') {
      for (const x of xs) { const r = Float64Array.from(x.re), i = Float64Array.from(x.im); rappInPlace(r, i, model.sat, 2); out.push({ re: r, im: i }); }
      return out;
    }
    const amps = new Float64Array(n);
    xs.forEach((x, s) => { for (let t = 0; t < size; t++) amps[s * size + t] = Math.hypot(x.re[t], x.im[t]); });
    const y = model.net.predict(amps);
    xs.forEach((x, s) => {
      const r = new Float64Array(size), i = new Float64Array(size);
      for (let t = 0; t < size; t++) {
        const a = amps[s * size + t], v = y[s * size + t];
        if (a > 0) { r[t] = x.re[t] * v / a; i[t] = x.im[t] * v / a; } else { r[t] = v; i[t] = 0; }
      }
      out.push({ re: r, im: i });
    });
    return out;
  }
  function bussgang(xs, ys) {
    let num = 0, den = 0;
    xs.forEach((x, s) => { const y = ys[s]; for (let t = 0; t < x.re.length; t++) { num += x.re[t] * y.re[t] + x.im[t] * y.im[t]; den += x.re[t] * x.re[t] + x.im[t] * x.im[t]; } });
    return num / den;
  }

  /* FFB (thesis Algorithm 1): Y received subcarriers per symbol, H channel.
     Returns decisions after pass 0 and after the last pass, plus the soft values. */
  function ffb(Ys, H, q, model, alpha, size, passes) {
    const N = H.re.length;
    const Z = Ys.map(Y => {
      const r = new Float64Array(N), i = new Float64Array(N);
      for (let k = 0; k < N; k++) {
        const d = H.re[k] * H.re[k] + H.im[k] * H.im[k];
        r[k] = (Y.re[k] * H.re[k] + Y.im[k] * H.im[k]) / d;
        i[k] = (Y.im[k] * H.re[k] - Y.re[k] * H.im[k]) / d;
      }
      return { re: r, im: i };
    });
    let D = Ys.map(() => ({ re: new Float64Array(N), im: new Float64Array(N) }));
    let first = null, last = null, soft = null;
    for (let it = 0; it <= passes; it++) {
      soft = Z.map((z, s) => {
        const r = new Float64Array(N), i = new Float64Array(N);
        for (let k = 0; k < N; k++) { r[k] = (z.re[k] - D[s].re[k]) / alpha; i[k] = (z.im[k] - D[s].im[k]) / alpha; }
        return { re: r, im: i };
      });
      const dec = soft.map(v => { const r = new Float64Array(N), i = new Float64Array(N); sliceInto(v.re, v.im, q, r, i); return { re: r, im: i }; });
      if (it === 0) first = dec;
      last = dec;
      if (it === passes) break;
      const xs = dec.map(d => toTime(d.re, d.im, size));
      const ys = modelBatch(model, xs);
      D = xs.map((x, s) => {
        const r = new Float64Array(size), i = new Float64Array(size);
        for (let t = 0; t < size; t++) { r[t] = ys[s].re[t] - alpha * x.re[t]; i[t] = ys[s].im[t] - alpha * x.im[t]; }
        return toFreq(r, i, N);
      });
    }
    return { first, last, soft };
  }

  /* ================================================================ channel */
  function channelResponse(arrivals, N, spacing, centre) {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      const f = centre + (k < N / 2 ? k : k - N) * spacing;
      for (const a of arrivals) {
        const ph = a.phase_deg * Math.PI / 180 + 2 * Math.PI * f * (a.delay_ms / 1000);
        re[k] += a.amp * Math.cos(ph);
        im[k] -= a.amp * Math.sin(ph);
      }
    }
    let p = 0;
    for (let k = 0; k < N; k++) p += re[k] * re[k] + im[k] * im[k];
    const s = 1 / Math.sqrt(p / N);
    for (let k = 0; k < N; k++) { re[k] *= s; im[k] *= s; }
    return { re, im };
  }

  /* ================================================================ statistics helpers */
  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function ccdfCurve(sorted, grid) {
    const n = sorted.length, out = [];
    let j = 0;
    for (const z of grid) {
      while (j < n && sorted[j] <= z) j++;
      out.push([z, 1 - j / n]);
    }
    return out;
  }
  const classB = db => 100 * (Math.PI / 4) * Math.pow(10, -db / 20);
  const GRID = []; for (let z = 2; z <= 14.0001; z += 0.05) GRID.push(Math.round(z * 100) / 100);

  /* ================================================================ verification against Python */
  function verify(data) {
    const ref = data.reference, e = ref.expect, checks = [];
    const close = (a, b, tol) => Math.abs(a - b) <= tol;
    const add = (name, ok, detail) => checks.push({ name, ok, detail });
    const qp = qam(4), q64 = qam(64);
    const Xs = ref.qpsk_bits.map(b => modulate(b, qp));
    const px = Xs.map(X => { const x = toTime(X.re, X.im, 256); return paprDb(x.re, x.im); });
    add('PAPR, 64 subcarriers, oversampled x4', px.every((v, i) => close(v, e.papr_L4[i], 6e-5)), px.map(v => v.toFixed(4)).join(', '));
    const rcfOk = Xs.every((X, s) => {
      const x = toTime(X.re, X.im, 128); let ok = true;
      rcfInPlace(x.re, x.im, 64, 3, 4, (p, v) => { ok = ok && close(v, e.rcf_cr3_L2_passes[p][s], 6e-5); });
      return ok;
    });
    add('Repeated clipping and filtering, 4 passes', rcfOk, 'CR 3, I 2');
    const rct = Xs.map(X => { const x = toTime(X.re, X.im, 256); rootInPlace(x.re, x.im, 0.5); return paprDb(x.re, x.im); });
    add('Rooting companding, R = 0.5', rct.every((v, i) => close(v, e.rct_r05_L4[i], 6e-5)), rct.map(v => v.toFixed(4)).join(', '));
    const slm = Xs.map(X => slmBest(X, ref.slm_phase_index, 256));
    add('Selected mapping, U = 4', slm.every((r, i) => r.index === e.slm_u4_L4.index[i] && close(r.papr, e.slm_u4_L4.papr[i], 6e-5)), 'candidate ' + slm.map(r => r.index).join(', '));
    const pts = Xs.map(X => ptsBest(X, 4, 2, 256));
    add('Partial transmit sequences, V = 4, W = 2', pts.every((r, i) => r.index === e.pts_v4_w2_L4.index[i] && close(r.papr, e.pts_v4_w2_L4.papr[i], 6e-5)), 'combination ' + pts.map(r => r.index).join(', '));
    const net = new Network(data.networks['5'].weights);
    const nOut = net.predict(Float64Array.from(e.network_ibo5.input));
    add('Neural network forward pass (6-12-6)', e.network_ibo5.output.every((v, i) => close(nOut[i], v, 2e-8)), '12 amplitudes');
    const X64 = ref.qam64_bits.map(b => modulate(b, q64));
    const xs = X64.map(X => toTime(X.re, X.im, 256));
    const sat = Math.pow(10, 5 / 20);
    const ys = xs.map(x => { const r = Float64Array.from(x.re), i = Float64Array.from(x.im); rappInPlace(r, i, sat, 2); return { re: r, im: i }; });
    const alpha = bussgang(xs, ys);
    add('Bussgang gain of the amplifier', close(alpha, e.ffb_ibo5_alpha, 2e-9), alpha.toFixed(8));
    const Ytx = ys.map(y => toFreq(y.re, y.im, 64));
    const one = { re: new Float64Array(64).fill(1), im: new Float64Array(64) };
    const out = ffb(Ytx, one, q64, { kind: 'net', net }, alpha, 256, 2);
    const softOk = e.ffb_ibo5_soft_pass2.every((v, k) => close(out.soft[0].re[k], v[0], 2e-7) && close(out.soft[0].im[k], v[1], 2e-7));
    add('FFB soft values after two passes', softOk, '6 subcarriers');
    const H = channelResponse(data.channel.arrivals, data.uwa.subcarriers, data.uwa.bandwidth_hz / data.uwa.subcarriers, data.uwa.centre_hz);
    const hOk = e.channel_H.every(([k, r, i]) => close(H.re[k], r, 2e-7) && close(H.im[k], i, 2e-7));
    add('BELLHOP channel response', hOk, '7 subcarriers');
    return checks;
  }

  /* ================================================================ drawing */
  const C = { bg: '#0a0e14', panel: '#0d131e', grid: '#1c2634', axis: '#2a3648', text: '#c7d0dc', muted: '#8695a8',
    cyan: '#39d2ff', orange: '#ffb020', green: '#00e68a', magenta: '#e879f9', violet: '#a78bfa', red: '#f87171' };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶' };
  const pow10 = e => e === 0 ? '1' : '10' + String(e).split('').map(ch => SUP[ch] || ch).join('');

  function niceStep(span, target) {
    const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  }
  function Plot(canvas, height) { this.c = canvas; this.h = height; this.spec = null; }
  Plot.prototype.draw = function (spec) {
    if (spec) this.spec = spec;
    spec = this.spec;
    if (!spec) return;
    const cv = this.c, dpr = window.devicePixelRatio || 1, W = Math.max(260, cv.clientWidth || 600), Hh = this.h;
    cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); cv.style.height = Hh + 'px';
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, Hh);
    const m = { l: spec.y.log ? 48 : 50, r: 12, t: spec.title ? 26 : 10, b: 38 };
    const pw = W - m.l - m.r, ph = Hh - m.t - m.b;
    const lx = spec.y.log;
    const X = v => m.l + (v - spec.x.min) / (spec.x.max - spec.x.min) * pw;
    const Y = lx ? v => m.t + (Math.log10(spec.y.max) - Math.log10(v)) / (Math.log10(spec.y.max) - Math.log10(spec.y.min)) * ph
      : v => m.t + (spec.y.max - v) / (spec.y.max - spec.y.min) * ph;
    g.fillStyle = C.panel; g.fillRect(m.l, m.t, pw, ph);
    g.font = '11px "IBM Plex Mono", Consolas, monospace';
    if (spec.title) { g.fillStyle = C.text; g.font = '600 12px Inter, system-ui, sans-serif'; g.fillText(spec.title, m.l, 16); g.font = '11px "IBM Plex Mono", Consolas, monospace'; }
    (spec.bands || []).forEach(b => { g.fillStyle = b.color; g.fillRect(X(Math.max(b.x0, spec.x.min)), m.t, X(Math.min(b.x1, spec.x.max)) - X(Math.max(b.x0, spec.x.min)), ph); });
    g.strokeStyle = C.grid; g.lineWidth = 1; g.fillStyle = C.muted; g.textAlign = 'center';
    const xs = spec.x.step || niceStep(spec.x.max - spec.x.min, Math.max(3, Math.floor(pw / 70)));
    for (let v = Math.ceil(spec.x.min / xs - 1e-9) * xs; v <= spec.x.max + 1e-9; v += xs) {
      const px = Math.round(X(v)) + 0.5;
      g.beginPath(); g.moveTo(px, m.t); g.lineTo(px, m.t + ph); g.stroke();
      g.fillText(spec.x.fmt ? spec.x.fmt(v) : String(Math.round(v * 1000) / 1000), px, m.t + ph + 14);
    }
    g.textAlign = 'right';
    if (lx) {
      for (let e = Math.ceil(Math.log10(spec.y.min)); e <= Math.floor(Math.log10(spec.y.max)); e++) {
        const py = Math.round(Y(Math.pow(10, e))) + 0.5;
        g.beginPath(); g.moveTo(m.l, py); g.lineTo(m.l + pw, py); g.stroke();
        g.fillText(pow10(e), m.l - 6, py + 4);
      }
    } else {
      const ys = spec.y.step || niceStep(spec.y.max - spec.y.min, Math.max(3, Math.floor(ph / 45)));
      for (let v = Math.ceil(spec.y.min / ys - 1e-9) * ys; v <= spec.y.max + 1e-9; v += ys) {
        const py = Math.round(Y(v)) + 0.5;
        g.beginPath(); g.moveTo(m.l, py); g.lineTo(m.l + pw, py); g.stroke();
        g.fillText(spec.y.fmt ? spec.y.fmt(v) : String(Math.round(v * 1000) / 1000), m.l - 6, py + 4);
      }
    }
    g.strokeStyle = C.axis; g.strokeRect(m.l + 0.5, m.t + 0.5, pw, ph);
    g.save(); g.beginPath(); g.rect(m.l, m.t, pw, ph); g.clip();
    (spec.hlines || []).forEach(l => { g.strokeStyle = l.color; g.setLineDash(l.dash || [3, 3]); g.beginPath(); g.moveTo(m.l, Y(l.y)); g.lineTo(m.l + pw, Y(l.y)); g.stroke(); g.setLineDash([]); });
    (spec.vlines || []).forEach(l => { g.strokeStyle = l.color; g.setLineDash(l.dash || [3, 3]); g.beginPath(); g.moveTo(X(l.x), m.t); g.lineTo(X(l.x), m.t + ph); g.stroke(); g.setLineDash([]); });
    (spec.series || []).forEach(s => {
      const ok = p => isFinite(p[0]) && isFinite(p[1]) && (!lx || p[1] > 0);
      g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = s.width || 1.8; g.setLineDash(s.dash || []);
      if (s.stem) {
        s.points.forEach(p => { if (!ok(p)) return; g.beginPath(); g.moveTo(X(p[0]), Y(s.base)); g.lineTo(X(p[0]), Y(p[1])); g.stroke(); g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.6, 0, 7); g.fill(); });
      } else if (s.dots) {
        g.globalAlpha = s.alpha || 0.75;
        s.points.forEach(p => { if (ok(p)) g.fillRect(X(p[0]) - 1, Y(p[1]) - 1, s.dots, s.dots); });
        g.globalAlpha = 1;
      } else {
        g.beginPath(); let pen = false;
        s.points.forEach(p => { if (!ok(p)) { pen = false; return; } if (pen) g.lineTo(X(p[0]), Y(p[1])); else { g.moveTo(X(p[0]), Y(p[1])); pen = true; } });
        g.stroke();
        if (s.marker) s.points.forEach(p => { if (ok(p)) { g.beginPath(); g.arc(X(p[0]), Y(p[1]), s.marker, 0, 7); s.hollow ? g.stroke() : g.fill(); } });
      }
      g.setLineDash([]);
    });
    g.restore();
    g.fillStyle = C.muted; g.textAlign = 'center'; g.font = '11.5px Inter, system-ui, sans-serif';
    g.fillText(spec.x.label || '', m.l + pw / 2, Hh - 6);
    g.save(); g.translate(13, m.t + ph / 2); g.rotate(-Math.PI / 2); g.fillText(spec.y.label || '', 0, 0); g.restore();
    const items = (spec.series || []).filter(s => s.label);
    if (items.length) {
      g.font = '11px Inter, system-ui, sans-serif';
      const lw = Math.max.apply(null, items.map(s => g.measureText(s.label).width)) + 34, lh = 16;
      const pos = spec.legend || 'bl';
      const bx = pos[1] === 'l' ? m.l + 8 : m.l + pw - lw - 8, by = pos[0] === 'b' ? m.t + ph - items.length * lh - 10 : m.t + 8;
      g.fillStyle = 'rgba(10,14,20,0.82)'; g.fillRect(bx, by, lw, items.length * lh + 6);
      g.strokeStyle = C.axis; g.strokeRect(bx + 0.5, by + 0.5, lw, items.length * lh + 6);
      items.forEach((s, i) => {
        const yy = by + 12 + i * lh;
        g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = 2; g.setLineDash(s.dash || []);
        if (s.dots || s.stem) { g.fillRect(bx + 10, yy - 3, 6, 6); } else { g.beginPath(); g.moveTo(bx + 6, yy); g.lineTo(bx + 22, yy); g.stroke(); }
        g.setLineDash([]); g.fillStyle = C.text; g.textAlign = 'left'; g.fillText(s.label, bx + 28, yy + 4);
      });
    }
  };

  /* ================================================================ DOM helpers */
  function h(tag, attrs, children) {
    const n = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === 'class') n.className = v; else if (k === 'text') n.textContent = v; else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else if (v !== false && v != null) n.setAttribute(k, v === true ? '' : v);
    });
    (children || []).forEach(c => n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return n;
  }
  function seg(label, options, value, onChange) {
    const wrap = h('div', { class: 'plab-field' }, [h('span', { class: 'plab-label', text: label })]);
    const row = h('div', { class: 'plab-seg', role: 'group', 'aria-label': label });
    const buttons = options.map(([v, text]) => {
      const b = h('button', { type: 'button', class: 'plab-segbtn' + (v === value ? ' on' : ''), 'aria-pressed': String(v === value), text });
      b.addEventListener('click', () => { if (b.disabled) return; buttons.forEach(o => { o.classList.remove('on'); o.setAttribute('aria-pressed', 'false'); }); b.classList.add('on'); b.setAttribute('aria-pressed', 'true'); onChange(v); });
      b._value = v;
      row.appendChild(b);
      return b;
    });
    wrap.appendChild(row);
    wrap._buttons = buttons;
    return wrap;
  }
  function range(label, min, max, step, value, fmt, onInput) {
    const out = h('output', { text: fmt(value) });
    const inp = h('input', { type: 'range', min, max, step, value, 'aria-label': label });
    inp.addEventListener('input', () => { out.textContent = fmt(+inp.value); onInput(+inp.value); });
    return h('label', { class: 'plab-field' }, [h('span', { class: 'plab-label' }, [label + ' ', out]), inp]);
  }
  function select(label, options, value, onChange) {
    const s = h('select', { 'aria-label': label });
    options.forEach(([v, text]) => { const o = h('option', { value: String(v), text }); if (v === value) o.selected = true; s.appendChild(o); });
    s.addEventListener('change', () => onChange(isNaN(+s.value) ? s.value : +s.value));
    return h('label', { class: 'plab-field' }, [h('span', { class: 'plab-label', text: label }), s]);
  }
  function stat(key) {
    const v = h('b', { class: 'plab-v', text: '...' }), s = h('span', { class: 'plab-s', text: '' });
    return { node: h('div', { class: 'plab-stat' }, [h('span', { class: 'plab-k', text: key }), v, s]), v, s };
  }
  const fmtDb = v => (isFinite(v) ? v.toFixed(2) : '...') + ' dB';
  const fmtBer = v => v === 0 ? '0' : v.toExponential(1).replace('e-', 'e-');

  /* ================================================================ transmitter view */
  function transmitterView(root, redrawers) {
    const st = { method: 'rcf', N: 128, L: 4, order: 4, cr: 3, passes: 4, R: 0.5, U: 16, V: 4, W: 2, seed: 7 };
    const METHOD = {
      rcf: 'Repeated clipping and filtering (thesis chapter 3)', clip: 'Clipping only, no filter (for comparison)',
      rct: 'Rooting companding |x|^R (thesis chapter 3)', slm: 'Selected mapping (my ETASR 2021 paper)', pts: 'Partial transmit sequences (my JASA 2019 and 2020 papers)',
    };
    const note = h('p', { class: 'plab-note' });
    const params = {
      cr: range('Clipping ratio CR', 1.5, 4, 0.25, st.cr, v => v.toFixed(2) + ' (' + (10 * Math.log10(v)).toFixed(2) + ' dB)', v => { st.cr = v; restart(); }),
      passes: seg('Clip and filter passes', [[1, '1'], [2, '2'], [3, '3'], [4, '4']], st.passes, v => { st.passes = v; restart(); }),
      R: range('Rooting exponent R', 0.3, 0.9, 0.05, st.R, v => v.toFixed(2), v => { st.R = v; restart(); }),
      U: seg('Candidates U', [[2, '2'], [4, '4'], [8, '8'], [16, '16'], [32, '32']], st.U, v => { st.U = v; restart(); }),
      V: seg('Subblocks V', [[2, '2'], [4, '4'], [8, '8']], st.V, v => { st.V = v; syncW(); restart(); }),
      W: seg('Phase factors W', [[2, '±1'], [4, '±1, ±j']], st.W, v => { st.W = v; restart(); }),
    };
    const show = { rcf: ['cr', 'passes'], clip: ['cr'], rct: ['R'], slm: ['U'], pts: ['V', 'W'] };
    function syncW() {
      const four = params.W._buttons[1];
      four.disabled = st.V === 8;
      if (st.V === 8 && st.W === 4) { st.W = 2; params.W._buttons[0].click(); }
    }
    function syncParams() {
      Object.entries(params).forEach(([k, node]) => { node.style.display = show[st.method].includes(k) ? '' : 'none'; });
      note.textContent = {
        rcf: 'Thesis setting: 128 subcarriers, oversampling I = 2, CR = 3 (4.77 dB) and four passes.',
        clip: 'One hard clip with no filter: the peaks go, but the spectrum spreads (see the spectrum plot).',
        rct: 'The receiver undoes the companding with |y|^(1/R), which also stretches the noise.',
        slm: 'The transmitter sends log2(U) bits of side information so the receiver can undo the chosen rotation.',
        pts: 'Exhaustive search over W^(V-1) phase combinations; side information is (V-1) log2(W) bits.',
      }[st.method];
    }
    const controls = h('div', { class: 'plab-controls' }, [
      seg('Method', [['rcf', 'RCF'], ['clip', 'Clipping'], ['rct', 'Rooting'], ['slm', 'SLM'], ['pts', 'PTS']], st.method, v => { st.method = v; syncParams(); restart(); }),
      select('Subcarriers N', [[64, '64'], [128, '128'], [256, '256'], [512, '512'], [1024, '1024']], st.N, v => { st.N = v; restart(); }),
      select('Oversampling I', [[1, '1 (Nyquist rate)'], [2, '2'], [4, '4 (accurate peaks)']], st.L, v => { st.L = v; restart(); }),
      select('Modulation', [[4, 'QPSK'], [16, '16-QAM'], [64, '64-QAM']], st.order, v => { st.order = v; restart(); }),
      params.cr, params.passes, params.R, params.U, params.V, params.W, note,
      h('button', { type: 'button', class: 'plab-btn', text: 'New random data', onclick: () => { st.seed++; restart(); } }),
    ]);
    const s1 = stat('PAPR at CCDF 10⁻³'), s2 = stat('Amplifier efficiency'), s3 = stat('In-band distortion'), s4 = stat('Cost');
    const ccdf = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'CCDF of the PAPR' });
    const progress = h('div', { class: 'plab-progress' });
    const env = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Envelope of one OFDM symbol' });
    const spec = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Power spectrum' });
    root.appendChild(h('div', { class: 'plab-layout' }, [controls, h('div', { class: 'plab-main' }, [
      h('div', { class: 'plab-stats' }, [s1.node, s2.node, s3.node, s4.node]), ccdf, progress])]));
    root.appendChild(h('div', { class: 'plab-pair' }, [env, spec]));
    const pCcdf = new Plot(ccdf, 300), pEnv = new Plot(env, 220), pSpec = new Plot(spec, 220);
    redrawers.push(() => { pCcdf.draw(); pEnv.draw(); pSpec.draw(); });

    let job = null;
    function restart() {
      if (job) job.cancelled = true;
      const size = st.N * st.L, q = qam(st.order), rand = mulberry32(1000 + st.seed * 7919 + st.N);
      const cands = st.method === 'slm' ? st.U : st.method === 'pts' ? Math.pow(st.W, st.V - 1) + st.V : st.method === 'rcf' ? 1 + 2 * st.passes : 1;
      const target = Math.min(4000, Math.max(2000, Math.floor(6e7 / (size * cands))));
      const phaseIndex = st.method === 'slm' ? Array.from({ length: st.U }, (_, u) => Array.from({ length: st.N }, () => (u === 0 ? 0 : Math.floor(rand() * 4)))) : null;
      job = { cancelled: false, done: 0, target, orig: [], proc: [], sxx: 0, sxy_re: 0, syy: 0, oob: 0, all: 0,
        specO: new Float64Array(size), specP: new Float64Array(size), specCount: 0, env: null, envPeak: -1 };
      const J = job;
      function step() {
        if (J.cancelled) return;
        const t0 = performance.now();
        while (J.done < J.target && performance.now() - t0 < 14) {
          const bits = randomBits(rand, st.N * q.k), X = modulate(bits, q);
          const x = toTime(X.re, X.im, size), pO = paprDb(x.re, x.im);
          let y, pP;
          if (st.method === 'rcf' || st.method === 'clip') {
            y = { re: Float64Array.from(x.re), im: Float64Array.from(x.im) };
            if (st.method === 'rcf') rcfInPlace(y.re, y.im, st.N, st.cr, st.passes); else clipInPlace(y.re, y.im, st.cr);
            pP = paprDb(y.re, y.im);
          } else if (st.method === 'rct') {
            y = { re: Float64Array.from(x.re), im: Float64Array.from(x.im) }; rootInPlace(y.re, y.im, st.R); pP = paprDb(y.re, y.im);
          } else if (st.method === 'slm') { const r = slmBest(X, phaseIndex, size); y = r.x; pP = r.papr; }
          else { const r = ptsBest(X, st.V, st.W, size); y = r.x; pP = r.papr; }
          J.orig.push(pO); J.proc.push(pP);
          if (st.method === 'rcf' || st.method === 'clip' || st.method === 'rct') {
            const Y = toFreq(y.re, y.im, st.N);
            for (let k = 0; k < st.N; k++) { J.sxx += X.re[k] * X.re[k] + X.im[k] * X.im[k]; J.sxy_re += X.re[k] * Y.re[k] + X.im[k] * Y.im[k]; J.syy += Y.re[k] * Y.re[k] + Y.im[k] * Y.im[k]; }
          }
          if (J.specCount < 300) {
            const a = Float64Array.from(x.re), b = Float64Array.from(x.im), c = Float64Array.from(y.re), d = Float64Array.from(y.im);
            fft(a, b, false); fft(c, d, false);
            const h2 = st.N >> 1;
            for (let k = 0; k < size; k++) {
              const po = a[k] * a[k] + b[k] * b[k], pp = c[k] * c[k] + d[k] * d[k];
              J.specO[k] += po; J.specP[k] += pp; J.all += pp;
              if (k >= h2 && k < size - (st.N - h2)) J.oob += pp;
            }
            J.specCount++;
          }
          if (J.done < 60 && pO > J.envPeak) { J.envPeak = pO; J.env = { x, y }; }
          J.done++;
        }
        render(J, size);
        if (J.done < J.target) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    function render(J, size) {
      const so = J.orig.slice().sort((a, b) => a - b), sp = J.proc.slice().sort((a, b) => a - b);
      const po = quantile(so, 1 - 1e-3), pp = quantile(sp, 1 - 1e-3);
      s1.v.textContent = fmtDb(pp);
      s1.s.textContent = 'from ' + fmtDb(po) + ', ' + (pp - po >= 0 ? '+' : '') + (pp - po).toFixed(2) + ' dB';
      const eo = classB(po), ep = classB(pp);
      s2.v.textContent = ep.toFixed(1) + '%';
      s2.s.textContent = 'from ' + eo.toFixed(1) + '%: ideal class B, battery ' + (ep / eo).toFixed(2) + 'x';
      if (st.method === 'slm' || st.method === 'pts') {
        s3.v.textContent = 'none';
        s3.s.textContent = 'distortionless: the data are only rotated';
        const bits = st.method === 'slm' ? Math.log2(st.U) : (st.V - 1) * Math.log2(st.W);
        s4.v.textContent = bits + ' side bits';
        s4.s.textContent = st.method === 'slm' ? st.U + ' IFFTs per symbol' : st.V + ' IFFTs, ' + Math.pow(st.W, st.V - 1) + ' combinations';
      } else {
        const a = J.sxy_re / J.sxx, dist = Math.max(J.syy - a * a * J.sxx, 1e-30), sdr = 10 * Math.log10(a * a * J.sxx / dist);
        s3.v.textContent = isFinite(sdr) ? sdr.toFixed(1) + ' dB SDR' : '...';
        s3.s.textContent = 'signal to distortion ratio on the subcarriers';
        const oob = J.all > 0 ? 10 * Math.log10(Math.max(J.oob / J.all, 1e-30)) : NaN;
        if (st.L === 1) { s4.v.textContent = 'hidden at I = 1'; s4.s.textContent = 'oversample to see out-of-band power'; }
        else if (J.oob / J.all < 1e-12) { s4.v.textContent = 'no regrowth'; s4.s.textContent = st.method === 'rcf' ? (1 + 2 * st.passes) + ' FFT operations per symbol' : ''; }
        else { s4.v.textContent = oob.toFixed(1) + ' dB'; s4.s.textContent = 'power pushed out of band (spectral regrowth)'; }
      }
      progress.textContent = 'Simulated ' + J.done.toLocaleString() + ' of ' + J.target.toLocaleString() + ' OFDM symbols, ' + st.N + ' subcarriers, ' + size + ' samples each';
      const series = [
        { points: ccdfCurve(so, GRID), color: C.cyan, label: 'Original OFDM' },
        { points: ccdfCurve(sp, GRID), color: C.orange, label: METHOD[st.method].split(' (')[0] },
      ];
      if (st.L !== 2) {
        const a = st.L === 1 ? 1 : 2.8;
        series.push({ points: GRID.map(z => [z, 1 - Math.pow(1 - Math.exp(-Math.pow(10, z / 10)), a * st.N)]), color: C.muted, dash: [4, 4], width: 1.2, label: st.L === 1 ? 'Theory, Nyquist rate' : 'Theory, 2.8N approximation' });
      }
      pCcdf.draw({ title: METHOD[st.method], x: { min: 2, max: 13, label: 'PAPR threshold z (dB)' }, y: { min: 1e-3, max: 1, log: true, label: 'Pr[PAPR > z]' }, series, legend: 'bl' });
      if (J.env) {
        const e = J.env, pts = (s, arr) => Array.from({ length: size }, (_, t) => [t / st.L, Math.hypot(arr.re[t], arr.im[t])]);
        const clipLevel = st.method === 'rcf' || st.method === 'clip' ? Math.sqrt(st.cr) : null;
        const ymax = Math.max(1, Math.ceil(Math.max.apply(null, pts(0, e.x).map(p => p[1])) + 0.2));
        pEnv.draw({ title: 'The worst of the first 60 symbols', x: { min: 0, max: st.N, label: 'sample (Nyquist rate units)' },
          y: { min: 0, max: ymax, label: '|x(t)| / RMS' },
          hlines: clipLevel ? [{ y: clipLevel, color: C.muted }] : [],
          series: [{ points: pts(0, e.x), color: C.cyan, width: 1.1, label: 'original' }, { points: pts(0, e.y), color: C.orange, width: 1.1, label: 'after ' + st.method.toUpperCase() }], legend: 'tr' });
      }
      if (J.specCount) {
        const specPts = arr => {
          const out = [], n = size;
          let ref = 0; for (let k = 0; k < n; k++) ref = Math.max(ref, J.specO[k]);
          for (let i = 0; i < n; i++) { const k = (i + n / 2) % n; out.push([(i - n / 2) / (n / 2), 10 * Math.log10(Math.max(arr[k] / ref, 1e-12))]); }
          return out;
        };
        pSpec.draw({ title: 'Average power spectrum', x: { min: -1, max: 1, label: 'frequency / (fs / 2)' }, y: { min: -60, max: 5, label: 'dB' },
          bands: [{ x0: -1 / st.L, x1: 1 / st.L, color: 'rgba(57,210,255,0.06)' }],
          series: [{ points: specPts(J.specP), color: C.orange, width: 1.1, label: 'after ' + st.method.toUpperCase() }, { points: specPts(J.specO), color: C.cyan, width: 1.1, label: 'original' }], legend: 'bl' });
      }
    }
    syncParams(); syncW();
    return { start: restart };
  }

  /* ================================================================ receiver view */
  function receiverView(root, data, redrawers) {
    const st = { channel: 'bellhop', order: 64, ibo: 5, snr: 30, passes: 3, model: 'net', seed: 3 };
    const U = data.uwa, N = U.subcarriers, size = N * 4;
    const Hb = channelResponse(data.channel.arrivals, N, U.bandwidth_hz / N, U.centre_hz);
    const Hf = { re: new Float64Array(N).fill(1), im: new Float64Array(N) };
    const nets = {}; Object.keys(data.networks).forEach(k => { nets[k] = new Network(data.networks[k].weights); });
    const controls = h('div', { class: 'plab-controls' }, [
      seg('Channel', [['awgn', 'White noise'], ['bellhop', 'BELLHOP 2 km']], st.channel, v => { st.channel = v; restart(); }),
      seg('Amplifier back-off', [[3, '3 dB'], [5, '5 dB'], [7, '7 dB']], st.ibo, v => { st.ibo = v; restart(); }),
      select('Modulation', [[4, 'QPSK'], [16, '16-QAM'], [64, '64-QAM']], st.order, v => { st.order = v; restart(); }),
      range('SNR per subcarrier', 10, 40, 1, st.snr, v => v + ' dB', v => { st.snr = v; restart(); }),
      seg('FFB passes', [[0, '0'], [1, '1'], [2, '2'], [3, '3']], st.passes, v => { st.passes = v; restart(); }),
      seg('Receiver model of the amplifier', [['net', 'Learned network'], ['exact', 'Exact equations']], st.model, v => { st.model = v; restart(); }),
      h('p', { class: 'plab-note', text: 'The network is the 6-12-6 model trained in Python on 25,800 measured amplifier samples; its 162 weights run here unchanged.' }),
      h('button', { type: 'button', class: 'plab-btn', text: 'New random data', onclick: () => { st.seed++; restart(); } }),
    ]);
    const s1 = stat('BER without FFB'), s2 = stat('BER with FFB'), s3 = stat('Improvement');
    const cBer = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Bit error rate curves' });
    const progress = h('div', { class: 'plab-progress' });
    const cA = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Constellation before FFB' });
    const cB = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Constellation after FFB' });
    root.appendChild(h('div', { class: 'plab-layout' }, [controls, h('div', { class: 'plab-main' }, [h('div', { class: 'plab-stats three' }, [s1.node, s2.node, s3.node]), cBer, progress])]));
    root.appendChild(h('div', { class: 'plab-pair' }, [cA, cB]));
    const pBer = new Plot(cBer, 300), pA = new Plot(cA, 260), pB = new Plot(cB, 260);
    redrawers.push(() => { pBer.draw(); pA.draw(); pB.draw(); });
    let job = null;
    function restart() {
      if (job) job.cancelled = true;
      const q = qam(st.order), rand = mulberry32(99 + st.seed * 104729), gauss = gaussian(rand);
      const H = st.channel === 'awgn' ? Hf : Hb, info = data.networks[String(st.ibo)], sat = Math.pow(10, st.ibo / 20);
      const model = st.model === 'net' ? { kind: 'net', net: nets[String(st.ibo)] } : { kind: 'exact', sat };
      const alpha = st.model === 'net' ? info.alpha_network : info.alpha_exact;
      const target = st.order === 4 ? 400 : 240;
      job = { cancelled: false, done: 0, target, bits: 0, e0: 0, e1: 0, before: [], after: [] };
      const J = job;
      function step() {
        if (J.cancelled) return;
        const t0 = performance.now();
        while (J.done < J.target && performance.now() - t0 < 16) {
          const batch = 4, bitsList = [], Ys = [];
          const Ytx = [];
          for (let b = 0; b < batch; b++) {
            const bits = randomBits(rand, N * q.k), X = modulate(bits, q), x = toTime(X.re, X.im, size);
            rappInPlace(x.re, x.im, sat, 2);
            bitsList.push(bits); Ytx.push(toFreq(x.re, x.im, N));
          }
          let es = 0; Ytx.forEach(Y => { for (let k = 0; k < N; k++) es += Y.re[k] * Y.re[k] + Y.im[k] * Y.im[k]; });
          es /= batch * N;
          const sd = Math.sqrt(es / Math.pow(10, st.snr / 10) / 2);
          Ytx.forEach(Y => {
            const r = new Float64Array(N), i = new Float64Array(N);
            for (let k = 0; k < N; k++) { r[k] = H.re[k] * Y.re[k] - H.im[k] * Y.im[k] + sd * gauss(); i[k] = H.re[k] * Y.im[k] + H.im[k] * Y.re[k] + sd * gauss(); }
            Ys.push({ re: r, im: i });
          });
          const out = ffb(Ys, H, q, model, alpha, size, st.passes);
          bitsList.forEach((bits, s) => {
            J.e0 += countBitErrors(out.first[s].re, out.first[s].im, bits, q);
            J.e1 += countBitErrors(out.last[s].re, out.last[s].im, bits, q);
            J.bits += bits.length;
          });
          if (J.before.length < 2400) {
            const z = out.soft[0];
            for (let k = 0; k < N && J.before.length < 2400; k += 1) {
              const d = H.re[k] * H.re[k] + H.im[k] * H.im[k];
              if (st.channel === 'bellhop' && d < 0.1) continue;
              const zr = (Ys[0].re[k] * H.re[k] + Ys[0].im[k] * H.im[k]) / d / alpha, zi = (Ys[0].im[k] * H.re[k] - Ys[0].re[k] * H.im[k]) / d / alpha;
              J.before.push([zr, zi]); J.after.push([z.re[k], z.im[k]]);
            }
          }
          J.done += batch;
        }
        render(J);
        if (J.done < J.target) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    function render(J) {
      const b0 = J.e0 / J.bits, b1 = J.e1 / J.bits;
      s1.v.textContent = fmtBer(b0); s1.s.textContent = J.e0.toLocaleString() + ' errors in ' + J.bits.toLocaleString() + ' bits';
      s2.v.textContent = fmtBer(b1); s2.s.textContent = st.passes + ' pass' + (st.passes === 1 ? '' : 'es') + ', ' + (st.model === 'net' ? 'learned network' : 'exact equations');
      s3.v.textContent = J.e1 === 0 ? (J.e0 ? 'all errors removed' : 'no errors') : (b0 / b1).toFixed(1) + 'x fewer errors';
      s3.s.textContent = J.e1 === 0 && J.e0 ? 'in ' + J.bits.toLocaleString() + ' bits' : 'live Monte Carlo at ' + st.snr + ' dB';
      progress.textContent = 'Simulated ' + J.done + ' of ' + J.target + ' OFDM symbols (' + N + ' subcarriers, 6.25 kHz, 4x oversampled amplifier)';
      const rc = data.receiver, snr = rc.snr_db, key = st.channel + '_ibo' + st.ibo;
      const curve = name => rc.curves[name] ? snr.map((s, i) => [s, rc.curves[name][i]]) : null;
      const series = [];
      if (st.order === 64) {
        const lin = curve(st.channel + '_linear'), none = curve(key + '_none'), nn = curve(key + '_ffb_nn_3');
        if (none) series.push({ points: none, color: C.red, marker: 2.5, label: 'Python: without FFB' });
        if (nn) series.push({ points: nn, color: C.green, marker: 2.5, label: 'Python: FFB, learned model' });
        if (lin) series.push({ points: lin, color: C.cyan, dash: [5, 4], label: 'Python: linear amplifier' });
      }
      series.push({ points: [[st.snr, Math.max(b0, 1e-7)]], color: C.red, marker: 6, hollow: true, width: 2, label: 'live: without FFB' });
      series.push({ points: [[st.snr, Math.max(b1, 1e-7)]], color: C.green, marker: 6, hollow: true, width: 2, label: 'live: with FFB' });
      pBer.draw({ title: st.order === 64 ? '64-QAM bit error rate (curves: Python, 400 symbols per point)' : 'Live bit error rate (Python curves are for 64-QAM)',
        x: { min: 10, max: 40, label: 'SNR per subcarrier (dB)' }, y: { min: 1e-6, max: 0.5, log: true, label: 'bit error rate' }, series, legend: 'bl' });
      const lim = st.order === 4 ? 1.4 : 1.45;
      const scatter = (pts, color, title) => ({ title, x: { min: -lim, max: lim, step: 0.5, label: 'in phase' }, y: { min: -lim, max: lim, step: 0.5, label: 'quadrature' }, series: [{ points: pts, dots: 2, color, alpha: 0.7 }] });
      pA.draw(scatter(J.before, C.red, 'Before FFB (equalised, one symbol at a time)'));
      pB.draw(scatter(J.after, C.green, 'After ' + st.passes + ' FFB pass' + (st.passes === 1 ? '' : 'es')));
    }
    return { start: restart };
  }

  /* ================================================================ channel view */
  function channelView(root, data, redrawers) {
    const ch = data.channel, g = ch.geometry, U = data.uwa;
    const T = U.fft_points / U.fs_hz * 1000;   // 81.92 ms OFDM symbol
    const st = { guard: 50 };
    const s1 = stat('Energy after the guard'), s2 = stat('Guard overhead'), s3 = stat('First arrival check');
    const cGeo = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Channel geometry and ray paths' });
    const cArr = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'BELLHOP arrivals' });
    const cH = h('canvas', { class: 'plab-canvas', role: 'img', 'aria-label': 'Channel frequency response' });
    const controls = h('div', { class: 'plab-controls' }, [
      range('Guard interval', 5, 150, 1, st.guard, v => v + ' ms', v => { st.guard = v; draw(); }),
      h('p', { class: 'plab-note', html: 'BELLHOP channel of the chapter 4 link: ' + g.depth_m + ' m of water, source at ' + g.source_m + ' m, hydrophone at ' + g.receiver_m + ' m, ' + (g.range_m / 1000) + ' km apart, ' + (g.freq_hz / 1000) + ' kHz, sandy bottom (' + g.bottom.speed + ' m/s).' }),
      h('p', { class: 'plab-note', text: 'Each OFDM symbol lasts ' + T.toFixed(2) + ' ms. Slide the guard interval to see how much of the multipath it captures and what it costs in airtime.' }),
    ]);
    root.appendChild(h('div', { class: 'plab-layout' }, [controls, h('div', { class: 'plab-main' }, [h('div', { class: 'plab-stats three' }, [s1.node, s2.node, s3.node]), cGeo])]));
    root.appendChild(h('div', { class: 'plab-pair' }, [cArr, cH]));
    const pArr = new Plot(cArr, 240), pH = new Plot(cH, 240);
    function geometry() {
      const cv = cGeo, dpr = window.devicePixelRatio || 1, W = Math.max(260, cv.clientWidth || 600), Hh = 230;
      cv.width = W * dpr; cv.height = Hh * dpr; cv.style.height = Hh + 'px';
      const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, Hh);
      const m = { l: 44, r: 14, t: 16, b: 30 }, pw = W - m.l - m.r, ph = Hh - m.t - m.b, D = g.depth_m, R = g.range_m;
      const X = x => m.l + x / R * pw, Y = z => m.t + z / D * ph;
      const grad = c.createLinearGradient(0, m.t, 0, m.t + ph); grad.addColorStop(0, '#0f2a3f'); grad.addColorStop(1, '#0a1622');
      c.fillStyle = grad; c.fillRect(m.l, m.t, pw, ph);
      c.fillStyle = '#3a2f22'; c.fillRect(m.l, m.t + ph, pw, 6);
      c.strokeStyle = C.axis; c.strokeRect(m.l + 0.5, m.t + 0.5, pw, ph);
      const strongest = ch.arrivals.slice().sort((a, b) => b.amp - a.amp).slice(0, 9);
      strongest.forEach(a => {
        const down = a.angle > 0, nt = a.top, nb = a.bottom, zs = g.source_m, zr = g.receiver_m;
        let V;
        if (nt + nb === 0) V = Math.abs(zr - zs);
        else {
          const firstPart = down ? D - zs : zs, lastBottom = down ? nb > nt : nb >= nt;
          V = firstPart + (nt + nb - 1) * D + (lastBottom ? D - zr : zr);
        }
        c.strokeStyle = 'rgba(57,210,255,' + (0.25 + 0.65 * a.amp).toFixed(2) + ')'; c.lineWidth = 1 + 1.2 * a.amp;
        c.beginPath(); c.moveTo(X(0), Y(zs));
        let z = zs, dir = down ? 1 : -1, x = 0; const slope = V / R;
        while (x < R - 1e-6) {
          const toWall = dir > 0 ? (D - z) : z, dx = Math.min(toWall / slope, R - x);
          x += dx; z += dir * dx * slope; c.lineTo(X(x), Y(z));
          if (x < R - 1e-6) dir = -dir;
        }
        c.stroke();
      });
      c.fillStyle = C.orange; c.beginPath(); c.arc(X(0), Y(g.source_m), 5, 0, 7); c.fill();
      c.fillStyle = C.green; c.fillRect(X(R) - 5, Y(g.receiver_m) - 5, 10, 10);
      c.font = '11px Inter, system-ui, sans-serif';
      const tag = (text, x, y, align, color) => {
        c.textAlign = align;
        const w = c.measureText(text).width, x0 = align === 'right' ? x - w : x;
        c.fillStyle = 'rgba(10,14,20,0.78)'; c.fillRect(x0 - 3, y - 11, w + 6, 15);
        c.fillStyle = color; c.fillText(text, x, y);
      };
      tag('source ' + g.source_m + ' m', X(0) + 9, Y(g.source_m) - 8, 'left', C.orange);
      tag('hydrophone ' + g.receiver_m + ' m', X(R) - 9, Y(g.receiver_m) - 8, 'right', C.green);
      tag('surface', m.l + 4, m.t + 13, 'left', C.muted);
      tag('sandy bottom, ' + D + ' m', m.l + 4, m.t + ph - 5, 'left', C.muted);
      c.fillStyle = C.muted; c.textAlign = 'left'; c.fillText('0 km', m.l, m.t + ph + 18);
      c.textAlign = 'right'; c.fillText((R / 1000) + ' km', m.l + pw, m.t + ph + 18);
      c.textAlign = 'center';
      c.fillText(W < 560 ? '9 strongest of ' + ch.arrivals.length + ' paths, straight segments' : '9 strongest of ' + ch.arrivals.length + ' paths as straight segments; BELLHOP bends them with the sound speed', m.l + pw / 2, m.t + ph + 18);
    }
    function draw() {
      const E = ch.arrivals.reduce((s, a) => s + a.amp * a.amp, 0);
      const late = ch.arrivals.reduce((s, a) => s + (a.delay_ms > st.guard ? a.amp * a.amp : 0), 0) / E;
      s1.v.textContent = (100 * late).toFixed(late < 0.01 ? 2 : 1) + '%';
      s1.s.textContent = late > 0.05 ? 'spills into the next symbol (intersymbol interference)' : 'small enough to treat each subcarrier on its own';
      s2.v.textContent = (100 * st.guard / (T + st.guard)).toFixed(0) + '% of airtime';
      s2.s.textContent = st.guard + ' ms guard on ' + T.toFixed(1) + ' ms symbols';
      s3.v.textContent = (ch.first_arrival_error_pct >= 0 ? '+' : '') + ch.first_arrival_error_pct.toFixed(3) + '%';
      s3.s.textContent = ch.first_arrival_s.toFixed(4) + ' s vs ' + ch.straight_line_s.toFixed(4) + ' s straight line';
      pArr.draw({ title: 'BELLHOP arrivals (rms delay spread ' + ch.rms_delay_ms.toFixed(1) + ' ms)', x: { min: 0, max: 240, label: 'delay after the first arrival (ms)' },
        y: { min: -50, max: 3, label: 'amplitude (dB)' }, bands: [{ x0: 0, x1: st.guard, color: 'rgba(255,176,32,0.12)' }],
        series: [{ points: ch.arrivals.map(a => [a.delay_ms, 20 * Math.log10(a.amp)]), stem: true, base: -50, color: C.cyan }] });
      const f = ch.response_db.freq_hz, mag = ch.response_db.mag_db;
      pH.draw({ title: 'Frequency response across the band', x: { min: f[0] / 1000, max: f[f.length - 1] / 1000, label: 'frequency (kHz)' },
        y: { min: -30, max: 10, label: '|H(f)| (dB)' }, series: [{ points: f.map((v, i) => [v / 1000, mag[i]]), color: C.violet, width: 1.2 }] });
    }
    redrawers.push(() => { geometry(); pArr.draw(); pH.draw(); });
    return { start: () => { geometry(); draw(); } };
  }

  /* ================================================================ mount */
  function mount(root, data) {
    root.innerHTML = '';
    const checks = verify(data), passed = checks.filter(c => c.ok).length;
    const badge = h('details', { class: 'plab-check' + (passed === checks.length ? ' ok' : ' bad') }, [
      h('summary', { text: (passed === checks.length ? '✓ ' : '✗ ') + 'Browser engine matches the Python reference: ' + passed + '/' + checks.length + ' checks' }),
      h('ul', {}, checks.map(c => h('li', { text: (c.ok ? '✓ ' : '✗ ') + c.name + ' (' + c.detail + ')' }))),
    ]);
    const tabs = [['tx', 'Transmitter', 'PAPR'], ['rx', 'Receiver', 'FFB'], ['ch', 'Channel', 'BELLHOP']];
    const bar = h('div', { class: 'plab-tabs', role: 'tablist' });
    const views = {}, redrawers = [], started = {};
    const ctl = {};
    root.appendChild(badge); root.appendChild(bar);
    tabs.forEach(([k, label, sub], i) => {
      const view = h('div', { class: 'plab-view', role: 'tabpanel', id: 'plab-' + k });
      if (i) view.hidden = true;
      views[k] = view;
      const b = h('button', { type: 'button', role: 'tab', class: 'plab-tab' + (i ? '' : ' on'), 'aria-selected': String(!i), 'aria-controls': 'plab-' + k }, [label, h('span', { class: 'plab-tab-sub', text: ': ' + sub })]);
      b.addEventListener('click', () => {
        bar.querySelectorAll('.plab-tab').forEach(t => { t.classList.remove('on'); t.setAttribute('aria-selected', 'false'); });
        b.classList.add('on'); b.setAttribute('aria-selected', 'true');
        Object.entries(views).forEach(([kk, v]) => { v.hidden = kk !== k; });
        if (!started[k]) { started[k] = true; ctl[k].start(); } else redrawers.forEach(fn => fn());
      });
      bar.appendChild(b);
      root.appendChild(view);
    });
    ctl.tx = transmitterView(views.tx, redrawers);
    ctl.rx = receiverView(views.rx, data, redrawers);
    ctl.ch = channelView(views.ch, data, redrawers);
    started.tx = true; ctl.tx.start();
    let t = null;
    const onResize = () => { clearTimeout(t); t = setTimeout(() => redrawers.forEach(fn => fn()), 120); };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(root); else window.addEventListener('resize', onResize);
    return { checks };
  }

  window.PaprLab = { mount, verify, _engine: { fft, toTime, toFreq, paprDb, qam, modulate, channelResponse } };
})();
