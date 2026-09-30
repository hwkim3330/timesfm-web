// timesfm.js - run Google's TimesFM 2.5 (200M) in the browser with onnxruntime-web.
// Pre/post-processing mirrors timesfm 2.5 `forecast()` with ForecastConfig(max_context=512, max_horizon=128,
// normalize_inputs, use_continuous_quantile_head, force_flip_invariance, infer_is_positive, fix_quantile_crossing).
// Model: Apache-2.0, (c) Google LLC. See LICENSE / NOTICE.
const ORT_VER = '1.30.0';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const HERE = new URL('.', import.meta.url).href;
const CACHE = 'timesfm-web-v1';

async function fetchCached(url, onBytes) {
  let cache = null;
  try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length, true); return b; } } catch (e) { cache = null; }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const reader = r.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes(value.length, false); }
  const b = new Uint8Array(n); let o = 0; for (const c of chunks) { b.set(c, o); o += c.length; }
  if (cache) try { await cache.put(url, new Response(b)); } catch (e) { /* quota: skip caching */ }
  return b;
}

// Returns {backend, forecast(seriesList, horizon), manifest, loadMs}. onProgress({loaded,total,cached}).
export async function loadTimesFM({ base = HERE + 'model/', onProgress = () => {}, backend = 'auto' } = {}) {
  const t0 = performance.now();
  const man = await (await fetch(base + 'manifest.json')).json();
  const total = man.model_bytes + man.data_bytes; let loaded = 0, cachedAll = true;
  const tick = (k, cached) => { loaded += k; cachedAll = cachedAll && cached; onProgress({ loaded, total, cached: cachedAll }); };
  const ortP = import(ORT_URL + 'ort.webgpu.min.mjs');
  const graph = fetchCached(base + man.model, tick);
  const data = new Uint8Array(man.data_bytes); let off = 0;
  const parts = await Promise.all(man.data_parts.map((p) => fetchCached(base + p, tick)));
  for (const p of parts) { data.set(p, off); off += p.length; }
  const ort = await ortP; ort.env.wasm.wasmPaths = ORT_URL;
  let useGpu = backend === 'webgpu' || (backend === 'auto' && 'gpu' in navigator && !!(await navigator.gpu.requestAdapter().catch(() => null)));
  const opts = (ep) => ({ executionProviders: [ep], externalData: [{ path: man.data_name, data }], graphOptimizationLevel: 'all' });
  let sess;
  if (useGpu) { try { sess = await ort.InferenceSession.create(await graph, opts('webgpu')); } catch (e) { console.warn('TimesFM: WebGPU failed, using WASM', e); useGpu = false; } }
  if (!sess) sess = await ort.InferenceSession.create(await graph, opts('wasm'));
  const C = man.context;

  // seriesList: array of number arrays (oldest -> newest). Returns [{mean, q:[9][h] (q10..q90), median}] per series.
  async function forecast(seriesList, horizon = 20) {
    if (horizon > man.horizon) throw new Error('horizon > ' + man.horizon);
    const B = seriesList.length, X = new Float32Array(2 * B * C), M = new Float32Array(2 * B * C), meta = [];
    seriesList.forEach((s, b) => {
      const v = s.filter((x) => Number.isFinite(x)).slice(-C), pad = C - v.length;
      const x = new Float64Array(C); v.forEach((y, i) => { x[pad + i] = y; });
      let mu = 0; for (let i = 0; i < C; i++) mu += x[i]; mu /= C;
      let ss = 0; for (let i = 0; i < C; i++) ss += (x[i] - mu) ** 2; const sd = Math.sqrt(ss / (C - 1));
      const div = sd < 1e-6 ? 1 : sd, pos = x.every((y) => y >= 0);
      for (let i = 0; i < C; i++) {
        const z = (x[i] - mu) / div, m = i < pad ? 1 : 0;
        X[(2 * b) * C + i] = z; X[(2 * b + 1) * C + i] = -z; M[(2 * b) * C + i] = m; M[(2 * b + 1) * C + i] = m;
      }
      meta.push({ mu, sd, pos });
    });
    const out = await sess.run({ inputs: new ort.Tensor('float32', X, [2 * B, C]), masks: new ort.Tensor('float32', M, [2 * B, C]) });
    const PT = out.point.data, QS = out.qspread.data, H = man.horizon, Q = 10;
    const at = (A, r, t, q) => A[(r * H + t) * Q + q];
    const flipq = (q) => (q === 0 ? 0 : Q - q); // flip_quantile_fn
    return meta.map(({ mu, sd, pos }, b) => {
      const f = [];
      for (let t = 0; t < horizon; t++) {
        const row = new Array(Q), qs = new Array(Q);
        for (let q = 0; q < Q; q++) {
          row[q] = (at(PT, 2 * b, t, q) - at(PT, 2 * b + 1, t, flipq(q))) / 2;
          qs[q] = (at(QS, 2 * b, t, q) - at(QS, 2 * b + 1, t, flipq(q))) / 2;
        }
        for (const q of [1, 2, 3, 4, 6, 7, 8, 9]) row[q] = qs[q] - qs[5] + row[5]; // continuous quantile head
        for (const q of [4, 3, 2, 1]) row[q] = Math.min(row[q], row[q + 1]); // fix quantile crossing
        for (const q of [6, 7, 8, 9]) row[q] = Math.max(row[q], row[q - 1]);
        f.push(row.map((y) => { const r = y * sd + mu; return pos ? Math.max(r, 0) : r; }));
      }
      return { mean: f.map((r) => r[0]), median: f.map((r) => r[5]), q: [1, 2, 3, 4, 5, 6, 7, 8, 9].map((q) => f.map((r) => r[q])) };
    });
  }
  return { backend: useGpu ? 'webgpu' : 'wasm', forecast, manifest: man, loadMs: performance.now() - t0, sizeBytes: total };
}
