"""Compare ONNX (onnxruntime, CPU) against the official PyTorch TimesFM 2.5 forecast() on real series.
usage: compare.py model.onnx [more.onnx ...]"""
import sys, json, time, numpy as np, onnxruntime as ort, torch, timesfm
C, H = 512, 128
series = json.load(open("build/series.json"))
# exercise the padding/mask path too: shorten a few series
names = list(series)
for k in names[-5:]: series[k] = series[k][-(250 + 30 * names.index(k) % 200):]
inputs = [np.array(series[k], dtype=np.float64) for k in names]

def prep(v):
  v = v[-C:]; w = len(v)
  return np.pad(v, (C - w, 0)).astype(np.float32), np.array([1.0] * (C - w) + [0.0] * w, np.float32)

def flip(x): return np.concatenate([x[..., :1], x[..., 1:][..., ::-1]], -1)

def onnx_forecast(sess, v):  # mirrors forecast.js exactly
  x, m = prep(v)
  pos = bool(np.all(x >= 0))
  mu = x.mean(dtype=np.float64); sd = x.std(ddof=1, dtype=np.float64)
  xn = ((x - mu) / (1.0 if sd < 1e-6 else sd)).astype(np.float32)
  pt, qs = sess.run(None, {"inputs": np.stack([xn, -xn]), "masks": np.stack([m, m])})
  full = (pt[0] - flip(pt[1])) / 2; q = (qs[0] - flip(qs[1])) / 2
  for i in [1, 2, 3, 4, 6, 7, 8, 9]: full[:, i] = q[:, i] - q[:, 5] + full[:, 5]
  full = full[:H].copy()
  for i in [4, 3, 2, 1]: full[:, i] = np.minimum(full[:, i], full[:, i + 1])
  for i in [6, 7, 8, 9]: full[:, i] = np.maximum(full[:, i], full[:, i - 1])
  full = full * sd + mu
  return np.maximum(full, 0) if pos else full

tm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
tm.compile(timesfm.ForecastConfig(max_context=C, max_horizon=H, normalize_inputs=True, use_continuous_quantile_head=True,
                                  force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
t = time.time(); _, ref = tm.forecast(horizon=H, inputs=[v.copy() for v in inputs]); print(f"torch: {time.time()-t:.2f}s for {len(inputs)}")
np.save("build/ref.npy", ref)
for path in sys.argv[1:]:
  so = ort.SessionOptions(); t = time.time()
  sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); tl = time.time() - t
  t = time.time(); out = np.stack([onnx_forecast(sess, v) for v in inputs]); ti = time.time() - t
  rel = np.abs(out - ref) / np.abs(ref).mean(axis=(1, 2), keepdims=True)  # relative to series level
  last = np.array([v[-1] for v in inputs])[:, None]
  def st(i, hz=H):
    d = np.abs(out[:, :hz, i] - ref[:, :hz, i]) / last * 100
    return f"max {d.max():.4f}% mean {d.mean():.5f}%"
  print(f"{path}: load {tl:.1f}s infer {ti:.2f}s | diff as % of last close (h=128): median {st(5)}; q10 {st(1)}; q90 {st(9)}; all-q {np.abs(out-ref).max(axis=(1,2)).__truediv__(last[:,0]).max()*100:.4f}% | h=20 median {st(5,20)}")
  np.save(path + ".out.npy", out)
