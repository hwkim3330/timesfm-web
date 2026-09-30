# timesfm-web

Google Research's **TimesFM 2.5 (200M)** time-series foundation model, converted to ONNX so it runs **locally in the browser** with [onnxruntime-web](https://onnxruntime.ai/) on WebGPU (WASM fallback). Files are served from GitHub Pages (CORS `*`).

Demo: https://hwkim3330.github.io/timesfm-web/

Source model: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch), Apache-2.0 (see `LICENSE`, `NOTICE`). TimesFM 3 is *not* used (non-commercial licence).

## Files

| file | size |
|---|---|
| `model/timesfm-2.5-200m-q8.onnx` (graph) | 0.8 MB |
| `model/timesfm-2.5-200m-q8.onnx.data.{0,1,2}` (weights, split < 100 MB for git; concatenated in JS) | 260.5 MB total |
| `model/manifest.json` | sizes, parts, sha256 |
| `timesfm.js` | loader (Cache API) + pre/post-processing |

Graph: context **512** points, output **128** steps, inputs `inputs[B,512]`, `masks[B,512]` (1 = left padding), outputs `point[B,128,10]`, `qspread[B,128,10]` (mean, q10…q90). `timesfm.js` reproduces `ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)`.

## Accuracy (vs official PyTorch `forecast()`, 20 KRX series, 5 with < 512 points)

Max / mean abs difference as % of the last close:

| variant | size | median h=20 | median h=128 | any quantile h=128 |
|---|---|---|---|---|
| fp32 ONNX | 926 MB | 0.0001 / 0.00001 % | 0.0003 / 0.00003 % | 0.0004 % |
| fp16 | 464 MB | 0.12 / 0.025 % | 0.24 / 0.031 % | 0.56 % |
| **q8 (shipped)** | **261 MB** | **0.24 / 0.049 %** | 1.15 / 0.11 % | 1.24 % |
| q4 | 146 MB | 3.1 / 0.74 % | 10.0 / 1.57 % | 18 % (rejected) |

Browser (Chrome, M4 Max) output matches Python onnxruntime q8 to < 0.0001 % (WebGPU and WASM). Inference for 3 series: ~40 ms WebGPU (warm; first run ~0.8 s shader compile), ~1 s WASM.

## Usage

```js
import { loadTimesFM } from 'https://hwkim3330.github.io/timesfm-web/timesfm.js';
const tfm = await loadTimesFM({ onProgress: ({ loaded, total }) => console.log(loaded / total) });
const [f] = await tfm.forecast([closes], 20);   // f.median, f.mean, f.q[0] (q10) … f.q[8] (q90)
console.log(tfm.backend);                          // 'webgpu' or 'wasm'
```

Rebuild: `tools/export_onnx.py` → `tools/quantize.py` → `tools/compare.py build/*.onnx` → `tools/package.py q8`.
