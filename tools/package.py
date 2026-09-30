"""Write build/<variant>.onnx as graph + external data, split data into <90 MB parts for GitHub Pages."""
import onnx, os, sys, json, hashlib
v = sys.argv[1] if len(sys.argv) > 1 else "q8"
name = f"timesfm-2.5-200m-{v}"; out = "model"; os.makedirs(out, exist_ok=True)
m = onnx.load(f"build/{name}.onnx")
onnx.save_model(m, f"{out}/{name}.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                location=f"{name}.onnx.data", size_threshold=1024)
data = open(f"{out}/{name}.onnx.data", "rb").read(); os.remove(f"{out}/{name}.onnx.data")
CH = 90 * 1024 * 1024; parts = []
for i in range(0, len(data), CH):
  p = f"{name}.onnx.data.{i // CH}"; open(f"{out}/{p}", "wb").write(data[i:i + CH]); parts.append(p)
man = {"model": f"{name}.onnx", "model_bytes": os.path.getsize(f"{out}/{name}.onnx"), "data_name": f"{name}.onnx.data",
       "data_parts": parts, "data_bytes": len(data), "context": 512, "horizon": 128,
       "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "sha256_data": hashlib.sha256(data).hexdigest()}
json.dump(man, open(f"{out}/manifest.json", "w"), indent=1); print(man)
