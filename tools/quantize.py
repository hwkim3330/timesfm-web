"""Weight-only quantization of the fp32 ONNX: MatMulNBits (4-bit and 8-bit, block 32) and fp16."""
import onnx, sys
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
src = "build/timesfm-2.5-200m-fp32.onnx"
for bits in (4, 8):
  m = onnx.load(src)
  cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=bits)
  q = MatMulNBitsQuantizer(m, algo_config=cfg); q.process()
  onnx.save_model(q.model.model, f"build/timesfm-2.5-200m-q{bits}.onnx")
  print("q", bits, "done")
from onnxruntime.transformers.float16 import convert_float_to_float16
m = onnx.load(src)
onnx.save_model(convert_float_to_float16(m, keep_io_types=True), "build/timesfm-2.5-200m-fp16.onnx")
