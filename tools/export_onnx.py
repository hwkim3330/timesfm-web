"""Export TimesFM 2.5 200M (google/timesfm-2.5-200m-pytorch) to ONNX.

The graph = one prefill pass of TimesFM_2p5_200M_torch_module.decode() for a
fixed context C (horizon <= 128, so no autoregressive steps are needed):
running per-patch stats -> RevIN -> transformer -> un-RevIN.
Inputs : inputs [B,C] float32, masks [B,C] float32 (1 = padded / missing)
Outputs: point [B,128,10]  (last patch output head, channel 0 = mean, 1..9 = q10..q90)
         qspread [B,128,10] (continuous quantile head, first 128 steps)
ForecastConfig-level post-processing (normalize_inputs, flip invariance,
continuous quantile head, quantile crossing fix, positivity) is done by the caller (JS).
"""
import sys, torch
from huggingface_hub import hf_hub_download
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from timesfm.torch import util, transformer

C = int(sys.argv[1]) if len(sys.argv) > 1 else 512
OUT = sys.argv[2] if len(sys.argv) > 2 else "build/timesfm-2.5-200m-fp32.onnx"
revin = util.revin

class Wrapper(torch.nn.Module):
  def __init__(self, m):
    super().__init__(); self.m = m
  def forward(self, inputs, masks):
    m = self.m; B = inputs.shape[0]; P = C // m.p
    x = inputs.reshape(B, P, m.p); mk = masks.reshape(B, P, m.p) > 0.5
    n = torch.zeros_like(inputs[:, 0]); mu = torch.zeros_like(n); sg = torch.zeros_like(n)
    mus, sgs = [], []
    for i in range(P):
      (n, mu, sg), _ = util.update_running_stats(n, mu, sg, x[:, i], mk[:, i])
      mus.append(mu); sgs.append(sg)
    cmu = torch.stack(mus, 1); csg = torch.stack(sgs, 1)
    xn = torch.where(mk, 0.0, revin(x, cmu, csg, reverse=False))
    h = m.tokenizer(torch.cat([xn, mk.to(xn.dtype)], -1))
    pm = mk[..., -1]
    for layer in m.stacked_xf:
      h, _ = layer(h, pm, None)
    last = h[:, -1]
    lmu = cmu[:, -1:]; lsg = csg[:, -1:]
    pt = m.output_projection_point(last) * lsg + lmu            # [B,1280]
    qs = m.output_projection_quantiles(last) * lsg + lmu        # [B,10240]
    return pt.reshape(B, m.o, m.q), qs.reshape(B, m.os, m.q)[:, :128]

if __name__ == "__main__":
  m = TimesFM_2p5_200M_torch_module()
  m.load_checkpoint(hf_hub_download("google/timesfm-2.5-200m-pytorch", "model.safetensors"))
  for layer in m.stacked_xf:  # explicit masked softmax (same math as the fused SDPA path, export-friendly)
    layer.attn.attention_fn = transformer._dot_product_attention
  w = Wrapper(m).eval()
  x = torch.randn(2, C).cumsum(-1) + 100; mk = torch.zeros(2, C)
  import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)
  torch.onnx.export(w, (x, mk), OUT, input_names=["inputs", "masks"], output_names=["point", "qspread"],
                    dynamic_axes={"inputs": {0: "batch"}, "masks": {0: "batch"}, "point": {0: "batch"}, "qspread": {0: "batch"}},
                    opset_version=17, dynamo=False, external_data=True)
  print("exported", OUT)
