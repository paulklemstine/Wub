"""Export the bar-arranger GPT to ONNX for onnxruntime-web.

Outputs frontend/public/data/arranger_gpt.onnx (+ _int8 variant) and
frontend/public/data/arranger_gpt.json (vocab layout + sampling config).
Verifies ORT-vs-torch logit parity before writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/arranger_gpt.pt")
    ap.add_argument("--out", default="frontend/public/data/arranger_gpt.onnx")
    ap.add_argument("--meta-out", default="frontend/public/data/arranger_gpt.json")
    ap.add_argument("--tokens-per-bar", type=int, default=6)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.train_bar_arranger import BarGPT

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = BarGPT(vocab=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randint(0, cfg["vocab"], (1, cfg["ctx"]), dtype=torch.long)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["tokens"], output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # Parity: ORT vs torch on fixed seed.
    import onnxruntime as ort
    torch.manual_seed(0)
    x = torch.randint(0, cfg["vocab"], (2, cfg["ctx"]), dtype=torch.long)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"tokens": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"parity max|diff|={diff:.2e}", flush=True)
    assert diff < 1e-4, f"ONNX parity failed: {diff}"


    if not args.no_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = out_path.with_name(out_path.stem + "_int8.onnx")
        quantize_dynamic(str(out_path), str(q_path), weight_type=QuantType.QInt8)
        print(f"int8: {q_path.stat().st_size/1024:.0f}KB (fp32 {out_path.stat().st_size/1024:.0f}KB)", flush=True)

    Path(args.meta_out).write_text(json.dumps({
        "version": 1, "vocab": cfg["vocab"], "ctx": cfg["ctx"],
        "tokens_per_bar": args.tokens_per_bar, "model": Path(args.out).name,
        "model_int8": Path(args.out).stem + "_int8.onnx",
    }, separators=(",", ":")))
    print("export ok:", out_path, flush=True)


if __name__ == "__main__":
    main()
