"""Export the reader and layout models to ONNX for mobile / on-device apps:

  crnn.onnx    line reader, fp32 (int8 read worse on the phone), input "image" (B,1,32,W) ink-high
               0..1, output "log_probs" (B,T,C)
  layout.onnx  layout regions, int8 (dynamic), input "gray" (1,1,640,640) 0..255 page letterboxed at
               top-left, output "prob" (1,13,160,160): softmax over background + 12 classes (the box
               edges are not needed on the phone: each line is tagged with the region under it)

Checks each export against PyTorch before writing it. Writes to --out (default: export/android).

Run:  python -m model.export_android [--out DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch

from layout.model import CLASSES, N_CLASSES, SIZE, LayoutNet, net_input
from model.crnn import CRNN

APP = Path(__file__).resolve().parents[1] / "export" / "android"  # default output folder; pass --out to change


class Reader(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        return self.net(image).log_softmax(-1)


class Layout(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, gray):
        return self.net(net_input(gray))[:, : self.net.n_classes + 1].softmax(1)


def export_reader(out: Path) -> None:
    ck = torch.load(ROOT / "model" / "weights" / "crnn_finetune_best.pt", map_location="cpu", weights_only=False)
    net = CRNN(ck["classes"]).eval()
    net.load_state_dict(ck["state"])
    model = Reader(net).eval()
    x = torch.rand(2, 1, 32, 600)
    path = out / "crnn.onnx"
    torch.onnx.export(model, (x,), str(path), input_names=["image"], output_names=["log_probs"], opset_version=17,
                      dynamic_axes={"image": {0: "batch", 3: "width"}, "log_probs": {0: "batch", 1: "time"}}, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for w in (120, 600, 1400):
        x = torch.rand(1, 1, 32, w)
        a = model(x).detach().numpy()
        b = sess.run(None, {"image": x.numpy()})[0]
        print(f"crnn width {w}: max diff {np.abs(a - b).max():.2e}, same argmax {np.mean(a.argmax(-1) == b.argmax(-1)):.4f}")
    print(f"crnn.onnx {path.stat().st_size / 1e6:.1f} MB (classes {ck['classes']}, select_cer {ck.get('select_cer')})")


def export_layout(out: Path, pages: list[Path]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from PIL import Image

    from layout.demo import square

    ck = torch.load(ROOT / "model" / "weights" / "layout.pt", map_location="cpu", weights_only=False)
    net = LayoutNet().eval()
    net.load_state_dict(ck["state"])
    model = Layout(net).eval()
    fp32 = out / "layout_fp32.onnx"
    torch.onnx.export(model, (torch.zeros(1, 1, SIZE, SIZE),), str(fp32), input_names=["gray"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    path = out / "layout.onnx"
    quantize_dynamic(str(fp32), str(path), weight_type=QuantType.QUInt8)
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    agree, total = 0, 0
    for p in pages:
        with Image.open(p) as im:
            x, _s, _w, _h = square(im.convert("L"))
        x = x.float()
        ref = model(x).detach().numpy()
        a = s32.run(None, {"gray": x.numpy()})[0]
        b = s8.run(None, {"gray": x.numpy()})[0]
        ink = ref[0, 0] < 0.5  # cells that are some region
        agree += int(((ref.argmax(1) == b.argmax(1))[0] & ink).sum())
        total += int(ink.sum())
        print(f"{p.name}: fp32 diff {np.abs(ref - a).max():.1e} | int8 region cells same class {np.mean((ref.argmax(1) == b.argmax(1))[0][ink]):.3f}")
    print(f"layout int8 agreement {agree / max(1, total):.3f} | layout.onnx {path.stat().st_size / 1e6:.1f} MB (fp32 {fp32.stat().st_size / 1e6:.1f} MB)")
    fp32.unlink()


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else APP
    out.mkdir(parents=True, exist_ok=True)
    export_reader(out)
    pages = sorted((ROOT / "layout" / "real_test").glob("*.png"))[::4]
    export_layout(out, pages)
    print("classes:", ["background"] + CLASSES)


if __name__ == "__main__":
    main()
