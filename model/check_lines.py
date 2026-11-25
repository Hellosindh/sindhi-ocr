"""Check every line of training_data/sindhi: image problems, bad text, and labels that don't match the image.

Writes model/check_lines.tsv (one row per line: filename, problem, cer, label, read).
problem is "" for a good line, otherwise: missing / unreadable / tiny / blank / duplicate / bad_text / no_image_row.
cer is the CER between the label and a greedy read by crnn_finetune_best (white-on-black crops read flipped,
heh forms counted as one letter); a high cer means the label belongs to another line or the image has no text.
Deletes nothing; deleting is a separate step.

Run:  python -m model.check_lines [folder]   (default: training_data/sindhi; other folders write folder/check.tsv)
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageOps

from model.charset import Charset
from model.crnn import CRNN
from model.decode import greedy
from model.heh_normalize import no_heh
from model.heh_normalize import normalize_heh
from model.layers import edit_distance
from model.order import paragraph_base, to_logical, to_visual
from model.overfit import to_tensor
from model.train_finetune import usable

LIB = ROOT / "assets" / "training_data" / "sindhi"
OUT = ROOT / "model" / "check_lines.tsv"


class Lines(torch.utils.data.Dataset):
    def __init__(self, rows, lib):
        self.rows = rows
        self.lib = lib  # passed in: worker processes re-import this module and would see the default LIB

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, _text = self.rows[index]
        path = self.lib / "lines" / name
        if not path.exists():
            return index, "missing", None, ""
        try:
            data = path.read_bytes()
            with Image.open(path) as photo:
                gray = photo.convert("L")
        except Exception:
            return index, "unreadable", None, ""
        pixels = np.asarray(gray)
        if gray.height < 8 or gray.width < 16:
            return index, "tiny", None, hashlib.md5(data).hexdigest()
        if pixels.std() < 4:
            return index, "blank", None, hashlib.md5(data).hexdigest()
        if np.median(pixels) < 128:  # white text on a dark band: read it flipped
            gray = ImageOps.invert(gray)
        return index, "", to_tensor(gray), hashlib.md5(data).hexdigest()


def main() -> None:
    global LIB, OUT
    if len(sys.argv) > 1:  # check another folder of the same layout (lines/ + transcripts.tsv)
        LIB = Path(sys.argv[1]).resolve()
        OUT = LIB / "check.tsv"
    started = time.time()
    rows = []
    for line in (LIB / "transcripts.tsv").read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.lower().startswith("filename"):
            name, text = line.split("\t", 1)
            rows.append((Path(name.strip()).name, text.strip()))
    charset = Charset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ROOT / "model" / "weights" / "crnn_finetune_best.pt", map_location="cpu", weights_only=False)
    model = CRNN(checkpoint["classes"]).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    loader = torch.utils.data.DataLoader(Lines(rows, LIB), batch_size=None, num_workers=6)
    result = [None] * len(rows)
    first_of_hash = {}
    with torch.no_grad():
        for done, (index, problem, tensor, digest) in enumerate(loader):
            name, text = rows[index]
            if not problem and digest in first_of_hash:
                problem = "duplicate"  # same picture as an earlier line: keep the first only
            first_of_hash.setdefault(digest, index)
            # letter_* crops are single handwritten letters: one character is their whole label
            if not problem and (not usable(text) or (len(text.replace(" ", "")) < 2 and not name.startswith(("letter_", "hletter_", "lig_")))):
                problem = "bad_text"
            cer, read = "", ""
            if not problem:
                text = normalize_heh(text)
                base = paragraph_base(text)
                try:
                    target = no_heh(to_logical(to_visual(text, charset), charset, base_dir=base))
                except AssertionError:  # broken direction marks: training skips these too
                    problem = "bad_text"
            if not problem:
                pred = greedy(model(tensor.unsqueeze(0).to(device))[0].log_softmax(-1), charset)
                read = no_heh(normalize_heh(to_logical(pred, charset, base_dir=base)))
                cer = f"{edit_distance(list(target), list(read)) / max(1, len(target)):.4f}"
            result[index] = (name, problem, cer, rows[index][1], read)
            if done % 10000 == 0:
                print(f"{done}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    names = {r[0] for r in rows}
    orphans = [p.name for p in (LIB / "lines").glob("*.png") if p.name not in names]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write("filename\tproblem\tcer\tlabel\tread\n")
        for r in result:
            handle.write("\t".join(v.replace("\t", " ") for v in r) + "\n")
        for name in orphans:
            handle.write(f"{name}\tno_image_row\t\t\t\n")
    counts = {}
    for r in result:
        counts[r[1] or "ok"] = counts.get(r[1] or "ok", 0) + 1
    print(f"done {len(rows)} lines, {len(orphans)} images without a row, {counts} in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
