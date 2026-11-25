"""Table structure: given a cropped table, find its rows, columns and header rows.

The net (same U-Net as LayoutNet, 3 outputs at 1/4 resolution) paints
    0  row gaps     the blank band between two rows of cells, across the table
    1  column gaps  the blank band between two columns of cells, down the table
    2  header       rows that are table headers
Rows and columns come from projecting the gap maps: each run of gap is one separator line.
Cells are the grid between separators; merged cells are found later from where text crosses a
separator.

Training data:
  PubTables-1M (assets/layout_data/pubtables): grid rebuilt from the OTSL tokens (fcel / ecel / ched /
      srow start a cell, lcel / ucel / xcel continue the cell to the left / above) and the cell boxes.
  Synthetic Sindhi tables (assets/layout_data/sindhi_tables): written by layout.synth_tables.

Run:  python -m layout.tables [--epochs 8] [--batch 8]     -> model/weights/table.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layout.model import STRIDE, LayoutNet, net_input

DATA_ROOT = ROOT / "assets" / "layout_data"
WEIGHTS = ROOT / "model" / "weights" / "table.pt"
LOG = Path(__file__).resolve().parent / "table_log.txt"
SIZE = 640
ORIGIN = {"fcel", "ecel", "ched", "srow", "rhed"}
CONTINUE = {"lcel", "ucel", "xcel"}


class TableNet(LayoutNet):
    """LayoutNet body with a 3-channel head (row gap, column gap, header)."""

    def __init__(self, width: int = 32):
        super().__init__(width)
        c = width * 2
        self.cls = torch.nn.Conv2d(c, 3, 1)
        self.box = None

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        y = self.u4(s5, s4)
        y = self.u3(y, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        return self.cls(y)


def grid_from_otsl(otsl: list[str], cells: list) -> list[dict]:
    """[{r, c, rs, cs, box, header}] for every cell that starts in the grid."""
    rows: list[list[str]] = [[]]
    for tok in otsl:
        if tok == "nl":
            rows.append([])
        else:
            rows[-1].append(tok)
    rows = [r for r in rows if r]
    out = []
    k = 0
    for r, row in enumerate(rows):
        for c, tok in enumerate(row):
            if tok not in ORIGIN:
                continue
            cs = 1
            while c + cs < len(row) and row[c + cs] in ("lcel", "xcel"):
                cs += 1
            rs = 1
            while r + rs < len(rows) and c < len(rows[r + rs]) and rows[r + rs][c] in ("ucel", "xcel"):
                rs += 1
            box = cells[k][:4] if k < len(cells) else None
            out.append({"r": r, "c": c, "rs": rs, "cs": cs, "box": box, "header": tok == "ched"})
            k += 1
    return out


def gaps(cells: list[dict], axis: str) -> list[tuple[float, float]]:
    """Blank bands between consecutive rows (axis 'r') or columns ('c') from single-span cells."""
    lo, hi, span = ("box_y0", "box_y1", "rs") if axis == "r" else ("box_x0", "box_x1", "cs")
    idx = 1 if axis == "r" else 0
    ext: dict[int, list[float]] = {}
    for cell in cells:
        if cell["box"] is None or cell[span] != 1:
            continue
        a, b = cell["box"][idx], cell["box"][idx + 2]
        if b <= a:
            continue
        e = ext.setdefault(cell[axis], [a, b])
        e[0], e[1] = min(e[0], a), max(e[1], b)
    keys = sorted(ext)
    out = []
    for k0, k1 in zip(keys, keys[1:]):
        a, b = ext[k0][1], ext[k1][0]
        if b - a < 2:  # touching rows: a thin band at the midpoint
            m = (a + b) / 2
            a, b = m - 1, m + 1
        out.append((a, b))
    return out


def header_rows(cells: list[dict]) -> tuple[float, float] | None:
    ys = [cell["box"] for cell in cells if cell["header"] and cell["box"]]
    if not ys:
        return None
    return min(b[1] for b in ys), max(b[3] for b in ys)


class Tables(torch.utils.data.Dataset):
    """Table crops with (row gaps, column gaps, header band, table box)."""

    def __init__(self, name: str, split: str, train: bool, limit: int | None = None):
        self.dir = DATA_ROOT / name / split
        self.rows = []
        for line in (self.dir / "labels.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if "otsl" in row:  # PubTables: rebuild the grid
                cells = grid_from_otsl(row["otsl"], row["cells"])
                row = {"image": row["image"], "table": row["table"], "rows": gaps(cells, "r"),
                       "cols": gaps(cells, "c"), "header": header_rows(cells)}
            self.rows.append(row)
            if limit and len(self.rows) >= limit:
                break
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.dir / row["image"]).convert("L")
        x0, y0, x1, y1 = row["table"]
        rng = random.Random() if self.train else random.Random(index)
        pad = rng.randint(2, 20) if self.train else 8
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(image.width, x1 + pad), min(image.height, y1 + pad)
        image = image.crop((cx0, cy0, cx1, cy1))
        rows = [(a - cy0, b - cy0) for a, b in row["rows"]]
        cols = [(a - cx0, b - cx0) for a, b in row["cols"]]
        header = (row["header"][0] - cy0, row["header"][1] - cy0) if row["header"] else None
        if self.train:
            if rng.random() < 0.5:  # right-to-left tables (Sindhi): columns mirror
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                cols = [(image.width - b, image.width - a) for a, b in cols]
            if rng.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
        scale = SIZE / max(image.size)
        w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        page = Image.new("L", (SIZE, SIZE), 255)
        page.paste(image.resize((w, h), Image.Resampling.BILINEAR), (0, 0))
        g = SIZE // STRIDE
        target = np.zeros((3, g, g), np.float32)
        valid = np.zeros((g, g), np.float32)
        gw, gh = int(np.ceil(w / STRIDE)), int(np.ceil(h / STRIDE))
        valid[:gh, :gw] = 1
        for a, b in rows:
            ga, gb = int(a * scale / STRIDE), int(np.ceil(b * scale / STRIDE))
            target[0, max(0, ga) : min(gh, max(ga + 1, gb)), :gw] = 1
        for a, b in cols:
            ga, gb = int(a * scale / STRIDE), int(np.ceil(b * scale / STRIDE))
            target[1, :gh, max(0, ga) : min(gw, max(ga + 1, gb))] = 1
        if header:
            ga, gb = int(header[0] * scale / STRIDE), int(np.ceil(header[1] * scale / STRIDE))
            target[2, max(0, ga) : min(gh, gb), :gw] = 1
        gray = torch.from_numpy(np.asarray(page, np.uint8).copy())[None]
        return gray, torch.from_numpy(target), torch.from_numpy(valid)


def separators(prob: np.ndarray, axis: int, min_run: int = 1, thresh: float = 0.5) -> list[float]:
    """Gap map -> separator positions (cell units): centre of each run of high mean gap."""
    profile = prob.mean(axis=axis)
    on = profile > thresh
    out, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_run:
                out.append((start + i) / 2)
            start = None
    return out


def structure(prob: np.ndarray, w_cells: int, h_cells: int) -> tuple[list[float], list[float], list[float]]:
    """(row separators, column separators, header) in table pixels at net scale (x STRIDE)."""
    p = prob[:, :h_cells, :w_cells]
    rows = [r * STRIDE for r in separators(p[0], 1)]
    cols = [c * STRIDE for c in separators(p[1], 0)]
    header = p[2].mean(axis=1)
    return rows, cols, header


def evaluate(model, device, data: Tables) -> dict:
    """Row / column count accuracy and separator recall (within 2 cells) on a test set."""
    model.eval()
    exact_r = exact_c = hit = total = 0
    with torch.no_grad():
        for i in range(len(data)):
            gray, target, valid = data[i]
            with torch.autocast("cuda", dtype=torch.float16):
                out = torch.sigmoid(model(net_input(gray[None].to(device))).float())[0].cpu().numpy()
            gh = int(valid.sum(1).gt(0).sum())
            gw = int(valid.sum(0).gt(0).sum())
            t = target.numpy()
            pr, pc = separators(out[0, :gh, :gw], 1), separators(out[1, :gh, :gw], 0)
            tr, tc = separators(t[0, :gh, :gw], 1), separators(t[1, :gh, :gw], 0)
            exact_r += len(pr) == len(tr)
            exact_c += len(pc) == len(tc)
            for truth, pred in ((tr, pr), (tc, pc)):
                total += len(truth)
                hit += sum(any(abs(a - b) <= 2 for b in pred) for a in truth)
    model.train()
    n = max(1, len(data))
    return {"rows_exact": exact_r / n, "cols_exact": exact_c / n, "sep_recall": hit / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sets", default="pubtables,sindhi_tables")
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args()
    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda")
    names = [n for n in args.sets.split(",") if (DATA_ROOT / n / "train" / "labels.jsonl").exists()]
    parts = [Tables(n, "train", True) for n in names]
    tests = {n: Tables(n, "test", False, args.test) for n in names}
    data = torch.utils.data.ConcatDataset(parts)
    weights = torch.cat([torch.full((len(p),), 1.0 / len(p)) for p in parts])
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=min(len(data), 30000), replacement=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True)
    model = TableNet().to(device)
    layout = ROOT / "model" / "weights" / "layout.pt"
    if layout.exists():  # start from the layout model's body: it already knows tables
        state = torch.load(layout, map_location="cpu", weights_only=False)["state"]
        own = model.state_dict()
        own.update({k: v for k, v in state.items() if k in own and own[k].shape == v.shape})
        model.load_state_dict(own)
    say(f"sets={names} train={[len(p) for p in parts]} per epoch={len(sampler)} batch={args.batch}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    pos_weight = torch.tensor([3.0, 3.0, 2.0], device=device).view(1, 3, 1, 1)
    best = -1.0
    started = time.time()
    for epoch in range(args.epochs):
        for step, (gray, target, valid) in enumerate(loader):
            gray, target, valid = gray.to(device), target.to(device), valid.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(net_input(gray))
            loss = (F.binary_cross_entropy_with_logits(out.float(), target, pos_weight=pos_weight, reduction="none")
                    * valid[:, None]).sum() / valid.sum().clamp(min=1) / 3
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 300 == 0:
                say(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} {time.time() - started:.0f}s")
        scores = {n: evaluate(model, device, t) for n, t in tests.items()}
        mean = float(np.mean([s["sep_recall"] * 0.5 + (s["rows_exact"] + s["cols_exact"]) * 0.25 for s in scores.values()]))
        say(f"epoch {epoch} " + " | ".join(f"{n}: rows exact {s['rows_exact']:.2f} cols exact {s['cols_exact']:.2f} separators found {s['sep_recall']:.2f}" for n, s in scores.items()) + f" | score {mean:.3f}")
        if mean > best:
            best = mean
            torch.save({"state": model.state_dict(), "score": mean, "scores": scores, "epoch": epoch}, WEIGHTS)
            say(f"saved {WEIGHTS.name} score={mean:.3f}")
    say(f"done best score={best:.3f} in {(time.time() - started) / 60:.0f} min")


def say(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
