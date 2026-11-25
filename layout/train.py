"""Train LayoutNet on DocLayNet pages and synthetic Sindhi pages (assets/layout_data/<set>), evaluate box
AP@0.5 per class on each set's test split, keep the best weights (mean of the sets' mAP) in
model/weights/layout.pt.

Pages are letterboxed to 640 x 640. Training pages are also flipped left-right half the time, so the
model sees right-to-left layouts like Sindhi books (column order and alignment mirror).

Run:  python -m layout.train [--epochs 15] [--batch 6] [--sets doclaynet,sindhi_synth] [--init model/weights/layout.pt]
                             [--out model/weights/layout_news.pt]
--init starts from earlier weights (a grown class count keeps the old classes' head rows).
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

from layout.model import CLASSES, N_CLASSES, SIZE, STRIDE, LayoutNet, decode, net_input, split_lines, tidy

DATA_ROOT = ROOT / "assets" / "layout_data"
WEIGHTS = ROOT / "model" / "weights" / "layout.pt"
LOG = Path(__file__).resolve().parent / "train_log.txt"
CORE = 0.6  # central share of a region painted as its core


def say(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def letterbox(image: Image.Image, boxes: list, size: int = SIZE):
    scale = size / max(image.size)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (0, 0))
    return canvas, [[x0 * scale, y0 * scale, x1 * scale, y1 * scale, c] for x0, y0, x1, y1, c in boxes], scale


def targets(boxes: list, size: int = SIZE):
    g = size // STRIDE
    cls = np.zeros((g, g), np.int64)
    edge = np.zeros((4, g, g), np.float32)
    # big regions first, so small ones (footers, captions) painted later stay visible
    for x0, y0, x1, y1, c in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * CORE / 2, (y1 - y0) * CORE / 2
        gx0, gx1 = int(max(0, (cx - hw) // STRIDE)), int(min(g - 1, (cx + hw) // STRIDE))
        gy0, gy1 = int(max(0, (cy - hh) // STRIDE)), int(min(g - 1, (cy + hh) // STRIDE))
        cls[gy0 : gy1 + 1, gx0 : gx1 + 1] = c + 1
        px = (np.arange(gx0, gx1 + 1) + 0.5) * STRIDE
        py = (np.arange(gy0, gy1 + 1) + 0.5) * STRIDE
        edge[0, gy0 : gy1 + 1, gx0 : gx1 + 1] = (px - x0)[None, :] / size
        edge[1, gy0 : gy1 + 1, gx0 : gx1 + 1] = (py - y0)[:, None] / size
        edge[2, gy0 : gy1 + 1, gx0 : gx1 + 1] = (x1 - px)[None, :] / size
        edge[3, gy0 : gy1 + 1, gx0 : gx1 + 1] = (y1 - py)[:, None] / size
    return cls, np.clip(edge, 0, None)


class Pages(torch.utils.data.Dataset):
    def __init__(self, split: str, train: bool, name: str = "doclaynet"):
        self.dir = DATA_ROOT / name / split
        self.rows = [json.loads(line) for line in (self.dir / "labels.jsonl").open(encoding="utf-8")]
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.dir / row["image"]).convert("L")
        boxes = [list(b) for b in row["boxes"]]
        rng = random.Random() if self.train else random.Random(index)
        if self.train:
            if rng.random() < 0.5:  # right-to-left layouts
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes = [[image.width - x1, y0, image.width - x0, y1, c] for x0, y0, x1, y1, c in boxes]
            if rng.random() < 0.5:  # zoomed / cropped photo of the page
                s = rng.uniform(0.75, 1.0)
                cw, ch = int(image.width * s), int(image.height * s)
                ox, oy = rng.randint(0, image.width - cw), rng.randint(0, image.height - ch)
                image = image.crop((ox, oy, ox + cw, oy + ch))
                kept = []
                for x0, y0, x1, y1, c in boxes:
                    nx0, ny0, nx1, ny1 = max(x0 - ox, 0), max(y0 - oy, 0), min(x1 - ox, cw), min(y1 - oy, ch)
                    if nx1 - nx0 > 0.6 * (x1 - x0) and ny1 - ny0 > 0.6 * (y1 - y0):
                        kept.append([nx0, ny0, nx1, ny1, c])
                boxes = kept
            if rng.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
            if rng.random() < 0.3:
                lo, hi = rng.uniform(0, 60), rng.uniform(190, 255)
                image = image.point(lambda v: int(lo + v * (hi - lo) / 255))
        page, boxes, _scale = letterbox(image, boxes)
        cls, edge = targets(boxes)
        gray = torch.from_numpy(np.asarray(page, np.uint8).copy())[None]
        return gray, torch.from_numpy(cls), torch.from_numpy(edge)


def loss_fn(out, cls, edge, weight):
    ce = F.cross_entropy(out[:, : N_CLASSES + 1].float(), cls, weight=weight)
    fg = (cls > 0).unsqueeze(1).float()
    pred = out[:, N_CLASSES + 1 :].float()
    l1 = (torch.abs(torch.log1p(pred * 20) - torch.log1p(edge * 20)) * fg).sum() / fg.sum().clamp(min=1) / 4
    return ce + l1, ce.item(), l1.item()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(model, device, limit: int | None = None, name: str = "doclaynet") -> tuple[float, dict]:
    data = Pages("test", train=False, name=name)
    rows = data.rows[:limit] if limit else data.rows
    preds = {k: [] for k in range(N_CLASSES)}
    n_true = {k: 0 for k in range(N_CLASSES)}
    model.eval()
    with torch.no_grad():
        for i, row in enumerate(rows):
            gray, _cls, _edge = data[i]
            image = Image.open(data.dir / row["image"])
            _page, truth, _s = letterbox(image.convert("L"), row["boxes"])
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(net_input(gray[None].to(device)))
            found = tidy(split_lines(decode(out)[0], gray[0].numpy()))
            for k in range(N_CLASSES):
                t = [b for b in truth if b[4] == k]
                n_true[k] += len(t)
                used = set()
                for box in sorted([b for b in found if b[4] == k], key=lambda b: -b[5]):
                    best, best_j = 0.0, -1
                    for j, tb in enumerate(t):
                        if j not in used:
                            v = iou(box, tb)
                            if v > best:
                                best, best_j = v, j
                    hit = best >= 0.5
                    if hit:
                        used.add(best_j)
                    preds[k].append((box[5], hit))
    aps = {}
    for k in range(N_CLASSES):
        if n_true[k] == 0:
            continue
        p = sorted(preds[k], key=lambda x: -x[0])
        tp = np.cumsum([h for _s, h in p]) if p else np.array([0])
        fp = np.cumsum([not h for _s, h in p]) if p else np.array([0])
        recall = tp / n_true[k]
        precision = tp / np.maximum(tp + fp, 1)
        ap = 0.0
        for r in np.linspace(0, 1, 11):  # 11-point interpolated AP
            ap += (precision[recall >= r].max() if (recall >= r).any() else 0.0) / 11
        aps[CLASSES[k]] = ap
    model.train()
    return float(np.mean(list(aps.values()))), aps


def load_init(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    own = model.state_dict()
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
        elif key in own and own[key].shape[1:] == value.shape[1:] and own[key].shape[0] > value.shape[0]:
            own[key][: value.shape[0]] = value  # class head grew: keep the old classes' rows
    model.load_state_dict(own)


def score(model, device, sets, limit) -> tuple[float, dict]:
    parts = {name: evaluate(model, device, limit, name) for name in sets}
    return float(np.mean([m for m, _a in parts.values()])), parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-pages", type=int, default=600)
    parser.add_argument("--sets", default="doclaynet")
    parser.add_argument("--init", default="")
    parser.add_argument("--out", default="", help="save the best weights here instead of model/weights/layout.pt")
    args = parser.parse_args()
    global WEIGHTS
    if args.out:
        WEIGHTS = ROOT / args.out
    sets = [s for s in args.sets.split(",") if s]
    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda")
    parts = [Pages("train", train=True, name=name) for name in sets]
    data = torch.utils.data.ConcatDataset(parts)
    # every set gets an equal share of each epoch, whatever its size
    weights = torch.cat([torch.full((len(p),), 1.0 / len(p)) for p in parts])
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(data), replacement=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True)
    model = LayoutNet().to(device)
    if args.init:
        load_init(model, ROOT / args.init)
    say(f"sets={sets} train pages={[len(p) for p in parts]} classes={N_CLASSES} params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M batch={args.batch} init={args.init or '-'}")
    weight = torch.ones(N_CLASSES + 1, device=device)
    weight[0] = 0.1  # background is most pixels
    # rare classes lose to Text otherwise (the English-only run never found a footnote or title)
    for name in ("Caption", "Footnote", "List-item", "Page-footer", "Page-header", "Title", "Contents-entry",
                 "Masthead", "Jump-line"):
        weight[CLASSES.index(name) + 1] = 2.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    best = -1.0
    if args.init:
        best, start_parts = score(model, device, sets, args.eval_pages)
        say("start " + " | ".join(f"{n} mAP@0.5 {m:.3f}" for n, (m, _a) in start_parts.items()) + f" | mean {best:.3f}")
    started = time.time()
    for epoch in range(args.epochs):
        total = 0.0
        for step, (gray, cls, edge) in enumerate(loader):
            gray, cls, edge = gray.to(device), cls.to(device), edge.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(net_input(gray))
            loss, ce, l1 = loss_fn(out, cls, edge, weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 200 == 0:
                say(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f} (ce {ce:.3f} box {l1:.3f}) {time.time() - started:.0f}s")
        mean, parts_score = score(model, device, sets, args.eval_pages)
        for name, (m, aps) in parts_score.items():
            say(f"epoch {epoch} {name} mAP@0.5 {m:.3f} | " + " ".join(f"{k}={v:.2f}" for k, v in aps.items()))
        say(f"epoch {epoch} mean loss {total / len(loader):.3f} | mean mAP@0.5 {mean:.3f}")
        if mean > best:
            best = mean
            WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "map50": mean, "sets": {n: m for n, (m, _a) in parts_score.items()},
                        "aps": {n: a for n, (_m, a) in parts_score.items()}, "epoch": epoch, "classes": CLASSES}, WEIGHTS)
            say(f"saved {WEIGHTS.name} mean map50={mean:.3f}")
    say(f"done best mean map50={best:.3f} in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
