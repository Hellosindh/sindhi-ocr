"""Train LineNet on the synthetic pages from linedet.synth_pages.

Pages are scaled so the long side lands between 1100 and 2300 px (the detector runs
at ~1600), then random 512x512 tiles are cut; each tile keeps its page coordinates.

Run:  python -m linedet.train --steps 30000
Out:  model/weights/linedet.pt  (best on held-out pages)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linedet.model import N_TYPES, LineNet, page_input  # noqa: E402

PAGES = ROOT / "assets" / "linedet_pages"
WEIGHTS = ROOT / "model" / "weights" / "linedet.pt"
TILE = 512
HELD_OUT = 150
TYPE_WEIGHT = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0])


def read_gray(path: Path) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)


def core_quad(quad: np.ndarray) -> tuple[np.ndarray, float]:
    """Thin band through the middle of a line box (TL,TR,BR,BL) and the line height."""
    tl, tr, br, bl = quad
    height = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    length = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    top_l, top_r = tl + (bl - tl) * 0.32, tr + (br - tr) * 0.32
    bot_l, bot_r = tl + (bl - tl) * 0.68, tr + (br - tr) * 0.68
    shrink = min(0.25 * height, max(0.0, length / 2 - 1)) / max(length, 1e-3)
    direction_l = (top_r - top_l) * shrink
    direction_b = (bot_r - bot_l) * shrink
    return np.array([top_l + direction_l, top_r - direction_l, bot_r - direction_b, bot_l + direction_b]), height


def targets(lines: list, h: int, w: int, scale: float, ox: int, oy: int):
    """Core mask, log-height map and type map at half resolution for one tile."""
    hh, ww = h // 2, w // 2
    core = np.zeros((hh, ww), np.uint8)
    logh = np.zeros((hh, ww), np.float32)
    kind = np.full((hh, ww), -1, np.int64)
    for quad, line_type in lines:
        q = (np.array(quad, np.float32) * scale - [ox, oy])
        band, height = core_quad(q)
        if height < 4:
            continue
        pts = np.round(band / 2 * 8).astype(np.int32)  # 3 fractional bits for a smooth thin band
        mask = np.zeros((hh, ww), np.uint8)
        cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=3)
        if not mask.any():
            # a very thin band can vanish at half resolution: keep at least its centre line
            mid = ((band[0] + band[3]) / 2 / 2, (band[1] + band[2]) / 2 / 2)
            cv2.line(mask, tuple(np.int32(mid[0])), tuple(np.int32(mid[1])), 1, 1)
        on = mask > 0
        core[on] = 1
        logh[on] = math.log(height)
        kind[on] = line_type
    return core, logh, kind


class PageTiles(Dataset):
    def __init__(self, names: list[str], tiles_per_page: int, train: bool):
        self.names = names
        self.tiles = tiles_per_page
        self.train = train

    def __len__(self):
        return len(self.names) * self.tiles

    def __getitem__(self, index):
        rng = random.Random(None if self.train else index)
        name = self.names[index // self.tiles]
        gray = read_gray(PAGES / f"{name}.jpg")
        lines = json.loads((PAGES / f"{name}.json").read_text(encoding="utf-8"))["lines"]
        h, w = gray.shape
        scale = rng.uniform(1100, 2300) / max(h, w)
        sw, sh = max(TILE, round(w * scale)), max(TILE, round(h * scale))
        page = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if page.shape != (sh, sw):
            page = cv2.copyMakeBorder(page, 0, sh - page.shape[0], 0, sw - page.shape[1], cv2.BORDER_REPLICATE)
        if self.train and rng.random() < 0.15:
            page = 255 - page if rng.random() < 0.15 else cv2.equalizeHist(page)
        ox = rng.randrange(0, sw - TILE + 1)
        oy = rng.randrange(0, sh - TILE + 1)
        if rng.random() < 0.5 and lines:
            # half the tiles are centred on a random non-body line so headers/footers get seen
            special = [q for q, k in lines if k != 0] or [q for q, _k in lines]
            cx, cy = np.mean(np.array(rng.choice(special)), 0) * scale
            ox = int(np.clip(cx - rng.uniform(0.2, 0.8) * TILE, 0, sw - TILE))
            oy = int(np.clip(cy - rng.uniform(0.2, 0.8) * TILE, 0, sh - TILE))
        tile = page[oy : oy + TILE, ox : ox + TILE]
        core, logh, kind = targets(lines, TILE, TILE, scale, ox, oy)
        return (
            torch.from_numpy(np.ascontiguousarray(tile))[None],
            torch.tensor([ox, oy, round(w * scale), round(h * scale)], dtype=torch.float32),
            torch.from_numpy(core),
            torch.from_numpy(logh),
            torch.from_numpy(kind),
        )


def net_input(tile, where, device):
    tile = tile.to(device, non_blocking=True)
    parts = [page_input(tile[i : i + 1], *where[i].tolist()) for i in range(tile.shape[0])]
    return torch.cat(parts, 0)


def losses(out, core, logh, kind):
    prob_logit = out[:, 0]
    core_f = core.float()
    bce = F.binary_cross_entropy_with_logits(prob_logit, core_f, pos_weight=torch.tensor(3.0, device=out.device))
    prob = torch.sigmoid(prob_logit)
    inter = (prob * core_f).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + core_f.sum() + 1)
    on = core > 0
    if on.any():
        height = F.smooth_l1_loss(out[:, 1][on], logh[on], beta=0.1)
        types = F.cross_entropy(out[:, 2 : 2 + N_TYPES].permute(0, 2, 3, 1)[on], kind[on], weight=TYPE_WEIGHT.to(out.device))
    else:
        height = types = out.sum() * 0
    return bce + dice + height + 0.5 * types, (bce.item(), dice.item(), height.item(), types.item())


def evaluate(net, loader, device) -> float:
    net.eval()
    total, count, type_ok, type_n = 0.0, 0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for tile, where, core, logh, kind in loader:
            core, logh, kind = core.to(device), logh.to(device), kind.to(device)
            out = net(net_input(tile, where, device)).float()
            loss, _parts = losses(out, core, logh, kind)
            total += loss.item()
            count += 1
            on = core > 0
            type_ok += (out[:, 2:].argmax(1)[on] == kind[on]).sum().item()
            type_n += on.sum().item()
    net.train()
    print(f"  val loss={total / max(1, count):.4f} type_acc={type_ok / max(1, type_n):.3f}", flush=True)
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = "cuda"
    names = sorted(p.stem for p in PAGES.glob("*.json"))
    train_names, val_names = names[:-HELD_OUT], names[-HELD_OUT:]
    print(f"pages train={len(train_names)} val={len(val_names)}", flush=True)
    train = DataLoader(PageTiles(train_names, 4, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    val = DataLoader(PageTiles(val_names, 2, False), batch_size=args.batch, num_workers=2)
    net = LineNet().to(device)
    if args.resume and WEIGHTS.exists():
        net.load_state_dict(torch.load(WEIGHTS, map_location=device)["model"])
    print(f"params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    best = float("inf")
    step = 0
    started = time.time()
    running = np.zeros(4)
    while step < args.steps:
        for tile, where, core, logh, kind in train:
            core, logh, kind = core.to(device), logh.to(device), kind.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(net_input(tile, where, device))
            loss, parts = losses(out.float(), core, logh, kind)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running = running * 0.98 + np.array(parts) * 0.02
            step += 1
            if step % 200 == 0:
                print(f"step {step} bce={running[0]:.3f} dice={running[1]:.3f} h={running[2]:.3f} type={running[3]:.3f} "
                      f"{(time.time() - started) / step:.3f}s/step", flush=True)
            if step % 2000 == 0 or step == args.steps:
                score = evaluate(net, val, device)
                if score < best:
                    best = score
                    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": net.state_dict(), "step": step, "val": score}, WEIGHTS)
                    print(f"  saved {WEIGHTS.name}", flush=True)
            if step >= args.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
