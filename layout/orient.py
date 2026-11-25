"""Straighten a page before layout: tilt, sideways (90 / 270) and upside down (180).

Tilt and sideways: the angle where the ink's row projection is sharpest (thin rows of text with
empty gaps), searched on both axes.
Upside down: in Sindhi / Arabic script most ink sits on the baseline at the bottom of a line, with
tall letters (alif, lam, kaf) rising above it. So in an upright line the densest ink row is below
the middle of the line; turned over, it is above.

    upright, info = straighten(gray)      # info = {"turns": k, "degrees": d, "flipped": bool}
"""

from __future__ import annotations

import cv2
import numpy as np

SIDE = 1100  # search on a copy this size
MAX_SKEW = 20.0


def _ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15) > 0


def _text_points(mask: np.ndarray) -> np.ndarray:
    """Centres-free ink pixels of text-sized blobs (no rules, frames or pictures)."""
    n, labels, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((0, 2), np.float32)
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    area = stats[1:, cv2.CC_STAT_AREA]
    typical = np.median(h[area > 8]) if (area > 8).any() else 10
    keep = np.zeros(n, bool)
    keep[1:] = (area > 4) & (h < typical * 6) & (w < mask.shape[1] * 0.3) & (h < mask.shape[0] * 0.3)
    ys, xs = np.nonzero(keep[labels])
    return np.stack([xs, ys], 1).astype(np.float32)


def _sharpness(values: np.ndarray) -> float:
    if values.size < 50:
        return 0.0
    lo, hi = np.percentile(values, [1, 99])
    hist, _e = np.histogram(values, bins=max(8, int(hi - lo) + 1), range=(lo, hi + 1))
    hist = hist.astype(np.float64)
    return float(np.sum(np.diff(hist) ** 2)) / max(1.0, hist.sum()) ** 2 * len(hist)


def find_angle(gray: np.ndarray) -> list[tuple[int, float]]:
    """[(quarter turns for np.rot90, skew degrees)] that make the text lines horizontal, one per
    axis, likeliest first (with few lines the profile can prefer the wrong axis)."""
    scale = SIDE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    pts = _text_points(_ink(small))
    if len(pts) < 200:
        return [(0, 0.0)]
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    xs, ys = pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean()

    def score(deg: float, sideways: bool) -> float:
        a = np.deg2rad(deg)
        v = xs * np.cos(a) + ys * np.sin(a) if sideways else ys * np.cos(a) - xs * np.sin(a)
        return _sharpness(v)

    found = []
    for sideways in (False, True):
        coarse = max(np.arange(-MAX_SKEW, MAX_SKEW + 0.01, 1.0), key=lambda d: score(float(d), sideways))
        fine = max(np.arange(coarse - 1, coarse + 1.01, 0.1), key=lambda d: score(float(d), sideways))
        found.append((score(float(fine), sideways), 3 if sideways else 0, round(float(fine), 1)))
    found.sort(reverse=True)
    return [(t, d) for _s, t, d in found]


def rotate(gray: np.ndarray, turns: int, degrees: float) -> np.ndarray:
    if turns:
        gray = np.ascontiguousarray(np.rot90(gray, turns))
    if abs(degrees) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


def baseline_side(gray: np.ndarray) -> float:
    """> 0 when lines look upright (densest ink row below each line's middle), < 0 upside down."""
    scale = SIDE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    mask = _ink(small)
    keep = np.zeros_like(mask)
    pts = _text_points(mask).astype(int)
    keep[pts[:, 1], pts[:, 0]] = True
    # lines are rows of the page's row profile, but columns may sit side by side: use vertical strips
    votes, weight = 0.0, 0.0
    strips = max(1, keep.shape[1] // 250)
    for strip in np.array_split(keep, strips, axis=1):
        prof = strip.sum(1).astype(np.float32)
        if prof.max() <= 0:
            continue
        on = prof > prof.max() * 0.08
        start = None
        for i, v in enumerate(list(on) + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                a, b = start, i
                start = None
                if b - a < 5:
                    continue
                band = prof[a:b]
                peak = a + float(np.argmax(np.convolve(band, np.ones(3) / 3, mode="same")))
                mid = (a + b - 1) / 2
                ink = float(band.sum())
                votes += (peak - mid) / (b - a) * ink
                weight += ink
    return votes / weight if weight else 0.0


_reader = None


def _load_reader():
    global _reader
    if _reader is None:
        import sys
        from pathlib import Path

        import torch

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.crnn import CRNN

        weights = root / "model" / "weights" / "crnn_finetune_best.pt"
        if not weights.exists():  # no fine-tune yet (being retrained): the v2 base reads well enough here
            weights = root / "model" / "weights" / "crnn_best_v2.pt"
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = CRNN(ck["classes"]).to(device).eval()
        net.load_state_dict(ck["state"])
        _reader = (net, device)
    return _reader


def line_crops(gray: np.ndarray, count: int = 5) -> list[np.ndarray]:
    """A few of the inkiest text lines of an upright page (row bands of its inkiest vertical strip)."""
    scale = SIDE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    mask = _ink(small)
    keep = np.zeros_like(mask)
    pts = _text_points(mask).astype(int)
    keep[pts[:, 1], pts[:, 0]] = True
    strips = np.array_split(np.arange(keep.shape[1]), max(1, keep.shape[1] // 350))
    cols = max(strips, key=lambda c: keep[:, c].sum())
    prof = keep[:, cols].sum(1).astype(np.float32)
    if prof.max() <= 0:
        return []
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")  # dots and thin joins must not split a line
    on = prof > prof.max() * 0.05
    bands, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 4:
                bands.append((start, i, float(prof[start:i].sum())))
            start = None
    if not bands:
        return []
    med = np.median([b - a for a, b, _ in bands])
    bands = [b for b in bands if med * 0.5 <= b[1] - b[0] <= med * 2.2]
    bands = sorted(bands, key=lambda b: -b[2])[:count]
    out = []
    for a, b, _ in bands:
        pad = (b - a) * 0.25
        y0, y1 = int(max(0, (a - pad) / scale if scale < 1 else a - pad)), int(min(gray.shape[0], (b + pad) / scale if scale < 1 else b + pad))
        x0, x1 = int(cols[0] / scale if scale < 1 else cols[0]), int((cols[-1] + 1) / scale if scale < 1 else cols[-1] + 1)
        out.append(gray[y0:y1, x0:x1])
    return out


def read_confidence(gray: np.ndarray) -> float:
    """Mean best-label log-prob of the reader over a few lines (higher = reads like real text)."""
    import torch
    from PIL import Image

    from model.overfit import to_tensor

    net, device = _load_reader()
    tensors = [to_tensor(Image.fromarray(c)) for c in line_crops(gray) if c.size]
    if not tensors:
        return -10.0
    width = max(t.shape[-1] for t in tensors)
    batch = torch.zeros(len(tensors), 1, tensors[0].shape[1], width)
    for i, t in enumerate(tensors):
        batch[i, :, :, : t.shape[-1]] = t
    with torch.no_grad():
        lp = net(batch.to(device)).log_softmax(-1).cpu()
    scores = []
    for i, t in enumerate(tensors):
        frames = max(1, lp.shape[1] * t.shape[-1] // width)  # ignore the padding's frames
        best, labels = lp[i, :frames].max(-1)
        ink = labels != 0
        if ink.any():
            scores.append(float(best[ink].mean()))
    return float(np.mean(scores)) if scores else -10.0


SURE = -0.2  # mean log-prob: reads like real text -> stop trying other orientations
UNSURE = -0.35  # best of all candidates below this: too little text to judge, leave the page as it is


def straighten(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """Try each axis the profile offers, upright and turned over; the reader picks. Small bonuses
    for the profile's favourite axis and for not turning the page over (most photos are upright).
    Stops early when the first try already reads well."""
    best = None
    for rank, (turns, degrees) in enumerate(find_angle(gray)):
        upright = rotate(gray, turns, degrees)
        for flipped in (False, True):
            page = np.ascontiguousarray(upright[::-1, ::-1]) if flipped else upright
            conf = read_confidence(page)
            score = conf + (0.05 if rank == 0 else 0) + (0.1 if not flipped else 0)
            if best is None or score > best[0]:
                best = (score, page, {"turns": turns, "degrees": degrees, "flipped": flipped, "conf": round(conf, 3)})
            if rank == 0 and not flipped and conf > SURE:
                return best[1], best[2]
    if best[2]["conf"] < UNSURE:
        return gray, {"turns": 0, "degrees": 0.0, "flipped": False, "conf": best[2]["conf"], "unsure": True}
    return best[1], best[2]
