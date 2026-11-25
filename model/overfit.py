"""Memorize a few hundred short lines. This shows whether CTC can learn at all."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_synth import draw_line, eligible, load_font_sets, visual_from_image_order, word_ok
from model.charset import Charset
from model.crnn import CRNN
from model.decode import greedy
from model.order import to_logical, to_visual

HEIGHT = 32
COUNT = 300
EPOCHS = 80
MAX_CHARS = 20


def short_lines(
    charset: Charset,
    char_fonts: dict[str, set[str]],
    count: int,
    max_chars: int = MAX_CHARS,
    word_limit: int = 20000,
) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    words = []
    corpus = ROOT / "sindhi_corpus.txt"
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            for word in line.split():
                if word_ok(word, char_fonts):
                    words.append(word)
            if len(words) > word_limit:
                break
    index = 0
    while index < len(words) and len(chosen) < count:
        pieces = []
        while index < len(words) and len(" ".join(pieces + [words[index]])) <= max_chars:
            pieces.append(words[index])
            index += 1
        if not pieces:
            index += 1
            continue
        text = " ".join(pieces)
        visual = to_visual(text, charset)
        if not charset.encode(visual) or len(visual) > max_chars:
            continue
        fonts = eligible(text, char_fonts)
        if not fonts:
            continue
        chosen.append((text, sorted(fonts)[0]))
    return chosen


class MemorySet(Dataset):
    def __init__(self, rows: list[tuple[torch.Tensor, torch.Tensor, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def crop_to_ink(image: Image.Image, pad: int = 4) -> Image.Image:
    """Cut the page down to the ink, then add the same pad the synthetic lines use."""
    gray = image.convert("L")
    width, height = gray.size
    raw = gray.tobytes()
    row_hits = [0] * height
    col_hits = [0] * width
    for y in range(height):
        start = y * width
        for x in range(width):
            if raw[start + x] < 200:
                row_hits[y] += 1
                col_hits[x] += 1
    rows = [y for y, hits in enumerate(row_hits) if hits >= 2]
    cols = [x for x, hits in enumerate(col_hits) if hits >= 2]
    if not rows or not cols:
        return gray
    tight = gray.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    canvas = Image.new("L", (tight.width + pad * 2, tight.height + pad * 2), 255)
    canvas.paste(tight, (pad, pad))
    return canvas


def to_tensor(image: Image.Image) -> torch.Tensor:
    gray = crop_to_ink(image)
    width, height = gray.size
    new_width = max(8, round(width * HEIGHT / max(1, height)))
    new_width = max(8, (new_width + 3) // 4 * 4)
    gray = gray.resize((new_width, HEIGHT), Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).float()
    return 1.0 - pixels.view(1, HEIGHT, new_width) / 255.0


def collate(batch):
    widths = [item[0].shape[-1] for item in batch]
    max_width = max(widths)
    images = torch.zeros(len(batch), 1, HEIGHT, max_width)
    targets = []
    texts = []
    input_lengths = []
    for image, target, _text, _visual in batch:
        images[len(targets), :, :, : image.shape[-1]] = image
        targets.append(target)
        texts.append(_text)
        input_lengths.append(CRNN.time_steps(image.shape[-1]))
    return (
        images[: len(targets)],
        torch.cat(targets),
        torch.tensor(input_lengths),
        torch.tensor([len(item) for item in targets]),
        texts,
    )


def say(message: str) -> None:
    log = Path(__file__).resolve().parent / "overfit_log.txt"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message.encode("ascii", "backslashreplace").decode(), flush=True)


def main() -> None:
    log = Path(__file__).resolve().parent / "overfit_log.txt"
    log.write_text("", encoding="utf-8")
    charset = Charset()
    pairs = short_lines(charset, load_font_sets(), COUNT)
    rows = []
    for text, font_name in pairs:
        image, missing = draw_line(text, font_name)
        if image is None or missing:
            continue
        tensor = to_tensor(image)
        visual = to_visual(text, charset)
        encoded = charset.encode(visual)
        if CRNN.time_steps(tensor.shape[-1]) < len(encoded):
            continue
        rows.append((tensor, torch.tensor(encoded), text, visual))
        if len(rows) >= COUNT:
            break
    if len(rows) < 50:
        raise SystemExit(f"only built {len(rows)} short lines")
    say(f"lines={len(rows)} chars={sum(len(text) for _tensor, _target, text, _visual in rows) / len(rows):.1f}")
    loader = DataLoader(MemorySet(rows), batch_size=16, shuffle=True, collate_fn=collate)
    model = CRNN(charset.size)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = 1e9
    stall = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for images, targets, input_lengths, target_lengths, _texts in loader:
            logits = model(images)
            log_probs = logits.log_softmax(2)
            loss = loss_fn(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        say(f"epoch={epoch} loss={total / max(1, seen):.4f}")
        exact = 0
        model.eval()
        with torch.no_grad():
            previews = []
            for tensor, _target, text, visual in rows:
                pred = greedy(model(tensor.unsqueeze(0))[0].log_softmax(-1), charset)
                if pred == visual:
                    exact += 1
                if len(previews) < 5:
                    previews.append((text, to_logical(pred, charset), pred == visual))
        for text, logical, ok in previews:
            say(f"  label={text}")
            say(f"  pred ={logical} match={ok}")
        say(f"exact={exact}/{len(rows)}")
        if exact != len(rows) and (epoch == EPOCHS or exact >= len(rows) - 15):
            missed = 0
            for tensor, _target, text, visual in rows:
                pred = greedy(model(tensor.unsqueeze(0))[0].log_softmax(-1), charset)
                if pred != visual and missed < 8:
                    say(f"  fail={text}")
                    say(f"  got ={to_logical(pred, charset)}")
                    missed += 1
        model.train()
        loss_now = total / max(1, seen)
        if loss_now < best - 0.01:
            best = loss_now
            stall = 0
        else:
            stall += 1
            if stall >= 6:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                stall = 0
                say(f"lr={optimizer.param_groups[0]['lr']}")
        if exact == len(rows):
            say("memorized=yes")
            break
    out = Path(__file__).resolve().parent / "weights" / "crnn_overfit.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": charset.size}, out)
    say(f"saved={out}")


if __name__ == "__main__":
    main()
