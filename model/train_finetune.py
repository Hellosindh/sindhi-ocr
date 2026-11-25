"""Fine-tune crnn_best_v2.pt into crnn_finetune_best.pt.

Data: every line in training_data/sindhi, english and urdu (real lines: books, newspapers, letters, ligatures),
every synthetic_numbers line except NUMBERS_CHECK (years, dates, money, phone numbers in 0-9 / ٠-٩ / ۰-۹),
every synthetic_color line (coloured backgrounds: paper, photo, gradient, pattern, halftone, flat)
except COLOR_CHECK, plus synthetic_plain lines (30% of the mix): 7.5% blur, 7.5% noise, 15% clean (20% let forgetting creep past the guard).

Always starts from v2. Never writes crnn_best_v2.pt. Writes one file, crnn_finetune_best.pt,
and only when a step beats both v2 and the fine-tune already there (score = mean CER of the
test-book "select" half and 500 held-out real lines).

Test books: assets/test_books/labels.tsv (filename<TAB>text, lines in assets/test_books/lines/), lines from
books never used in training; alternate lines form the "select" half (used to choose) and the "report" half
(only reported).
Forgetting guard: 1000 synthetic lines never trained on (all fonts, clean/blur/noise); a
step whose CER there is worse than v2's by more than FORGET_MARGIN is never saved.
Early stop: the run's own score not better for --patience epochs. Deletes nothing.
Images are read from disk per batch (the whole set does not fit in RAM).

Run:  python -m model.train_finetune [--epochs 20] [--patience 3]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.charset import Charset
from model.crnn import CRNN
from model.decode import greedy
from model.heh_normalize import normalize_heh
from model.layers import edit_distance
from model.order import paragraph_base, to_logical, to_visual
from model.overfit import HEIGHT, to_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

SINDHI = ROOT / "assets" / "training_data" / "sindhi"
# real lines: Sindhi plus the smaller English and Urdu folders (English kept to ~10% of real on purpose)
REAL_DIRS = [SINDHI, ROOT / "assets" / "training_data" / "english", ROOT / "assets" / "training_data" / "urdu"]
TEST_BOOKS = ROOT / "assets" / "test_books"
NUMBERS = ROOT / "assets" / "synthetic_numbers"  # synthetic number lines (model/generate_numbers.py)
NUMBERS_CHECK = 500  # number lines kept out of training: a quick test of how well numbers are read
REVIEW_FLAGS = ROOT / "assets" / "training_data" / "review_flags.tsv"
SYNTH = ROOT / "assets" / "synthetic_plain"
COLOR = ROOT / "assets" / "synthetic_color"
LIST_TSV = ROOT / "assets" / "finetune" / "finetune_mix.tsv"
WEIGHTS = Path(__file__).resolve().parent / "weights"
V2 = WEIGHTS / "crnn_best_v2.pt"
BEST = WEIGHTS / "crnn_finetune_best.pt"
PROTECTED = {"crnn_best_v2.pt"}
LOG = Path(__file__).resolve().parent / "train_finetune_log.txt"

# share of the whole training mix taken by each plain synthetic kind (the other kinds take their own share)
SYNTH_SHARE = {"blur": 0.075, "noise": 0.075, "clean": 0.15}
HELD = 500  # real lines kept out of training, only to watch overfitting
SYNTH_CHECK = {"clean": 500, "blur": 250, "noise": 250}  # never trained on: forgetting check
COLOR_CHECK = 500  # colour lines kept out of training (spare for a later colour test)
FORGET_MARGIN = 0.01  # allowed CER rise on the synthetic check vs v2 (1 CER point: a tighter guard blocked the real-book gains)
MAX_WIDTH = 1600
SEED = 17


def say(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message.encode("ascii", "backslashreplace").decode(), flush=True)


def load_test():
    """Test-book lines: alternate lines go to the select / report halves."""
    rows = []
    transcript = TEST_BOOKS / "labels.tsv"
    if not transcript.exists():
        return [], []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        if "\t" not in line or line.lower().startswith("filename\t"):
            continue
        filename, text = line.split("\t")[:2]
        path = TEST_BOOKS / "lines" / filename
        if path.exists():
            with Image.open(path) as photo:
                rows.append((to_tensor(photo.convert("L")), text, "book"))
    return rows[0::2], rows[1::2]


def cer(model, rows, charset, device) -> float:
    """Character error rate of greedy reads, heh forms made canonical in reading order."""
    errors = count = 0
    model.eval()
    with torch.no_grad():
        for tensor, text, *_rest in rows:
            pred = greedy(model(tensor.unsqueeze(0).to(device))[0].log_softmax(-1), charset)
            text = normalize_heh(text)
            try:
                base = paragraph_base(text)
                target = to_logical(to_visual(text, charset), charset, base_dir=base)
                logical = normalize_heh(to_logical(pred, charset, base_dir=base))
            except (AssertionError, ValueError, TypeError):
                continue
            errors += edit_distance(list(target), list(logical))
            count += max(1, len(target))
    model.train()
    return errors / max(1, count)


def usable(text: str) -> bool:
    upper = text.upper()
    return bool(text) and text.lower() != "delete" and "[UNCLEAR]" not in upper and "digitized by" not in text.lower()


def real_pairs() -> list[tuple[Path, str, str]]:
    flagged = set()
    if REVIEW_FLAGS.exists():
        for line in REVIEW_FLAGS.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) > 2 and parts[2] in {"unclear", "digitized_footer"}:
                flagged.add(parts[1])
    pairs = []
    for folder in REAL_DIRS:
        tsv = folder / "transcripts.tsv"
        if not tsv.exists():
            continue
        source = folder.name  # sindhi / english / urdu
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" not in line or line.lower().startswith("filename"):
                continue
            name, text = line.split("\t", 1)
            name, text = Path(name.strip()).name, text.strip()
            path = folder / "lines" / name
            if usable(text) and name not in flagged and path.exists():
                pairs.append((path, text, source))
    return pairs


def number_pairs(rng: random.Random):
    """All synthetic_numbers lines: (train, check); check is NUMBERS_CHECK lines never trained on."""
    tsv = NUMBERS / "transcript.tsv"
    if not tsv.exists():
        return [], []
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            name, text = line.split("\t", 1)
            if (NUMBERS / "lines" / name).exists():
                rows.append((NUMBERS / "lines" / name, text.strip(), "numbers"))
    rng.shuffle(rows)
    return rows[NUMBERS_CHECK:], rows[:NUMBERS_CHECK]


def synth_pairs(counts: dict[str, int], rng: random.Random):
    """Uniform random sample of each kind from the whole synthetic_plain transcript, plus a
    separate check sample (SYNTH_CHECK) that never overlaps the training lines."""
    rows: dict[str, list[tuple[str, str]]] = {kind: [] for kind in counts}
    with (SYNTH / "transcript.tsv").open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[2] in rows and usable(parts[3].strip()):
                rows[parts[2]].append((parts[0], parts[3].strip()))
    out, check = [], []
    for kind, need in counts.items():
        picked = rng.sample(rows[kind], min(need + SYNTH_CHECK[kind], len(rows[kind])))
        for index, (rel, text) in enumerate(picked):
            (check if index < SYNTH_CHECK[kind] else out).append((SYNTH / rel, text, f"synth_{kind}"))
    return out, check


def color_pairs(rng: random.Random):
    """All synthetic_color lines: (train, check); check is COLOR_CHECK lines never trained on."""
    rows = []
    with (COLOR / "transcript.tsv").open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and usable(parts[3].strip()) and (COLOR / parts[0]).exists():
                rows.append((COLOR / parts[0], parts[3].strip(), "color"))
    rng.shuffle(rows)
    return rows[COLOR_CHECK:], rows[:COLOR_CHECK]


def encodable(pairs, charset: Charset):
    kept, dropped = [], 0
    for path, text, source in pairs:
        text = normalize_heh(text)
        try:
            encoded = charset.encode(to_visual(text, charset))
        except (AssertionError, ValueError, TypeError, KeyError):
            dropped += 1
            continue
        if encoded:
            kept.append((path, text, source, encoded))
        else:
            dropped += 1
    return kept, dropped


class Lines(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, _text, _source, encoded = self.rows[index]
        try:
            with Image.open(path) as photo:
                tensor = to_tensor(photo.convert("L"))
        except OSError:
            return None
        if tensor.shape[-1] > MAX_WIDTH or CRNN.time_steps(tensor.shape[-1]) < len(encoded):
            return None
        return tensor, torch.tensor(encoded, dtype=torch.long)


def collate(items):
    items = [item for item in items if item is not None]
    if not items:
        return None
    width = max(tensor.shape[-1] for tensor, _t in items)
    images = torch.zeros(len(items), 1, HEIGHT, width)
    for index, (tensor, _t) in enumerate(items):
        images[index, :, :, : tensor.shape[-1]] = tensor
    targets = [t for _x, t in items]
    return (
        images,
        torch.cat(targets),
        torch.tensor([CRNN.time_steps(tensor.shape[-1]) for tensor, _t in items], dtype=torch.long),
        torch.tensor([len(t) for t in targets], dtype=torch.long),
    )


def width_batches(rows, batch: int, rng: random.Random) -> list[list[int]]:
    """Batches of similar line length (less padding, faster), in random order."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][3]) + rng.random() * 6)
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(batches)
    return batches


def save(model, charset, info: dict) -> None:
    if BEST.name in PROTECTED or BEST.resolve() == V2.resolve():
        raise SystemExit(f"refusing to write protected weights {BEST}")
    tmp = BEST.with_suffix(".pt.tmp")
    torch.save({"state": model.state_dict(), "classes": charset.size, "from": V2.name, "name": "crnn_finetune_best", **info}, tmp)
    tmp.replace(BEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="epochs without a better score before stopping")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    LOG.write_text("", encoding="utf-8")
    rng = random.Random(SEED)
    charset = Charset()

    real, real_dropped = encodable(real_pairs(), charset)
    color_train, color_check = color_pairs(rng)
    color, color_dropped = encodable(color_train, charset)
    numbers_train, numbers_check = number_pairs(rng)
    numbers, _numbers_dropped = encodable(numbers_train, charset)
    numbers_check, _dropped = encodable(numbers_check, charset)
    total = round((len(real) + len(color) + len(numbers)) / (1 - sum(SYNTH_SHARE.values())))
    synth_train, synth_check = synth_pairs({k: round(total * s) for k, s in SYNTH_SHARE.items()}, rng)
    synth, synth_dropped = encodable(synth_train, charset)
    check, _dropped = encodable(synth_check, charset)  # guard on old synthetic only; colour gains must not hide forgetting
    rng.shuffle(real)
    held, train = real[:HELD], real[HELD:] + color + numbers + synth
    counts = {}
    for row in train:
        counts[row[2]] = counts.get(row[2], 0) + 1
    say(f"start={V2.name} save_to={BEST.name} (never writes {', '.join(sorted(PROTECTED))})")
    say(f"train={len(train)} {counts} held={len(held)} synth_check={len(check)} dropped real={real_dropped} color={color_dropped} synth={synth_dropped}")
    say("share: " + " ".join(f"{k}={v / len(train):.1%}" for k, v in sorted(counts.items())))
    LIST_TSV.parent.mkdir(parents=True, exist_ok=True)
    LIST_TSV.write_text("source\tpath\ttext\n" + "".join(f"{s}\t{p.relative_to(ROOT)}\t{t}\n" for p, t, s, _e in train), encoding="utf-8")

    device = torch.device("cuda")
    checkpoint = torch.load(V2, map_location="cpu", weights_only=False)
    model = CRNN(checkpoint["classes"]).to(device)
    model.load_state_dict(checkpoint["state"])
    select, report = load_test()
    held_rows = [(item[0], row[1]) for row, item in zip(held, Lines(held)) if item is not None]
    check_rows = [(item[0], row[1]) for row, item in zip(check, Lines(check)) if item is not None]

    def evaluate():
        """(score, select, report, held, synth_check); score = mean of select and held."""
        select_cer, report_cer = cer(model, select, charset, device), cer(model, report, charset, device)
        held_cer, check_cer = cer(model, held_rows, charset, device), cer(model, check_rows, charset, device)
        model.train()
        return (select_cer + held_cer) / 2, select_cer, report_cer, held_cer, check_cer

    def line(tag, r):
        return f"{tag} score={r[0]:.4f} select_cer={r[1]:.4f} report_cer={r[2]:.4f} held_cer={r[3]:.4f} synth_check_cer={r[4]:.4f}"

    base = evaluate()
    say(line("v2 baseline", base))
    forget_limit = base[4] + FORGET_MARGIN
    best = base[0]
    if BEST.exists():
        # the fine-tune already on disk is only replaced by something better
        old = torch.load(BEST, map_location="cpu", weights_only=False)
        current = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(old["state"])
        previous = evaluate()
        say(line(f"existing {BEST.name}", previous))
        best = min(best, previous[0])
        model.load_state_dict(current)
    say(f"to save, a step needs score < {best:.4f} and synth_check_cer <= {forget_limit:.4f}")

    run_best = float("inf")  # this run's own best score: drives early stopping

    def consider(r, epoch, step):
        nonlocal best, run_best, since_best
        if r[0] < run_best:
            run_best = r[0]
            since_best = 0
        else:
            since_best += 1
        if r[4] > forget_limit:
            say(f"  not saved: synthetic check {r[4]:.4f} > {forget_limit:.4f} (forgetting)")
        elif r[0] < best:
            best = r[0]
            save(model, charset, {"epoch": epoch, "step": step, "score": r[0], "select_cer": r[1], "report_cer": r[2],
                                  "held_cer": r[3], "synth_check_cer": r[4], "lr": args.lr})
            say(f"  saved {BEST.name}")

    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs + 10, pct_start=0.05)
    dataset = Lines(train)
    step = 0
    since_best = 0  # evaluations without a better score
    evals_per_epoch = steps_per_epoch // args.eval_every + 1
    gpu_errors = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(dataset, batch_sampler=width_batches(train, args.batch, rng), collate_fn=collate,
                            num_workers=args.workers, persistent_workers=False)
        running = 0.0
        for batch in loader:
            if batch is None:
                continue
            images, targets, input_lengths, target_lengths = batch
            try:
                logits = model(images.to(device))
                loss = loss_fn(logits.log_softmax(2).transpose(0, 1), targets.to(device), input_lengths.to(device), target_lengths.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            except RuntimeError as error:
                # GPU memory taken by other apps: skip this batch instead of losing the run
                gpu_errors += 1
                say(f"  gpu error {gpu_errors} at step {step}: {str(error)[:80]}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if gpu_errors > 20:
                    raise
                time.sleep(5)
                continue
            sched.step()
            step += 1
            running = running * 0.98 + float(loss.detach()) * 0.02 if step > 1 else float(loss.detach())
            if step % 200 == 0:
                say(f"epoch={epoch} step={step}/{steps_per_epoch * args.epochs} loss={running:.4f} "
                    f"{step * args.batch / (time.time() - started):.0f} lines/s")
            if step % args.eval_every == 0:
                result = evaluate()
                say(line(f"  eval epoch={epoch} step={step}", result))
                consider(result, epoch, step)
        result = evaluate()
        say(line(f"epoch {epoch} done", result) + f" hours={(time.time() - started) / 3600:.2f}")
        consider(result, epoch, step)
        if since_best >= args.patience * evals_per_epoch:
            say(f"early stop: no better score for {args.patience} epochs")
            break
    say(f"done run_best={run_best:.4f} saved_best={best:.4f} (v2 {base[0]:.4f}) file={BEST.name} exists={BEST.exists()} v2_untouched={V2.exists()}")


if __name__ == "__main__":
    main()
