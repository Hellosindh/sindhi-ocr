# Sindhi OCR

Free Sindhi OCR by [Hello Sindh](https://hellosindh.com): turn pictures, scans and PDFs of Sindhi books into editable Sindhi text, on your own computer.

<p>
  <a href="https://github.com/Hellosindh/sindhi-ocr/releases/latest/download/SindhiOCR-Setup.exe"><img alt="Download for Windows" src="https://img.shields.io/badge/Download-Windows%2010%20%2F%2011-0078D4?style=for-the-badge&logo=windows&logoColor=white" height="40"></a>
  &nbsp;
  <a href="https://play.google.com/store/apps/details?id=com.hellosindh.sindhiocr"><img alt="Android (coming soon)" src="https://img.shields.io/badge/Android-coming%20soon-3DDC84?style=for-the-badge&logo=android&logoColor=white" height="40"></a>
</p>

---

# Open-source models

This repository also holds the models and code behind the app, for reading printed **Sindhi** (Arabic
script) from scans and photos: a line reader, a line detector, a page-layout model and a table model, with
the scripts that made their training data and trained them. All of it was designed and trained by Hello
Sindh and is released under the **MIT License**, code and weights alike, so you can use it, change it and
fine-tune it for your own books, newspapers or forms. The weights are in the release
[Model weights v1](https://github.com/Hellosindh/sindhi-ocr/releases/tag/weights-v1).

Third-party models are not included; see [What is not included](#what-is-not-included).

## Contents

- [The models](#the-models)
- [Test results](#test-results)
- [Quick start](#quick-start)
- [How a page is read](#how-a-page-is-read)
- [Fine-tuning the line reader](#fine-tuning-the-line-reader)
- [Avoiding catastrophic forgetting](#avoiding-catastrophic-forgetting)
- [Fine-tuning the layout model](#fine-tuning-the-layout-model)
- [Training the line detector and table model](#training-the-line-detector-and-table-model)
- [Making synthetic data](#making-synthetic-data)
- [Unwarping curled pages](#unwarping-curled-pages)
- [What is not included](#what-is-not-included)
- [Repository layout](#repository-layout)
- [License](#license)

---

## The models

| File | What it does | Architecture | Size |
|---|---|---|---|
| `crnn_finetune_best.pt` | **Line reader** (use this one): one text line image in, text out | CNN + 2-layer BiLSTM + CTC, input height 32 px | 8.8 M params |
| `crnn_best_v2.pt` | Line reader, **base model**: trained only on synthetic lines; the start point of the fine-tune | same | 8.8 M |
| `linedet.pt` | **Line detector**: finds every text line on a page and its type (body, heading, header, footer, footnote) | small U-Net, gray page + x/y coordinate channels, 1/2-resolution maps | 8.3 M |
| `layout.pt` | **Page layout**: regions (Text, Title, Section-header, Picture, Table, Caption, Footnote, List-item, Page-header, Page-footer, Formula, Contents-entry) | U-Net, page letterboxed to 640 px, per-pixel class + distance to the region's four edges | 8.4 M |
| `layout_news.pt` | Page layout also trained on newspaper pages; adds **Masthead** (and Jump-line) | same, 14 classes | 8.4 M |
| `table.pt` | **Table structure**: row gaps, column gaps and header rows of a cropped table | same U-Net body, 3 outputs | 8.3 M |

The reader's character set (267 classes: Sindhi, Urdu and English letters, vowel marks, Sindhi / Arabic /
Latin digits, punctuation) is in [`character_dictionary.txt`](character_dictionary.txt). Index 0 is the
CTC blank.

All models are small on purpose: they run on an ordinary laptop CPU, and on a phone after ONNX export
(`model/export_android.py`).

## Test results

Character error rate (CER) and word error rate (WER), lower is better. Greedy decoding, no language model,
no word fixer: only the reader (and for photos, the detector) is measured. Vowel marks count.

| Test set | Fine-tuned reader | Base reader (v2) |
|---|---|---|
| Books: 4,256 lines from 15 books never used in training | **5.4% CER**, 19.8% WER | 10.4% CER, 33.2% WER |
| Newspaper lines: 188 lines from 30 issues never used in training (headlines, ads, tables, Urdu, English) | **19.5% CER**, 50.0% WER | 52.8% CER, 85.1% WER |
| Phone photos: 14 whole book pages, line detector + reader | **0.9% CER**, 3.9% WER | 2.7% CER, 10.1% WER |

Where the reader still struggles (useful if you fine-tune): white-on-black headline bars, land-record
number tables like `145/10, (00_20)`, small Urdu text in Nastaliq ad fonts, and *pesh* / *shad*
(the reader prints about 83% of zabar and zer marks, 61% of pesh, almost no shad: most training labels
leave marks out).

Layout (box AP at IoU 0.5, mean over classes):

| Model | DocLayNet (documents) | Synthetic Sindhi pages | Sindhi newspaper pages |
|---|---|---|---|
| `layout.pt` | 0.42 | 0.64 | 0.05 |
| `layout_news.pt` | 0.40 | 0.64 | 0.30 (masthead 0.64) |

Table model: rows exactly right on 82% (PubTables-1M) / 80% (synthetic Sindhi tables) of tables, columns on
97% / 99.75%.

Speed, measured on a laptop: a 16-line book photo takes about 1.0 s to straighten, 1.9 s to find lines and
0.7 s to read on a laptop CPU; with an RTX 3050 laptop GPU the layout model takes about
35 ms per page.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/Hellosindh/sindhi-ocr.git
cd sindhi-ocr
pip install -r requirements.txt          # install torch first from pytorch.org if you want CUDA
python download_weights.py               # all weights, about 200 MB  (or: python download_weights.py reader)
python ocr_page.py page.jpg              # prints the page text and the time of each step
python ocr_page.py *.jpg --out text/     # also writes text/<name>.txt
```

Read single line images from your own code:

```python
import torch
from PIL import Image
from model.charset import Charset
from model.crnn import CRNN
from model.decode import greedy
from model.order import to_logical
from model.overfit import to_tensor

cs = Charset()
ck = torch.load("model/weights/crnn_finetune_best.pt", map_location="cpu", weights_only=False)
reader = CRNN(ck["classes"]).eval()
reader.load_state_dict(ck["state"])

x = to_tensor(Image.open("line.png").convert("L")).unsqueeze(0)   # (1, 1, 32, W), ink high
with torch.no_grad():
    text = to_logical(greedy(reader(x)[0].log_softmax(-1), cs), cs)  # visual -> logical (typing) order
print(text)
```

Score readers on your own labelled lines (before and after a fine-tune):

```bash
python evaluate.py --lines my_test/lines --labels my_test/labels.tsv \
       --weights model/weights/crnn_finetune_best.pt model/weights/crnn_best_v2.pt
```

Draw the layout model's regions on pages: `python -m layout.demo pages/ --layout model/weights/layout_news.pt`.

## How a page is read

1. **Straighten** (`layout/orient.py`): tilt, sideways and upside-down pages are found from the ink's row
   profile; the reader breaks ties by how confidently it reads a few lines.
2. **(Optional) unwarp** a curled photo with a separate model, see [Unwarping](#unwarping-curled-pages).
3. **Find lines** (`linedet/detect.py`): the detector paints a thin core band through every line plus the
   line's height and type; each band becomes a line box. Small print is re-read at a larger scale.
4. **Reading order** (`reading_order`): header lines, then columns right to left (XY cut), footnotes,
   footer.
5. **Layout and tables** (optional, `layout/`): region boxes tag lines as headings, captions, tables;
   tables are split into rows and columns by `table.pt`.
6. **Read each line** (`model/crnn.py`): the crop gets an 8 px white border, is scaled to height 32 and
   read with greedy CTC; the reader outputs visual order, `model/order.py` turns it into logical order.

## Fine-tuning the line reader

### Your data

One folder per source, each with:

```
my_lines/
  lines/0001.png  0002.png ...
  transcripts.tsv            # filename<TAB>text, one line per image, text in logical (typing) order
```

Before training, clean the labels; these rules removed most of the bad labels in our data:

- drop lines marked unsure (`[UNCLEAR]`, `??`), empty labels, labels without any Arabic-script letter;
- write Sindhi yeh `ي` (U+064A), not Persian yeh `ی` (U+06CC);
- in Sindhi text, `۾` is often mistyped as `پر` (they look alike in many fonts) and `ڪ` as `ک`; check them;
- **compare every label with what the current reader sees** and drop the line when they differ by more than
  30% CER (`python -m model.check_lines my_lines` writes `my_lines/check.tsv` with the CER of every
  line). Such lines almost always have the wrong label or a bad crop;
- never train on lines from the books or newspaper issues you test on.

Heh forms (ھ / ه / ہ) are written in many ways by different typists; labels and reads are made canonical
by `model/heh_normalize.py` (rules in `model/heh_rules.md`) before training and scoring.

### The recipe we used

`model/train_finetune.py` made `crnn_finetune_best.pt` from `crnn_best_v2.pt`. Point its folder
constants at your data (real lines in `assets/training_data/<language>/`, test lines from books you never train
on in `assets/test_books/`) and run `python -m model.train_finetune --epochs 20 --patience 3`.

**Training mix**: 423,366 line images, drawn so that each kind takes this share of every epoch:

| Kind | Lines | Share |
|---|---|---|
| Real Sindhi lines (books, newspaper lines, hand-checked lines) | 216,822 | 51.2% |
| Real English lines | 19,977 | 4.7% |
| Real Urdu lines | 10,476 | 2.5% |
| Synthetic, clean (`make_synth.py`, `model/generate_500k.py`) | 63,580 | 15.0% |
| Synthetic, blurred | 31,790 | 7.5% |
| Synthetic, noisy | 31,790 | 7.5% |
| Synthetic coloured backgrounds: posters, banners, photos (`model/generate_color.py`) | 29,431 | 7.0% |
| Synthetic numbers: dates, money, phone numbers in 3 digit sets (`model/generate_numbers.py`) | 19,500 | 4.6% |

So about **58% real and 42% synthetic**, with the plain synthetic lines (the base model's own training
data) held at **30%**. English was kept near 5% on purpose so the reader stays a Sindhi reader.

**Settings**: Adam, learning rate 3e-5 with a one-cycle schedule (5% warm-up, then down), batch 16 with
lines of similar width batched together, lines up to 1600 px wide, up to 20 epochs, early stop after 3
epochs without improvement. The best step came at epoch 16 (step 416,000); one epoch over the mix took
about 1 h on an RTX 3050 laptop GPU (around 110 lines/s).

**What came out** (the numbers the training script logs):

| | Base v2 | Fine-tuned |
|---|---|---|
| Book test, "select" half (CER) | 10.42% | **5.37%** |
| 500 held-out real lines (CER) | 11.73% | **3.71%** |
| 1,000 held-out synthetic lines, the forgetting check (CER) | 3.61% | **2.49%** |

The synthetic check got *better*, not worse: nothing the base model knew was lost.

### Continue from the fine-tune, or start again from v2?

- **New data, same model shape** (more books, a new newspaper, new fonts): continue from
  `crnn_finetune_best.pt`, train on **all** the data mixed together (old + new, real + synthetic, in the
  shares above), and restart the learning-rate schedule (short warm-up, then decay). It already knows what
  v2 knows; starting over from v2 costs many hours and usually ends at the same place.
- **Never train on the new data alone**: that is what makes a model forget.
- **Start from v2 (or from scratch)** only when the model itself changes: a new input height (for example
  40-48 px for small vowel marks), new characters in the character set, or a different network.
- When unsure, run both starts for a few hours on the same mix and keep the one that scores better on your
  test lines.

## Avoiding catastrophic forgetting

Catastrophic forgetting is a fine-tuned model getting worse at what it could do before (other fonts, clean
synthetic text, numbers) while it learns your new data. The training script guards against it in four ways:

1. **Mix old data into every batch.** The base model's own kind of data (plain synthetic lines) stays at
   30% of the mix. With 20% we saw the forgetting check creep past its limit; at 30% it held. Coloured
   lines (7%) and number lines (4.6%) keep those skills alive too.
2. **A held-out forgetting check.** 1,000 synthetic lines that are never trained on (500 clean, 250 blurred,
   250 noisy, all fonts) are read at every evaluation.
3. **A hard limit on forgetting.** A step may only be saved if its CER on that check is no more than
   **1 CER point** (absolute, `FORGET_MARGIN = 0.01`) worse than the base model's. v2 read the check at
   3.61%, so no saved model may read it worse than 4.61%. A tighter guard blocked real gains on books;
   1 point let them through without visible damage.
4. **Save only real improvements.** The score is the mean CER of the book test's "select" half and 500
   held-out real lines; a step is written only when it beats both the base model and the fine-tune already
   on disk. The other half of the book test ("report") is only looked at, never used to choose, so the
   reported number is honest.

Also kept out of training: 500 number lines and 500 coloured lines (quick checks of those skills).

The base weights are never overwritten: the script refuses to write `crnn_best_v2.pt`.

## Fine-tuning the layout model

```bash
python -m layout.train --sets doclaynet,sindhi_synth,my_pages --init model/weights/layout.pt \
       --out model/weights/layout_mine.pt --epochs 5 --lr 1e-3
```

- Each set is a folder `assets/layout_data/<set>/{train,test}/` with images and a `labels.jsonl`
  (`{"image": ..., "w": ..., "h": ..., "boxes": [[x0, y0, x1, y1, class_id], ...]}`).
- **Every set gets an equal share of each epoch**, whatever its size, so old sets keep their weight
  (this is the forgetting guard here), and every set's test split is scored after each epoch.
- `--init` starts from earlier weights; when you add classes (append them to `CLASSES` in
  `layout/model.py`), the old classes keep their trained head rows. Old 12-class weights still load.
- The best epoch is saved by the mean AP over all sets, so a gain on your pages cannot hide a loss on the
  others. Our newspaper run: newspapers 0.05 -> 0.30 AP, documents 0.42 -> 0.40, synthetic Sindhi pages
  0.645 -> 0.639.
- Newspaper labels were made automatically from the line detector's output
  (`layout/news_lines.py`, `layout/news_labels.py`, `layout/news_dataset.py`): stacked body lines become
  Text blocks, tall lines headlines, large text-free ink areas pictures, a giant line at the top of page 1
  the masthead. Hold out whole issues for testing.

## Training the line detector and table model

- **Line detector**: `python -m linedet.synth_pages --count 5000` draws synthetic book, newspaper and
  magazine pages with exact line boxes; `python -m linedet.train --steps 30000` trains on random 512 px
  tiles (pages scaled to 1100-2300 px on the long side). The published detector was trained only on
  synthetic pages.
- **Table model**: `python -m layout.tables --epochs 8`, on PubTables-1M plus synthetic Sindhi tables
  (`python -m layout.synth_tables`).

## Making synthetic data

All generators render real Sindhi text with HarfBuzz shaping (correct joining, marks and kashida) in any
TrueType font. Put your fonts in `Fonts/`.

| Script | Lines it makes |
|---|---|
| `make_synth.py`, `model/generate_500k.py` | plain lines, clean / blurred / noisy, from a text corpus (`sindhi_corpus.txt`, one sentence per line) |
| `model/generate_color.py` | posters and banners: colour backgrounds, outlines, shadows, stretched kashida |
| `model/generate_numbers.py` | dates, money, phone numbers, in Latin, Arabic-Indic and Persian digits |
| `model/generate_newfonts.py` | lines in a second font folder, `Fonts_extra/` (fonts checked for every Sindhi letter) |
| `model/generate_notebook.py` | "handwriting": a pen-like font (`--font`) on ruled notebook paper, word wobble, bold, italic, kashida drawn only in the image |
| `model/generate_gaps.py` | the reader's weak spots: white-on-black headlines, land-record numbers, Urdu, vowel-marked text, column pieces joined with " \| " |

Keep at least one font out of all synthetic data and test on it: set `SINDHI_HELD_OUT_FONT` to its file name
and no generator will use it. Then you can see whether the reader learned letter shapes or just memorised
fonts. `make_synth.py` lists which characters each font in `Fonts/` has on first use (`font_glyph_map.json`).

## Unwarping curled pages

Photos of thick books curl near the binding, which bends the lines. We flatten such photos **before** line
detection with a separate document-unwarping model, for example
[UVDoc](https://github.com/tanguymagne/UVDoc) (a grid-based unwarping network). It is not part of this
repository; plug any unwarper in front of `ocr_page.py`:

```python
flat = unwarper(gray)                   # same size, page flattened
blocks = reading_order(detector.detect(flat))
```

Use it only for photos: on flat scans it can blur the text slightly.

## What is not included

- Third-party models, such as the unwarping model.

## Repository layout

```
ocr_page.py              read page images end to end
evaluate.py              score readers on labelled line images
download_weights.py      fetch the trained weights (GitHub release)
character_dictionary.txt reader's 267 classes
make_synth.py            synthetic line rendering (HarfBuzz + FreeType)
model/                   line reader: network, decoding, reading order, heh rules, training, generators
linedet/                 line detector: network, detection + reading order, synthetic pages, training
layout/                  page layout, tables, straightening, newspaper auto-labels, training
```

## License

The model code and trained weights are MIT licensed, see [LICENSE](LICENSE). Made by Hello Sindh.
