"""Draw detected lines on pages: colour = line type, number = reading order.

Run:  python -m linedet.show <folder with page images> <output folder>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linedet.detect import LineDetector, reading_order  # noqa: E402

COLOURS = {"body": (0, 170, 0), "heading": (255, 0, 0), "header": (0, 0, 255), "footer": (255, 0, 255), "footnote": (0, 150, 255)}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    det = LineDetector()
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gray = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        det.detect(gray)  # warm
        started = time.time()
        lines = det.detect(gray)
        ms = (time.time() - started) * 1000
        blocks = reading_order(lines)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        thick = max(1, round(max(gray.shape) / 1200))
        order = 0
        for block in blocks:
            for line in block:
                order += 1
                cv2.polylines(vis, [np.int32(line["quad"])], True, COLOURS[line["type"]], thick)
                x1, y0 = line["box"][2], line["box"][1]
                cv2.putText(vis, str(order), (x1 + 2, y0 + 12 * thick), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, (0, 0, 200), thick)
        scale = min(1.0, 1400 / max(gray.shape))
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        (out / f"{path.stem}.jpg").write_bytes(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        types = {}
        for line in lines:
            types[line["type"]] = types.get(line["type"], 0) + 1
        print(f"{path.name} {gray.shape} lines={len(lines)} blocks={len(blocks)} {types} {ms:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
