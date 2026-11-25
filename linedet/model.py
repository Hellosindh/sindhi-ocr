"""Small U-Net that paints text lines on a whole page in one pass.

Input: gray page plus two coordinate channels (x and y as a fraction of the page),
so the net knows where on the page it is looking: a header sits at the top, a page
number at the bottom.

Output at half resolution:
    0       line core: a thin band through the middle of every text line
    1       log line height (in input pixels), read on core pixels
    2..6    line type logits: body, heading, header, footer, footnote
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TYPES = 5
OUT_CHANNELS = 2 + N_TYPES


def conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cout, 2), conv(cout, cout), conv(cout, cout))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(conv(cin + cskip, cout), conv(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat([x, skip], 1))


class LineNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        c = [width, width * 2, width * 4, width * 6, width * 8, width * 8]
        self.stem = nn.Sequential(conv(3, c[0], 2), conv(c[0], c[0]))  # /2
        self.d1 = Down(c[0], c[1])  # /4
        self.d2 = Down(c[1], c[2])  # /8
        self.d3 = Down(c[2], c[3])  # /16
        self.d4 = Down(c[3], c[4])  # /32
        self.d5 = Down(c[4], c[5])  # /64: wide view for headers/footers and big headlines
        self.u4 = Up(c[5], c[4], c[4])
        self.u3 = Up(c[4], c[3], c[3])
        self.u2 = Up(c[3], c[2], c[2])
        self.u1 = Up(c[2], c[1], c[1])
        self.u0 = Up(c[1], c[0], c[0])
        self.head = nn.Conv2d(c[0], OUT_CHANNELS, 1)

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
        y = self.u0(y, s0)
        return self.head(y)


def page_input(gray: torch.Tensor, x0: float = 0.0, y0: float = 0.0, page_w: float | None = None, page_h: float | None = None) -> torch.Tensor:
    """(B,1,H,W) uint8-range gray -> (B,3,H,W) net input with page-fraction coordinates.

    x0/y0 is where this tile starts on the page and page_w/page_h the page size, in the
    same pixels; a whole page leaves them at the defaults.
    """
    b, _c, h, w = gray.shape
    page_w = page_w or w
    page_h = page_h or h
    xs = (torch.arange(w, device=gray.device, dtype=torch.float32) + x0) / page_w
    ys = (torch.arange(h, device=gray.device, dtype=torch.float32) + y0) / page_h
    xx = xs.view(1, 1, 1, w).expand(b, 1, h, w)
    yy = ys.view(1, 1, h, 1).expand(b, 1, h, w)
    img = 1.0 - gray.float() / 255.0  # ink is high
    return torch.cat([img, xx * 2 - 1, yy * 2 - 1], 1)
