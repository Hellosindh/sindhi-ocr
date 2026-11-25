"""Small line recognizer: CNN, then BiLSTM, then a linear layer.

CTC loss is applied outside this module. Class 0 is the blank.
"""

from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            512,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(512, num_classes)
        self.form_fc = nn.Linear(512, 5)

    @staticmethod
    def time_steps(width: int) -> int:
        return width // 4 - 1

    def sequence(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        sequence = features.squeeze(2).transpose(1, 2)
        sequence, _state = self.rnn(sequence)
        return sequence

    def forward(self, images: torch.Tensor, return_forms: bool = False):
        sequence = self.sequence(images)
        characters = self.fc(sequence)
        if return_forms:
            return characters, self.form_fc(sequence)
        return characters

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
