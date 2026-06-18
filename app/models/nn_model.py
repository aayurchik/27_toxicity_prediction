import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class StrongToxNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        out = self.input(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.head(out)
        return out