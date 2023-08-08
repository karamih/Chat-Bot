import torch
import torch.nn as nn


class ChatBotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_size, 8),
                                nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(8, 8),
                                nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(8, output_size))

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out