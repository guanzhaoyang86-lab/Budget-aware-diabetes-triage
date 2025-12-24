from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_t)
        return logits / T


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, lbfgs_steps: int = 10, lr: float = 0.01):
    model = TemperatureScaler().to(logits.device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(model(logits), labels)
        loss.backward()
        return loss

    for _ in range(lbfgs_steps):
        optimizer.step(closure)

    return model
