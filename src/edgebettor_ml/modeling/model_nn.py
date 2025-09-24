from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class MultiHeadNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.BatchNorm1d(hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = hs
        self.trunk = nn.Sequential(*layers)
        self.head_home_win = nn.Sequential(nn.Linear(last, 1), nn.Sigmoid())
        self.head_home_cover = nn.Sequential(nn.Linear(last, 1), nn.Sigmoid())
        self.head_over = nn.Sequential(nn.Linear(last, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "p_home_win": self.head_home_win(h).squeeze(-1),
            "p_home_cover": self.head_home_cover(h).squeeze(-1),
            "p_over": self.head_over(h).squeeze(-1),
        }


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15


def brier_score(y_true: torch.Tensor, y_prob: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_prob - y_true) ** 2)


def train_model(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    cfg: TrainConfig,
    class_weights: Dict[str, float] | None = None,
) -> Tuple[MultiHeadNet, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCELoss(reduction="none")

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    X_tr = to_tensor(X_train)
    X_va = to_tensor(X_val)
    ytr = {k: to_tensor(v) for k, v in y_train.items()}
    yva = {k: to_tensor(v) for k, v in y_val.items()}

    best_val = float("inf")
    best_state = None
    patience = cfg.early_stopping_patience
    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        loss_heads = []
        for head, y in [("p_home_win", ytr["y_home_win"]), ("p_home_cover", ytr["y_home_cover_vs_closing"]), ("p_over", ytr["y_over_total_vs_closing"])]:
            loss_vec = bce(out[head], y)
            if class_weights and head in class_weights:
                loss_vec = loss_vec * class_weights[head]
            loss_heads.append(loss_vec.mean())
        loss = sum(loss_heads)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_va)
            briers = [
                brier_score(pred_val["p_home_win"], yva["y_home_win"]),
                brier_score(pred_val["p_home_cover"], yva["y_home_cover_vs_closing"]),
                brier_score(pred_val["p_over"], yva["y_over_total_vs_closing"]),
            ]
            val_metric = torch.stack(briers).mean().item()
        if val_metric < best_val:
            best_val = val_metric
            best_state = model.state_dict()
            patience = cfg.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {"val_brier_mean": best_val}
    return model, metrics


