from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """列→token: [CLS] + cat tokens + num tokens"""

    def __init__(self, cat_cardinalities: Dict[str, int], num_cols: List[str], d_model: int, dropout: float = 0.0):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.num_cols = list(num_cols)
        self.d_model = d_model

        self.emb_cat = nn.ModuleDict({c: nn.Embedding(card, d_model) for c, card in cat_cardinalities.items()})
        self.proj_num = nn.ModuleDict({c: nn.Linear(1, d_model) for c in num_cols})

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for emb in self.emb_cat.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for lin in self.proj_num.values():
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        B = cat.size(0) if cat.numel() > 0 else num.size(0)
        tokens = []

        if len(self.cat_cols) > 0:
            for i, c in enumerate(self.cat_cols):
                tokens.append(self.emb_cat[c](cat[:, i]))

        if len(self.num_cols) > 0:
            for i, c in enumerate(self.num_cols):
                tokens.append(self.proj_num[c](num[:, i : i + 1]))

        if len(tokens) > 0:
            X = torch.stack(tokens, dim=1)
        else:
            X = torch.zeros(B, 0, self.d_model, device=cat.device)

        cls = self.cls_token.expand(B, -1, -1)
        X = torch.cat([cls, X], dim=1)
        return self.dropout(X)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln1(x)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z

        z = self.ln2(x)
        z = self.ff(z)
        x = x + z
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        num_cols: List[str],
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, num_cols, d_model, dropout)
        self.blocks = nn.ModuleList([TransformerEncoder(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(cat, num)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0, :]
        return self.head(cls)
