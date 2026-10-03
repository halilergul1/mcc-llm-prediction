"""Sequential baselines of Section 3.6: LSTM, CNN, SASRec and DROS-SASRec.

All four read the integer-encoded category sequence (codes 0..3, oldest first) and
return logits over the four classes for the next transaction.

SASRec follows the PyTorch port of Huang (2020), https://github.com/pmixer/SASRec.pytorch
(Apache-2.0), with a four-way linear classifier in place of the ranking head.
DROS-SASRec uses the authors' released code (https://github.com/YangZhengyi98/DROS),
which is not redistributed here; see `load_dros`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config

SASREC_PAD = 0                    # SASRec: token 0 is padding, classes are tokens 1..4
DROS_PAD = config.NUM_CLASSES     # official DROS code: the padding item is `item_num`


class LSTMModel(nn.Module):
    """Single-layer LSTM over learned embeddings of the class codes, linear head on the last hidden state."""

    def __init__(self, hidden_size: int = 128, embedding_dim: int = 32, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                        # x: (B, X) int64 class codes, oldest first
        out, _ = self.lstm(self.embedding(x))
        return self.linear(out[:, -1])


class CNNModel(nn.Module):
    """Three causal dilated convolutions (dilation 1, 2, 4; kernel 3) over class embeddings, in the style of
    NextItNet; the output at the last position (receptive field 15) feeds the linear head.
    """

    def __init__(self, embedding_dim: int = 32, channels: int = 64, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.dilations = (1, 2, 4)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim if i == 0 else channels, channels, kernel_size=3, dilation=d)
            for i, d in enumerate(self.dilations))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):                        # x: (B, X) int64 class codes, oldest first
        h = self.embedding(x).transpose(1, 2)    # (B, E, X)
        for conv, d in zip(self.convs, self.dilations):
            h = self.relu(conv(nn.functional.pad(h, (2 * d, 0))))   # left padding only: causal, length kept
        return self.fc(h[:, :, -1])


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2) + inputs


class SASRecClassifier(nn.Module):
    """Self-attention blocks with learned positional embeddings and causal masking.

    Returns logits at every position, (B, L, classes); the last position is the prediction.
    Padded positions (inputs shorter than maxlen) are zeroed after the embedding and after
    every block, as in the reference implementation.
    """

    def __init__(self, hidden_units=50, num_blocks=2, num_heads=1, dropout_rate=0.2, maxlen=9,
                 num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.item_emb = nn.Embedding(num_classes + 1, hidden_units, padding_idx=SASREC_PAD)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, tokens):                   # tokens: (B, L) int64, 0 = padding (left)
        keep = (tokens != SASREC_PAD).unsqueeze(-1)
        seqs = self.item_emb(tokens) * (self.item_emb.embedding_dim ** 0.5)
        positions = torch.arange(1, tokens.shape[1] + 1, device=tokens.device).expand_as(tokens)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions * (tokens != SASREC_PAD))) * keep
        length = seqs.shape[1]
        causal = ~torch.tril(torch.ones((length, length), dtype=torch.bool, device=tokens.device))
        for ln_attn, attn, ln_ff, ff in zip(self.attention_layernorms, self.attention_layers,
                                            self.forward_layernorms, self.forward_layers):
            q = ln_attn(seqs)
            out, _ = attn(q, seqs, seqs, attn_mask=causal)
            seqs = ff(ln_ff(q + out)) * keep
        return self.classifier(self.last_layernorm(seqs))


def load_dros(dros_dir: str | Path):
    """Import `SASRec_bce.py` of the official DROS repository."""
    dros_dir = Path(dros_dir).resolve()
    if not (dros_dir / "SASRec_bce.py").exists():
        raise FileNotFoundError(
            f"official DROS code not found in {dros_dir}; get it with\n"
            f"  git clone {config.DROS_REPO} {dros_dir} && git -C {dros_dir} checkout {config.DROS_COMMIT}"
        )
    if str(dros_dir) not in sys.path:
        sys.path.insert(0, str(dros_dir))
    import SASRec_bce  # noqa: E402  (official module)

    SASRec_bce.item_num = config.NUM_CLASSES   # global read by calcu_propensity_score
    return SASRec_bce


def build_model(arch: str, cfg: dict, device="cpu", dros_dir=None) -> nn.Module:
    if arch == "lstm":
        model = LSTMModel(cfg["hidden_size"], cfg.get("embedding_dim", 32))
    elif arch == "cnn":
        model = CNNModel(cfg.get("embedding_dim", 32), cfg.get("channels", 64))
    elif arch == "sasrec":
        model = SASRecClassifier(cfg["hidden_units"], cfg["num_blocks"], cfg["num_heads"],
                                 cfg["dropout_rate"], cfg["maxlen"])
    elif arch == "dros":
        dros = load_dros(dros_dir)
        model = dros.SASRec(cfg["hidden_size"], config.NUM_CLASSES, cfg["state_size"], cfg["dropout"], torch.device(device))
    else:
        raise ValueError(arch)
    return model.to(device)


def make_inputs(arch: str, codes: np.ndarray, cfg: dict, device="cpu"):
    """Model inputs for (B, X) category codes, oldest first."""
    codes = np.asarray(codes, dtype=np.int64)
    if arch in ("lstm", "cnn"):                  # integer codes for the embedding layer
        return (torch.tensor(codes, dtype=torch.long, device=device),)
    if arch == "sasrec":                         # tokens 1..4; keep the last maxlen, pad on the left
        length = cfg["maxlen"]
        out = np.full((len(codes), length), SASREC_PAD, dtype=np.int64)
        tail = codes[:, -length:] + 1
        out[:, length - tail.shape[1]:] = tail
        return (torch.tensor(out, device=device),)
    if arch == "dros":                           # keep the last state_size items, pad on the left
        size = cfg["state_size"]
        out = np.full((len(codes), size), DROS_PAD, dtype=np.int64)
        tail = codes[:, -size:]
        out[:, size - tail.shape[1]:] = tail
        return torch.tensor(out, device=device), torch.full((len(codes),), size, dtype=torch.long, device=device)
    raise ValueError(arch)


def forward_logits(arch: str, model: nn.Module, inputs) -> torch.Tensor:
    """(B, classes) logits for the next transaction."""
    if arch == "sasrec":
        return model(inputs[0])[:, -1, :]
    if arch == "dros":
        return model.forward_eval(*inputs).reshape(-1, config.NUM_CLASSES)
    return model(inputs[0])
