"""Training loops of the sequential baselines, grid search on the validation split, prediction.

LSTM, CNN   Adam, class-weighted cross-entropy, learning rate halved after 3 epochs
            without improvement, early stopping on validation loss (patience 5), best epoch kept.
SASRec      one-step-shifted targets at every position, class-weighted cross-entropy summed over
            positions, Adam (0.9, 0.98), gradient clipping 1.0, same schedule and stopping rule.
DROS        the official DROS objective on the SASRec backbone: BCE on one sampled negative plus
            alpha times the distributionally robust term; early stopping on validation weighted F1.
Hyperparameters with several candidate values are chosen on validation weighted F1.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data, predictions
from .models import SASREC_PAD, build_model, forward_logits, load_dros, make_inputs


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse class frequency, scaled to sum to the number of classes."""
    w = 1.0 / np.bincount(y, minlength=config.NUM_CLASSES)
    return torch.tensor(w / w.sum() * config.NUM_CLASSES, dtype=torch.float32)


def grid(**candidates) -> list[dict]:
    """All combinations of the candidate values, e.g. grid(lr=[1e-3, 5e-4], hidden_size=[128])."""
    keys = list(candidates)
    return [dict(zip(keys, values)) for values in itertools.product(*(candidates[k] for k in keys))]


@torch.no_grad()
def predict(arch: str, model: nn.Module, codes: np.ndarray, cfg: dict, device="cpu", batch_size=1024) -> np.ndarray:
    """Predicted class codes for (N, X) input windows."""
    model.eval()
    out = [forward_logits(arch, model, make_inputs(arch, codes[i:i + batch_size], cfg, device)).argmax(1).cpu()
           for i in range(0, len(codes), batch_size)]
    return torch.cat(out).numpy()


def weighted_f1(y, p) -> float:
    return float(f1_score(y, p, average="weighted", labels=range(config.NUM_CLASSES), zero_division=0))


@dataclass
class Fitted:
    model: nn.Module
    cfg: dict
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- LSTM and CNN
def fit_classifier(arch: str, cfg: dict, train: np.ndarray, val: np.ndarray, device="cpu", seed=config.SEED) -> Fitted:
    """train, val: (N, X + 1) codes; the last column is the target."""
    seed_everything(seed)
    model = build_model(arch, cfg, device)
    x_tr, x_va = make_inputs(arch, train[:, :-1], cfg, device)[0], make_inputs(arch, val[:, :-1], cfg, device)[0]
    y_tr, y_va = torch.tensor(train[:, -1], device=device), torch.tensor(val[:, -1], device=device)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=cfg["batch_size"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train[:, -1]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    return _fit(model, cfg, train_loader, val_loader, lambda xb, yb: loss_fn(model(xb), yb), optimizer,
                clip=1.0 if arch == "lstm" else None)


# --------------------------------------------------------------------------- SASRec
def sasrec_tensors(windows: np.ndarray, maxlen: int):
    """Inputs = first X codes, targets = the same sequence shifted by one; left-padded to maxlen."""
    x = windows.shape[1] - 1
    inputs = np.full((len(windows), maxlen), SASREC_PAD, dtype=np.int64)
    targets = np.full((len(windows), maxlen), -100, dtype=np.int64)
    inputs[:, maxlen - x:] = windows[:, :-1] + 1
    targets[:, maxlen - x:] = windows[:, 1:]
    return torch.tensor(inputs), torch.tensor(targets)


def fit_sasrec(cfg: dict, train: np.ndarray, val: np.ndarray, device="cpu", seed=config.SEED) -> Fitted:
    seed_everything(seed)
    model = build_model("sasrec", cfg, device)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param.data)
    x_tr, y_tr = sasrec_tensors(train, cfg["maxlen"])
    x_va, y_va = sasrec_tensors(val, cfg["maxlen"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr[y_tr >= 0].numpy()).to(device), ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98))
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=cfg["batch_size"])

    def batch_loss(xb, yb):
        return loss_fn(model(xb.to(device)).reshape(-1, config.NUM_CLASSES), yb.to(device).reshape(-1))

    return _fit(model, cfg, train_loader, val_loader, batch_loss, optimizer, clip=1.0)


def _fit(model, cfg, train_loader, val_loader, batch_loss, optimizer, clip=None) -> Fitted:
    """Epoch loop with learning-rate halving and early stopping on validation loss."""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(xb, yb)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([batch_loss(xb, yb).item() for xb, yb in val_loader]))
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return Fitted(model, cfg, {"epochs_run": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss})


# --------------------------------------------------------------------------- DROS-SASRec
def fit_dros(cfg: dict, train: np.ndarray, val: np.ndarray, dros_dir, device="cpu", seed=config.SEED) -> Fitted:
    """Training step of the official `SASRec_bce.py` with the DROS term; inputs are exactly state_size long."""
    dros = load_dros(dros_dir)
    device = torch.device(device)
    x = train.shape[1] - 1
    buffer = pd.DataFrame({"seq": list(train[:, :-1]), "len_seq": x, "next": train[:, -1]})
    ps = torch.tensor(dros.calcu_propensity_score(buffer), device=device)
    seed_everything(seed)
    model = build_model("dros", cfg, device, dros_dir=dros_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-8, weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss()
    alpha, beta, batch_size = cfg["alpha"], cfg["beta"], cfg["batch_size"]
    rng = np.random.default_rng(seed)
    best_f1, best_epoch, best_state, bad = -1.0, 0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for _ in range(len(train) // batch_size):
            rows = train[rng.choice(len(train), size=batch_size, replace=False)]
            seq = torch.tensor(rows[:, :-1], device=device)
            target = torch.tensor(rows[:, -1:], device=device)
            shift = torch.tensor(rng.integers(1, config.NUM_CLASSES, size=(batch_size, 1)), device=device)
            negative = (target + shift) % config.NUM_CLASSES   # one negative class, uniform over the other three
            output = model.forward(seq, torch.full((batch_size,), x, device=device))
            scores = torch.cat((torch.gather(output, 1, target), torch.gather(output, 1, negative)), 0)
            labels = torch.cat((torch.ones(batch_size, 1), torch.zeros(batch_size, 1)), 0).to(device)
            loss = bce(scores, labels)
            pos_scores_dro = torch.gather(torch.mul(output * output, ps), 1, target).squeeze()
            pos_loss_dro = torch.gather(torch.mul((output - 1) * (output - 1), ps), 1, target).squeeze()
            inner = (torch.sum(torch.exp(torch.mul(output * output, ps) / beta), 1)
                     - torch.exp(pos_scores_dro / beta) + torch.exp(pos_loss_dro / beta))
            loss = loss + alpha * torch.mean(torch.log(inner + 1e-24))
            if not torch.isfinite(loss):
                return Fitted(model, cfg, {"diverged": True})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_f1 = weighted_f1(val[:, -1], predict("dros", model, val[:, :-1], cfg, device))
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return Fitted(model, cfg, {"epochs_run": epoch, "best_epoch": best_epoch})


# --------------------------------------------------------------------------- data, grid search and saving
def training_windows(data_dir, length: int = config.TRAIN_LENGTH):
    """(N, X + 1) codes of the training and validation customers, one window per customer."""
    train_tx, _ = data.load_split(data_dir, "train")
    val_tx, _ = data.load_split(data_dir, "val")
    train, val = data.last_windows(train_tx, length).codes(), data.last_windows(val_tx, length).codes()
    print(f"training windows {len(train):,}, validation windows {len(val):,} (length {length})", flush=True)
    return train, val


def write_test_predictions(arch: str, fitted: Fitted, data_dir, out_dir: Path, name: str,
                           lengths=config.EVAL_LENGTHS, device="cpu") -> None:
    """Predict every test customer with at least X + 1 transactions, at every length X."""
    test_tx, _ = data.load_split(data_dir, "test")
    for x in lengths:
        w = data.last_windows(test_tx, x)
        pred = predict(arch, fitted.model, w.codes()[:, :-1], fitted.cfg, device)
        path = predictions.write(Path(out_dir) / predictions.file_name(name, x), w.customer_id, w.target,
                                 predictions.to_names(pred))
        print(f"  length {x} (N = {len(w):,}): {path}", flush=True)


def grid_search(arch: str, configs: list[dict], fit_fn, val: np.ndarray, device="cpu") -> tuple[Fitted, list]:
    """Train every configuration and keep the one with the best validation weighted F1."""
    best, log = None, []
    for cfg in configs:
        start = time.perf_counter()
        fitted = fit_fn(cfg)
        seconds = time.perf_counter() - start
        if fitted.info.get("diverged"):
            log.append({"config": cfg, "diverged": True})
            print(f"  {cfg}: diverged, skipped", flush=True)
            continue
        score = weighted_f1(val[:, -1], predict(arch, fitted.model, val[:, :-1], cfg, device))
        fitted.info.update(val_weighted_f1=score, seconds=round(seconds, 1))
        log.append({"config": cfg, **fitted.info})
        print(f"  {cfg}: validation weighted F1 {score:.4f} ({seconds:.0f} s)", flush=True)
        if best is None or score > best.info["val_weighted_f1"]:
            best = fitted
    if best is None:
        raise RuntimeError("every configuration diverged")
    return best, log


def save_run(run_dir: Path, arch: str, fitted: Fitted, log: list) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in fitted.model.state_dict().items()}
    torch.save({"arch": arch, "config": fitted.cfg, "categories": config.CATEGORIES, "state_dict": state},
               run_dir / "model.pt")
    (run_dir / "training_log.json").write_text(json.dumps({"selected": fitted.cfg, "grid": log}, indent=1, default=str))
