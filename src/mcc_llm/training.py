"""Training loops of the sequential baselines, grid search on the validation split, prediction.

LSTM, CNN   Adam, cross-entropy, learning rate halved after 3 epochs without improvement,
            early stopping on validation loss (patience 5), best epoch kept.
SASRec      one-step-shifted targets at every position, cross-entropy averaged over positions,
            Adam (0.9, 0.98), gradient clipping 1.0, same schedule and stopping rule.
DROS        the official DROS objective on the SASRec backbone: BCE on one sampled negative plus
            alpha times the distributionally robust term; early stopping on the validation score.
The cross-entropy is either unweighted or weighted by inverse class frequency; this is a grid option
(config.CLASS_WEIGHTS). Hyperparameters with several candidate values are chosen on the validation
split by config.SELECTION_METRIC (macro-F1). The selected configuration is then refitted with every
seed of config.BASELINE_SEEDS; every model is saved, so it can be scored again later without training
(run() with score_only).
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data, predictions
from .evaluation import selection_score
from .models import SASREC_PAD, build_model, forward_logits, load_dros, make_inputs


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray, kind: str = "inverse") -> torch.Tensor:
    """Loss weights per class. "inverse": inverse class frequency scaled to sum to the number of classes;
    "none": equal weights."""
    if kind == "none":
        return torch.ones(config.NUM_CLASSES, dtype=torch.float32)
    w = 1.0 / np.maximum(np.bincount(y, minlength=config.NUM_CLASSES), 1)
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


@torch.no_grad()
def predict_proba(arch: str, model: nn.Module, codes: np.ndarray, cfg: dict, device="cpu", batch_size=1024) -> np.ndarray:
    """(N, 4) softmax class probabilities for (N, X) input windows (for PR-AUC and calibration)."""
    model.eval()
    out = [forward_logits(arch, model, make_inputs(arch, codes[i:i + batch_size], cfg, device)).float().softmax(1).cpu()
           for i in range(0, len(codes), batch_size)]
    return torch.cat(out).numpy()


def weighted_f1(y, p) -> float:
    return selection_score(y, p, "F1 (w)")


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
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train[:, -1], cfg.get("class_weights", "inverse")).to(device))
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
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr[y_tr >= 0].numpy(), cfg.get("class_weights", "inverse")).to(device),
                                  ignore_index=-100)
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
        val_f1 = selection_score(val[:, -1], predict("dros", model, val[:, :-1], cfg, device))
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
                           lengths=config.EVAL_LENGTHS, device="cpu", splits=("test",)) -> None:
    """Predict every customer of each evaluation split with at least X + 1 transactions, at every length X.

    One sub-folder of `out_dir` per split (for example bank_a_test, bank_b_confirm), with class
    probabilities in the files. With the single split "test" (prepare_data.py --earlier-rule), files go straight
    into `out_dir`.
    """
    for split in splits:
        tx, _ = data.load_split(data_dir, split)
        folder = Path(out_dir) if tuple(splits) == ("test",) else Path(out_dir) / split
        for x in lengths:
            w = data.last_windows(tx, x)
            if len(w) == 0:
                continue
            proba = predict_proba(arch, fitted.model, w.codes()[:, :-1], fitted.cfg, device)
            path = predictions.write(folder / predictions.file_name(name, x), w.customer_id, w.target,
                                     predictions.to_names(proba.argmax(1)), probs=proba)
            print(f"  {split}, length {x} (N = {len(w):,}): {path}", flush=True)


def grid_search(arch: str, configs: list[dict], fit_fn, val: np.ndarray, device="cpu") -> tuple[Fitted, list]:
    """Train every configuration and keep the one with the best validation score (config.SELECTION_METRIC)."""
    best, log = None, []
    for cfg in configs:
        start = time.perf_counter()
        fitted = fit_fn(cfg)
        seconds = time.perf_counter() - start
        if fitted.info.get("diverged"):
            log.append({"config": cfg, "diverged": True})
            print(f"  {cfg}: diverged, skipped", flush=True)
            continue
        val_pred = predict(arch, fitted.model, val[:, :-1], cfg, device)
        score = selection_score(val[:, -1], val_pred)
        fitted.info.update(val_selection=score, val_weighted_f1=weighted_f1(val[:, -1], val_pred),
                           selection_metric=config.SELECTION_METRIC, seconds=round(seconds, 1))
        log.append({"config": cfg, **fitted.info})
        print(f"  {cfg}: validation {config.SELECTION_METRIC} {score:.4f} ({seconds:.0f} s)", flush=True)
        if best is None or score > best.info["val_selection"]:
            best = fitted
    if best is None:
        raise RuntimeError("every configuration diverged")
    return best, log


def checkpoint_path(run_dir: Path, seed: int) -> Path:
    """model.pt for the default seed, model-seed<k>.pt for the others."""
    return Path(run_dir) / ("model.pt" if seed == config.SEED else f"model-seed{seed}.pt")


def save_model(run_dir: Path, arch: str, fitted: Fitted, seed: int) -> Path:
    path = checkpoint_path(run_dir, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in fitted.model.state_dict().items()}
    torch.save({"arch": arch, "config": fitted.cfg, "categories": list(config.CATEGORIES), "seed": seed,
                "state_dict": state}, path)
    return path


def load_model(run_dir: Path, seed: int, device="cpu", dros_dir=None) -> tuple[str, Fitted]:
    """Architecture name and model of a saved run, ready for prediction."""
    path = checkpoint_path(run_dir, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: train the model first (the script without --score-only)")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if list(ckpt["categories"]) != list(config.CATEGORIES):
        raise ValueError(f"{path} was trained with the classes {ckpt['categories']}, not {list(config.CATEGORIES)}")
    model = build_model(ckpt["arch"], ckpt["config"], device, dros_dir=dros_dir)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["arch"], Fitted(model.eval(), ckpt["config"], {"seed": ckpt["seed"]})


def run(arch: str, name: str, configs: list[dict], fit, args, dros_dir=None) -> None:
    """Shared driver of the baseline training scripts.

    `fit(cfg, train, val, seed)` trains one model. The grid `configs` is searched on the validation split
    with the first seed; the selected configuration is then refitted with every other seed. Every model
    is saved in args.run_dir and scored on args.splits. With args.score_only nothing is trained: the saved
    models are loaded and scored on args.splits.
    """
    if args.score_only:
        fitted = {seed: load_model(args.run_dir, seed, args.device, dros_dir)[1] for seed in args.seeds}
    else:
        train, val = training_windows(args.data)
        best, log = grid_search(arch, configs, lambda cfg: fit(cfg, train, val, args.seeds[0]), val, args.device)
        fitted = {args.seeds[0]: best}
        for seed in args.seeds[1:]:
            refit = fit(best.cfg, train, val, seed)
            if refit.info.get("diverged"):
                print(f"  seed {seed}: diverged, skipped", flush=True)
                continue
            refit.info["val_selection"] = selection_score(val[:, -1], predict(arch, refit.model, val[:, :-1], refit.cfg, args.device))
            fitted[seed] = refit
        args.run_dir.mkdir(parents=True, exist_ok=True)
        for seed, model in fitted.items():
            save_model(args.run_dir, arch, model, seed)
        seeds = {seed: {k: v for k, v in m.info.items() if k != "diverged"} for seed, m in fitted.items()}
        (args.run_dir / "training_log.json").write_text(json.dumps(
            {"selected": best.cfg, "selection_metric": config.SELECTION_METRIC, "grid": log, "seeds": seeds},
            indent=1, default=str))
        print(f"selected {best.cfg}; {len(fitted)} models saved in {args.run_dir}", flush=True)
    for seed, model in fitted.items():
        tag = name if seed == config.SEED else f"{name}-seed{seed}"
        write_test_predictions(arch, model, args.data, args.predictions, tag, device=args.device, splits=args.splits)


def add_run_arguments(p, run_dir: str) -> None:
    """Command-line options shared by the baseline training scripts."""
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--run-dir", type=Path, default=Path(run_dir))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--splits", nargs="+", default=list(config.EVAL_SPLITS),
                   help="evaluation splits to score (default: all)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(config.BASELINE_SEEDS),
                   help="the grid is searched with the first seed; the selected configuration is refitted with every seed")
    p.add_argument("--score-only", action="store_true",
                   help="train nothing: load the saved models of --run-dir and score --splits")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
