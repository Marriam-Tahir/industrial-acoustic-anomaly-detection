"""Phase 2: normal-only autoencoders for anomaly detection (one model per machine ID).

Two models, same data, same scoring rule, so they can be compared fairly:
  dense : the MIMII-paper style baseline. 5 neighbouring spectrogram frames (5 x 64 = 320 numbers) go in,
          the same 320 numbers must come out through a narrow 8-number bottleneck.
  conv  : a convolutional autoencoder on 64 x 64 spectrogram windows (about 2 seconds of sound), using the
          window settings in config/audio_config.yaml.

Both are trained ONLY on normal training clips. The anomaly score of a clip is the mean squared
reconstruction error over all its inputs (in normalised units). Higher = less like this machine's normal sound.
It is NOT a probability and says nothing about which fault is present.
"""
import copy
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from smart_listener import audio
from smart_listener.config import config_fingerprint, resolve_path

KINDS = ("dense", "conv")
CONTEXT = 5  # number of stacked frames for the dense baseline (as in the MIMII paper)


# ----------------------------------------------------------------------------- data access
def feature_paths(cfg: dict, group: str):
    d = resolve_path(cfg, "features_dir")
    return d / "logmel" / f"{group}.npy", d / "logmel" / f"{group}_files.csv", d / "norm" / f"{group}_anomaly.json"


def list_groups(cfg: dict):
    manifest = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
    return sorted(manifest["group"].unique())


def load_group(cfg: dict, group: str) -> dict:
    """Load one machine ID: spectrograms (memory-mapped), a row table aligned with them, and the
    normalisation statistics that were fitted on this machine's TRAINING files in Phase 1."""
    npy, fcsv, nj = feature_paths(cfg, group)
    for p in (npy, fcsv, nj):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/extract_features.py first.")
    fp = config_fingerprint(cfg)
    stats = json.loads(nj.read_text(encoding="utf-8"))
    if stats["config_fingerprint"] != fp:
        raise ValueError(f"{nj.name}: made with different audio/feature settings. Re-run extract_features.py --force.")
    meta_path = resolve_path(cfg, "features_dir") / "features_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("config_fingerprint") != fp:
            raise ValueError("features_meta.json fingerprint differs from the current config. Re-run extract_features.py --force.")
    manifest = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
    files = pd.read_csv(fcsv)
    rows = files.merge(manifest[["file_id", "label", "label_name", "split_ad", "index_in_label"]],
                       on="file_id", how="left", validate="one_to_one")
    if rows["split_ad"].isna().any():
        raise ValueError(f"{group}: some feature rows are not in the manifest")
    spec = np.load(npy, mmap_mode="r")
    if spec.shape[0] != len(rows):
        raise ValueError(f"{group}: {spec.shape[0]} feature rows but {len(rows)} file ids")
    return {"group": group, "spec": spec, "rows": rows, "fingerprint": fp,
            "mean": np.asarray(stats["mean"], dtype=np.float32), "std": np.asarray(stats["std"], dtype=np.float32)}


def select_rows(rows: pd.DataFrame, split: str, normal_only: bool = False) -> np.ndarray:
    sub = rows[rows["split_ad"] == split]
    if normal_only:
        sub = sub[sub["label"] == 0]
    return sub.index.to_numpy()


def training_rows(rows: pd.DataFrame) -> np.ndarray:
    """Row positions of the clips the autoencoder may learn from: normal training clips, nothing else."""
    idx = select_rows(rows, "train")
    if (rows.loc[idx, "label"] != 0).any():
        raise ValueError("training split contains abnormal clips; an anomaly detector must learn from normal clips only")
    if len(idx) == 0:
        raise ValueError("no training clips found")
    return idx


# ----------------------------------------------------------------------------- inputs
def stack_frames(spec_norm: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    """(n_mels, frames) -> (frames - context + 1, context * n_mels). Row t holds frames t..t+context-1, time-major."""
    w = np.lib.stride_tricks.sliding_window_view(spec_norm, context, axis=1)  # (n_mels, T', context)
    return np.ascontiguousarray(w.transpose(1, 2, 0)).reshape(w.shape[1], -1)


def clip_inputs(spec: np.ndarray, mean: np.ndarray, std: np.ndarray, kind: str, cfg: dict) -> np.ndarray:
    """One clip's log-mel spectrogram (n_mels, 313) -> model inputs.
    dense -> (309, 320);  conv -> (8, 1, 64, 64)."""
    x = audio.apply_norm(np.asarray(spec), mean, std)
    if kind == "dense":
        return stack_frames(x)
    f = cfg["features"]
    w = audio.make_windows(x, f["window_frames"], f["window_hop_frames"])
    return np.ascontiguousarray(w[:, None, :, :], dtype=np.float32)


def build_inputs(data: dict, row_idx, kind: str, cfg: dict) -> np.ndarray:
    return np.concatenate([clip_inputs(data["spec"][i], data["mean"], data["std"], kind, cfg) for i in row_idx]).astype(np.float32)


# ----------------------------------------------------------------------------- models
class DenseAE(nn.Module):
    """320 -> 64 -> 64 -> 8 -> 64 -> 64 -> 320 (the structure of the MIMII baseline)."""

    def __init__(self, d_in: int = CONTEXT * 64, hidden: int = 64, bottleneck: int = 8):
        super().__init__()

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

        self.net = nn.Sequential(*block(d_in, hidden), *block(hidden, hidden), *block(hidden, bottleneck),
                                 *block(bottleneck, hidden), *block(hidden, hidden), nn.Linear(hidden, d_in))

    def forward(self, x):
        return self.net(x)


class ConvAE(nn.Module):
    """Four stride-2 convolutions down to a 64-number code, four transposed convolutions back up."""

    def __init__(self, n_mels: int = 64, n_frames: int = 64, latent: int = 64):
        super().__init__()
        if n_mels % 16 or n_frames % 16:
            raise ValueError("n_mels and window_frames must be multiples of 16 for this model")
        self.hs, self.ws = n_mels // 16, n_frames // 16

        def down(i, o):
            return [nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU()]

        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU()]

        self.enc = nn.Sequential(*down(1, 16), *down(16, 32), *down(32, 64), *down(64, 64), nn.Flatten(),
                                 nn.Linear(64 * self.hs * self.ws, latent))
        self.fc = nn.Sequential(nn.Linear(latent, 64 * self.hs * self.ws), nn.ReLU())
        self.dec = nn.Sequential(*up(64, 64), *up(64, 32), *up(32, 16), nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, x):
        z = self.enc(x)
        h = self.fc(z).view(-1, 64, self.hs, self.ws)
        return self.dec(h)


def make_model(kind: str, cfg: dict) -> nn.Module:
    if kind == "dense":
        return DenseAE(CONTEXT * cfg["features"]["n_mels"])
    if kind == "conv":
        return ConvAE(cfg["features"]["n_mels"], cfg["features"]["window_frames"])
    raise ValueError(f"unknown model kind {kind!r}; use one of {KINDS}")


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------- training
def group_seed(seed: int, group: str, kind: str) -> int:
    return (seed + zlib.crc32(f"{group}|{kind}".encode("utf-8"))) % (2 ** 31)


@torch.no_grad()
def mean_loss(model: nn.Module, X: torch.Tensor, device, batch: int = 2048) -> float:
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(X), batch):
        xb = X[s:s + batch].to(device)
        total += float(((model(xb) - xb) ** 2).sum().item())
        n += xb.numel()
    return total / n


def fit_autoencoder(model, X_train: np.ndarray, X_val_normal: np.ndarray, epochs: int, batch_size: int, lr: float,
                    patience: int, device, seed: int, weight_decay: float = 1e-5, log=print):
    """Train on normal data. The best epoch is chosen by reconstruction loss on NORMAL validation clips
    (no abnormal labels, no test data involved). Returns (model with best weights, history)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device)
    Xt, Xv = torch.from_numpy(X_train), torch.from_numpy(X_val_normal)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        run, n = 0.0, 0
        for s in range(0, len(Xt), batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:  # BatchNorm needs at least 2 samples
                continue
            xb = Xt[idx].to(device)
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.item()) * len(idx)
            n += len(idx)
        tr, va = run / max(n, 1), mean_loss(model, Xv, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_normal_loss": va})
        improved = va < best - 1e-6
        if improved:
            best, best_state, best_epoch, bad = va, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        log(f"    epoch {epoch:3d}  train {tr:.4f}  val-normal {va:.4f}{'  <- best' if improved else ''}")
        if bad >= patience:
            log(f"    early stop (no improvement for {patience} epochs)")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


@torch.no_grad()
def score_rows(model, data: dict, row_idx, kind: str, cfg: dict, device, clip_chunk: int = 32) -> np.ndarray:
    """Anomaly score per clip: columns = [mean error, max error] over the clip's inputs. Shape (n_clips, 2)."""
    model.eval()
    out = []
    for s in range(0, len(row_idx), clip_chunk):
        idx = row_idx[s:s + clip_chunk]
        parts = [clip_inputs(data["spec"][i], data["mean"], data["std"], kind, cfg) for i in idx]
        n_per = len(parts[0])
        X = torch.from_numpy(np.concatenate(parts)).to(device)
        err = ((model(X) - X) ** 2).flatten(1).mean(1).cpu().numpy().reshape(len(idx), n_per)
        out.append(np.stack([err.mean(1), err.max(1)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2))


def save_checkpoint(path: Path, model, kind: str, data: dict, history, best_epoch: int, seed: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "group": data["group"], "state_dict": model.state_dict(),
                "config_fingerprint": data["fingerprint"], "norm_mean": data["mean"], "norm_std": data["std"],
                "history": history, "best_epoch": best_epoch, "seed": seed, **extra}, path)


def load_checkpoint(path: Path, cfg: dict, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["config_fingerprint"] != config_fingerprint(cfg):
        raise ValueError(f"{path.name} was trained with different audio/feature settings")
    model = make_model(ck["kind"], cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


# ----------------------------------------------------------------------------- metrics
def auc_metrics(y, s, max_fpr: float = 0.1):
    """ROC-AUC and partial AUC (false-positive rate up to max_fpr, standardised so 0.5 = chance, as in DCASE)."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)), float(roc_auc_score(y, s, max_fpr=max_fpr))


def operating_point(y, s, threshold: float):
    """Precision, recall and false-positive rate when score > threshold counts as 'abnormal'."""
    y, s = np.asarray(y), np.asarray(s)
    pred = s > threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def bootstrap_auc_ci(y, s, n: int = 1000, seed: int = 0):
    """95% interval for ROC-AUC by resampling normal and abnormal clips separately."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    vals = []
    for _ in range(n):
        p, q = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        vals.append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
