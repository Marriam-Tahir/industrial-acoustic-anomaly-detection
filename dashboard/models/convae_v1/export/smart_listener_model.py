"""
smart_listener_model.py - Convolutional autoencoder + inference helper (needs torch).

Input  : (1, 64, 32) log-mel window in [0, 1]   (one 1 s window from the Member 1 data)
Output : reconstruction of the same shape. Anomaly score = mean squared reconstruction error.

Shapes through the network (base=32):
  encoder 1x64x32 -> 32x32x16 -> 64x16x8 -> 128x8x4 -> flatten 4096 -> latent
  decoder latent -> 4096 -> 128x8x4 -> 64x16x8 -> 32x32x16 -> 1x64x32 (sigmoid)

Use for new audio (Member 3):
    from smart_listener_model import AnomalyScorer
    scorer = AnomalyScorer.from_dir("export/fan_6_dB", "preprocessing_config.json")
    result = scorer.score_file("clip.wav")   # dict: score, threshold, is_anomaly, window_scores
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import smart_listener_scoring as SC

IN_H, IN_W = 64, 32


def _down(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, 2, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))


def _up(c_in, c_out):
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):
    def __init__(self, base=32, latent=64):
        super().__init__()
        self.base, self.latent = base, latent
        self.encoder = nn.Sequential(_down(1, base), _down(base, base * 2), _down(base * 2, base * 4))
        self.flat = base * 4 * (IN_H // 8) * (IN_W // 8)
        self.to_latent = nn.Linear(self.flat, latent)
        self.from_latent = nn.Linear(latent, self.flat)
        self.decoder = nn.Sequential(_up(base * 4, base * 2), _up(base * 2, base),
                                     nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.to_latent(self.encoder(x).flatten(1))
        h = self.from_latent(z).view(-1, self.base * 4, IN_H // 8, IN_W // 8)
        return self.decoder(h)


@torch.no_grad()
def window_errors(model, X, device, batch=1024):
    """X: (n_recordings, n_windows, 64, 32) float32 -> (n_recordings, n_windows) mean squared error."""
    model.eval()
    n, w = X.shape[:2]
    flat = np.ascontiguousarray(X.reshape(-1, 1, IN_H, IN_W), dtype=np.float32)
    out = np.empty(len(flat), dtype=np.float32)
    for i in range(0, len(flat), batch):
        xb = torch.from_numpy(flat[i:i + batch]).to(device)
        out[i:i + batch] = ((model(xb) - xb) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return out.reshape(n, w)


@torch.no_grad()
def reconstruct(model, window, device):
    """window: (64, 32) -> reconstruction (64, 32)."""
    model.eval()
    x = torch.from_numpy(np.asarray(window, dtype=np.float32)).view(1, 1, IN_H, IN_W).to(device)
    return model(x)[0, 0].cpu().numpy()


def export_bundle(out_dir, model, arch, preprocessing_fingerprint, aggregation, threshold, extra=None):
    """Save everything Member 3 needs to score new audio: weights + settings + threshold."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
    cfg = dict(arch=arch, input_shape=[1, IN_H, IN_W], preprocessing_fingerprint=preprocessing_fingerprint,
               aggregation=aggregation, threshold=float(threshold), anomaly_rule="score > threshold")
    cfg.update(extra or {})
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


class AnomalyScorer:
    def __init__(self, model, cfg, preprocessor, device):
        self.model, self.cfg, self.pre, self.device = model, cfg, preprocessor, device

    @classmethod
    def from_dir(cls, bundle_dir, preprocessing_config_json, device=None):
        import smart_listener_preprocess as SLP
        bundle_dir = Path(bundle_dir)
        cfg = json.loads((bundle_dir / "model_config.json").read_text(encoding="utf-8"))
        pre = SLP.Preprocessor.from_json(preprocessing_config_json)
        if pre.fingerprint != cfg["preprocessing_fingerprint"]:
            raise ValueError(f"preprocessing fingerprint {pre.fingerprint} does not match the one this model was "
                             f"trained with ({cfg['preprocessing_fingerprint']}); scores would be meaningless")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = ConvAutoencoder(**cfg["arch"]).to(device)
        model.load_state_dict(torch.load(bundle_dir / "model.pt", map_location=device))
        model.eval()
        return cls(model, cfg, pre, device)

    def score_array(self, x):
        """x: (19, 64, 32) array from Preprocessor -> result dict."""
        err = window_errors(self.model, np.asarray(x, dtype=np.float32)[None], self.device)
        score = float(SC.aggregate(err, self.cfg["aggregation"])[0])
        return dict(score=score, threshold=self.cfg["threshold"], is_anomaly=bool(score > self.cfg["threshold"]),
                    window_scores=err[0].tolist())

    def score_file(self, path):
        return self.score_array(self.pre.preprocess_file(path))

    def score_wav_bytes(self, data):
        return self.score_array(self.pre.preprocess_wav_bytes(data))
