"""Tests for Phase 2 (normal-only autoencoders). They use small synthetic data, so they run in seconds."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smart_listener import anomaly  # noqa: E402
from smart_listener.config import config_fingerprint, load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_stack_frames_shape_and_content():
    spec = np.arange(64 * 20, dtype=np.float32).reshape(64, 20)
    out = anomaly.stack_frames(spec, 5)
    assert out.shape == (16, 320)
    expected_first = np.concatenate([spec[:, t] for t in range(5)])  # frames 0..4, time-major
    assert np.array_equal(out[0], expected_first)
    expected_last = np.concatenate([spec[:, t] for t in range(15, 20)])
    assert np.array_equal(out[-1], expected_last)


def test_clip_inputs_shapes(cfg):
    spec = np.random.default_rng(0).normal(size=(64, 313)).astype(np.float32)
    mean, std = np.zeros(64, np.float32), np.ones(64, np.float32)
    assert anomaly.clip_inputs(spec, mean, std, "dense", cfg).shape == (309, 320)
    assert anomaly.clip_inputs(spec, mean, std, "conv", cfg).shape == (8, 1, 64, 64)


def test_models_forward_shapes(cfg):
    d = anomaly.make_model("dense", cfg).eval()
    c = anomaly.make_model("conv", cfg).eval()
    assert d(torch.zeros(4, 320)).shape == (4, 320)
    assert c(torch.zeros(4, 1, 64, 64)).shape == (4, 1, 64, 64)
    with pytest.raises(ValueError):
        anomaly.make_model("resnet", cfg)


def _rows():
    return pd.DataFrame({
        "file_id": [f"f{i}" for i in range(6)],
        "label": [0, 0, 0, 0, 1, 1],
        "split_ad": ["train", "train", "val", "test", "val", "test"],
    })


def test_training_rows_are_normal_train_only():
    rows = _rows()
    idx = anomaly.training_rows(rows)
    assert list(idx) == [0, 1]
    assert (rows.loc[idx, "label"] == 0).all()


def test_training_rows_reject_abnormal_in_train():
    rows = _rows()
    rows.loc[1, "label"] = 1
    with pytest.raises(ValueError):
        anomaly.training_rows(rows)


def test_auc_metrics_and_operating_point():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.8, 0.95, 0.7])
    auc, pauc = anomaly.auc_metrics(y, perfect)
    assert auc == pytest.approx(1.0) and pauc == pytest.approx(1.0)
    op = anomaly.operating_point(y, perfect, 0.5)
    assert op == {"precision": 1.0, "recall": 1.0, "fpr": 0.0}
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 2000)
    auc_rand, _ = anomaly.auc_metrics(yy, rng.random(2000))
    assert 0.45 < auc_rand < 0.55
    lo, hi = anomaly.bootstrap_auc_ci(y, perfect, n=50)
    assert lo <= 1.0 and hi == pytest.approx(1.0)


def test_dense_autoencoder_learns_normal_and_flags_abnormal(cfg):
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(4, 320)).astype(np.float32)

    def normal(n):
        return (rng.normal(size=(n, 4)).astype(np.float32) @ basis + 0.05 * rng.normal(size=(n, 320))).astype(np.float32)

    X_tr, X_va = normal(2000), normal(300)
    model = anomaly.make_model("dense", cfg)
    model, hist, best = anomaly.fit_autoencoder(model, X_tr, X_va, epochs=20, batch_size=256, lr=2e-3, patience=20,
                                                device=torch.device("cpu"), seed=0, log=lambda *_: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    X_new = normal(200)
    X_bad = X_new + rng.normal(0, 2.0, size=X_new.shape).astype(np.float32)
    with torch.no_grad():
        e_norm = ((model(torch.from_numpy(X_new)) - torch.from_numpy(X_new)) ** 2).mean(1).numpy()
        e_bad = ((model(torch.from_numpy(X_bad)) - torch.from_numpy(X_bad)) ** 2).mean(1).numpy()
    auc, _ = anomaly.auc_metrics(np.r_[np.zeros(200), np.ones(200)], np.r_[e_norm, e_bad])
    assert auc > 0.95


def test_conv_autoencoder_trains_without_error(cfg):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 1, 64, 64)).astype(np.float32)
    model = anomaly.make_model("conv", cfg)
    model, hist, _ = anomaly.fit_autoencoder(model, X, X[:16], epochs=2, batch_size=16, lr=1e-3, patience=5,
                                             device=torch.device("cpu"), seed=0, log=lambda *_: None)
    assert len(hist) == 2 and np.isfinite(hist[-1]["train_loss"])


def test_checkpoint_roundtrip_and_fingerprint_guard(cfg, tmp_path):
    model = anomaly.make_model("dense", cfg).eval()
    data = {"group": "fan_id_00", "fingerprint": config_fingerprint(cfg),
            "mean": np.zeros(64, np.float32), "std": np.ones(64, np.float32)}
    path = tmp_path / "m.pt"
    anomaly.save_checkpoint(path, model, "dense", data, [], 1, 42, {})
    loaded, ck = anomaly.load_checkpoint(path, cfg)
    x = torch.randn(3, 320)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)
    cfg2 = load_config()
    cfg2["features"]["n_mels"] = 128  # pretend the settings changed
    with pytest.raises(ValueError):
        anomaly.load_checkpoint(path, cfg2)
