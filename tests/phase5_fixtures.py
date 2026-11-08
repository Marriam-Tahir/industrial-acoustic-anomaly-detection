"""A tiny fake MIMII-style project for the Phase 5 tests (nothing here touches your real data).

It makes synthetic 'fan' and 'pump' recordings, then runs the REAL pipeline code on them: log-mel features,
normalisation, dense autoencoder training, and severity calibration. The result behaves like a miniature
version of the real project, so inference, monitoring and the API can be tested end to end in about a minute.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from smart_listener import anomaly, audio, severity
from smart_listener.config import config_fingerprint, load_config, resolve_path

SR, SECONDS = 16000, 10


def synth_clip(kind: str, abnormal: bool, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(SR * SECONDS) / SR
    f0 = 120.0 if kind == "fan" else 200.0
    y = 0.10 * np.sin(2 * np.pi * f0 * t * (1 + 0.002 * rng.standard_normal())) \
        + 0.05 * np.sin(2 * np.pi * 3 * f0 * t) + 0.02 * rng.standard_normal(len(t))
    if abnormal:                                    # an extra tone plus short clicks: clearly unlike normal sound
        y += 0.08 * np.sin(2 * np.pi * 1800 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
        for c in rng.integers(0, len(t) - 200, 12):
            y[c:c + 80] += 0.4 * rng.standard_normal(80)
    return np.clip(y, -0.99, 0.99).astype(np.float32)


def build_project(root: Path, groups=("fan_id_06", "pump_id_04"), n_train=48, n_each=40, epochs=6, seed=0,
                  default_layout=False) -> dict:
    """default_layout=True builds inside a project folder with the REAL relative paths (data/interim/mono16k, ...),
    which lets the scripts and the real server run on it."""
    root = Path(root)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "audio_config.yaml")
    rel = cfg["paths"]
    cfg["paths"] = rel if default_layout else {"raw_zip_dir": str(root / "Dataset"), "mono_dir": str(root / "mono"),
                    "manifest_dir": str(root / "manifests"), "features_dir": str(root / "processed"),
                    "figures_dir": str(root / "figures")}
    mono_dir, man_dir, feat_dir = (resolve_path(cfg, k) for k in ("mono_dir", "manifest_dir", "features_dir"))
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        kind, ident = g.split("_", 1)
        for label_name in ("normal", "abnormal"):
            count = n_train + 2 * n_each if label_name == "normal" else 2 * n_each
            d = mono_dir / kind / ident / label_name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                y = synth_clip(kind, label_name == "abnormal", rng)
                sf.write(d / f"{i:08d}.wav", y, SR, subtype="PCM_16")
                if label_name == "normal":
                    split = "train" if i < n_train else ("val" if i < n_train + n_each else "test")
                else:
                    split = "val" if i < n_each else "test"
                rows.append({"file_id": f"{g}_{label_name}_{i:08d}", "path": f"{kind}/{ident}/{label_name}/{i:08d}.wav",
                             "machine_type": kind, "machine_id": ident, "group": g, "label_name": label_name,
                             "label": int(label_name == "abnormal"), "snr_db": 6, "index_in_label": i,
                             "exclude_reason": "", "split_ad": split, "split_cls": "train"})
    man = pd.DataFrame(rows)
    man_dir.mkdir(parents=True, exist_ok=True)
    man.to_csv(man_dir / "manifest.csv", index=False)

    fp = config_fingerprint(cfg)
    out = feat_dir
    (out / "logmel").mkdir(parents=True, exist_ok=True)
    (out / "norm").mkdir(parents=True, exist_ok=True)
    (out / "features_meta.json").write_text(json.dumps({"config_fingerprint": fp}))
    for g in groups:
        sub = man[man["group"] == g].sort_values(["label_name", "index_in_label"]).reset_index(drop=True)
        specs = np.stack([audio.log_mel(audio.load_mono_wav(mono_dir / p, cfg), cfg) for p in sub["path"]])
        np.save(out / "logmel" / f"{g}.npy", specs)
        sub[["file_id"]].to_csv(out / "logmel" / f"{g}_files.csv", index=False)
        rs = audio.RunningStats(cfg["features"]["n_mels"])
        for s in specs[(sub["split_ad"] == "train").to_numpy()]:
            rs.update(s)
        mean, std = rs.finalize()
        (out / "norm" / f"{g}_anomaly.json").write_text(json.dumps(
            {"mean": mean.tolist(), "std": std.tolist(), "config_fingerprint": fp, "n_files": rs.n_files}))

    models = root / "models" / "ad"
    machines, pooled = {}, []
    for g in groups:
        data = anomaly.load_group(cfg, g)
        tr = anomaly.training_rows(data["rows"])
        vn = anomaly.select_rows(data["rows"], "val", normal_only=True)
        model = anomaly.make_model("dense", cfg)
        model, hist, best = anomaly.fit_autoencoder(model, anomaly.build_inputs(data, tr, "dense", cfg),
                                                    anomaly.build_inputs(data, vn, "dense", cfg), epochs=epochs,
                                                    batch_size=512, lr=1e-3, patience=5, device="cpu", seed=seed,
                                                    log=lambda *_: None)
        anomaly.save_checkpoint(models / "dense" / f"{g}.pt", model, "dense", data, hist, best, seed, {})
        everything = np.arange(len(data["rows"]))
        allsc = anomaly.score_rows(model, data, everything, "dense", cfg, "cpu")
        rep = root / "reports" / "phase2"
        rep.mkdir(parents=True, exist_ok=True)
        tbl = data["rows"][["file_id", "label", "split_ad"]].rename(columns={"split_ad": "split"}).copy()
        tbl["score_mean"], tbl["score_max"] = allsc[:, 0], allsc[:, 1]
        tbl.to_csv(rep / f"scores_dense_{g}.csv", index=False)
        sc = allsc[anomaly.select_rows(data["rows"], "val", normal_only=True), 0]
        c, s = severity.fit_center_scale(sc)
        machines[g] = {"model": "dense", "center": c, "scale": s, "n_val_normal": int(len(vn))}
        pooled += list(severity.to_z(sc, c, s))
    cal = {"transform": "z = (ln(score) - center) / scale", "machines": machines,
           "thresholds": {"z_warn": float(np.quantile(pooled, 0.95)), "z_alarm": float(np.quantile(pooled, 0.99))},
           "rule": {**severity.DEFAULT_RULE, "n_window": 6, "k_warn": 2, "m_high": 3, "k_high": 2, "recover": 5,
                    "cooldown": 180, "single_alarm_warning": True}}
    # make the demo limits meaningful even though the tiny validation set gives noisy quantiles
    cal["thresholds"]["z_warn"] = max(cal["thresholds"]["z_warn"], 3.0)
    cal["thresholds"]["z_alarm"] = max(cal["thresholds"]["z_alarm"], cal["thresholds"]["z_warn"] + 3.0)
    cal_path = root / "config" / "severity_calibration.json" if default_layout else root / "calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(cal, indent=2))
    return {"root": root, "cfg": cfg, "calibration": cal, "models_dir": models, "groups": list(groups),
            "manifest": man}


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return build_project(tmp_path_factory.mktemp("synth_project"))


@pytest.fixture()
def analyzer(synth):
    from smart_listener.inference import Analyzer
    return Analyzer(synth["cfg"], synth["models_dir"], synth["calibration"])
