import copy

import numpy as np
import pandas as pd
import pytest

from conftest import make_tree, write_wav
from smart_listener import audit
from smart_listener.manifest import (assign_anomaly_splits, assign_classifier_splits, scan_folder,
                                     validate_manifest)


@pytest.fixture
def manifest(tmp_path, cfg):
    make_tree(tmp_path)
    df, unexpected = scan_folder(tmp_path, cfg)
    assert not unexpected
    return assign_classifier_splits(assign_anomaly_splits(df, cfg), cfg)


def test_scan_counts_and_natural_order(tmp_path, cfg):
    write_wav(tmp_path / "fan/id_00/normal/2.wav", np.zeros(16000, "float32"))
    write_wav(tmp_path / "fan/id_00/normal/10.wav", np.zeros(16000, "float32"))
    df, _ = scan_folder(tmp_path, cfg)
    assert list(df["path"].str.split("/").str[-1]) == ["2.wav", "10.wav"]


def test_unexpected_files_reported(tmp_path, cfg):
    write_wav(tmp_path / "fan/id_00/normal/a.wav", np.zeros(16000, "float32"))
    write_wav(tmp_path / "fan/id_99/normal/a.wav", np.zeros(16000, "float32"))
    df, unexpected = scan_folder(tmp_path, cfg)
    assert len(df) == 1 and len(unexpected) == 1


def test_clean_manifest_passes(manifest, cfg):
    assert validate_manifest(manifest, cfg) == []


def test_block_split_properties(tmp_path, cfg):
    cfg["split"]["anomaly"]["strategy"] = "block"
    make_tree(tmp_path)
    df, _ = scan_folder(tmp_path, cfg)
    manifest = assign_classifier_splits(assign_anomaly_splits(df, cfg), cfg)
    assert validate_manifest(manifest, cfg) == []
    b = cfg["split"]["anomaly"]["buffer_files"]
    for _, g in manifest.groupby("group"):
        assert not ((g.split_ad == "train") & (g.label == 1)).any()
        for s in ("val", "test"):
            x = g[g.split_ad == s]
            assert (x.label == 0).sum() == (x.label == 1).sum() > 0
        assert set(g.split_ad) == {"train", "val", "test", "buffer"}
        n = g[g.label == 0]
        # order: validation block first, training in the middle, test block last
        assert n[n.split_ad == "val"].index_in_label.max() < n[n.split_ad == "train"].index_in_label.min() - b
        assert n[n.split_ad == "train"].index_in_label.max() < n[n.split_ad == "test"].index_in_label.min() - b


def test_deterministic(tmp_path, cfg):
    make_tree(tmp_path)
    cfg["split"]["anomaly"]["strategy"] = "random"
    df, _ = scan_folder(tmp_path, cfg)
    a, b = assign_anomaly_splits(df, cfg), assign_anomaly_splits(df, cfg)
    assert a["split_ad"].equals(b["split_ad"])
    cfg2 = copy.deepcopy(cfg)
    cfg2["split"]["seed"] = 1
    assert not a["split_ad"].equals(assign_anomaly_splits(df, cfg2)["split_ad"])
    assert validate_manifest(assign_classifier_splits(a, cfg), cfg) == []


def test_classifier_splits_by_machine(manifest):
    per = manifest.groupby("group")["split_cls"].nunique()
    assert (per == 1).all()
    assert set(manifest.split_cls) == {"train", "val", "test"}


def test_too_few_abnormal_raises(tmp_path, cfg):
    make_tree(tmp_path, n_abn=3)
    df, _ = scan_folder(tmp_path, cfg)
    with pytest.raises(ValueError):
        assign_anomaly_splits(df, cfg)


def test_validation_catches_injected_leaks(manifest, cfg):
    bad = manifest.copy()
    i = bad.index[(bad.group == "fan_id_00") & (bad.label == 1)][0]
    bad.loc[i, "split_ad"] = "train"
    assert any("ABNORMAL" in p for p in validate_manifest(bad, cfg))

    bad = manifest.copy()
    j = bad.index[(bad.group == "fan_id_00") & (bad.split_ad == "test") & (bad.label == 0)][0]
    bad.loc[j, "split_ad"] = "train"          # test file placed right next to training: gap violated
    assert validate_manifest(bad, cfg)

    bad = manifest.copy()
    bad.loc[bad.index[bad.group == "fan_id_00"][0], "split_cls"] = "test"
    assert any("classifier" in p for p in validate_manifest(bad, cfg))

    bad = manifest.copy()
    bad["sha256"] = ""
    a = bad.index[(bad.split_ad == "train")][0]
    b = bad.index[(bad.split_ad == "test")][0]
    bad.loc[[a, b], "sha256"] = "x" * 64
    assert any("identical audio" in p for p in validate_manifest(bad, cfg))


def test_audit_flags_corruption_but_only_warns_for_loud(tmp_path, cfg):
    rng = np.random.default_rng(5)
    base = tmp_path / "fan/id_00/normal"
    ok = (rng.standard_normal(16000) * 0.05).astype("float32")
    write_wav(base / "0.wav", ok)
    write_wav(base / "1.wav", ok)                                         # exact duplicate of 0
    write_wav(base / "2.wav", np.zeros(16000, "float32"))                 # silent
    write_wav(base / "3.wav", (rng.standard_normal(5000) * .05).astype("float32"))   # wrong length
    write_wav(base / "4.wav", np.zeros((16000, 2), "float32") + .1)      # stereo
    write_wav(base / "5.wav", np.clip(rng.standard_normal(16000) * 2, -1, 1).astype("float32"))  # loud / clipped
    df, _ = scan_folder(tmp_path, cfg)
    res = audit.decide_exclusions(pd.concat([df, pd.DataFrame([audit.audit_file(tmp_path / p, cfg) for p in df.path])], axis=1), cfg)
    reason = dict(zip(res.path.str.split("/").str[-1], res.exclude_reason))
    assert reason["0.wav"] == "" and reason["1.wav"].startswith("duplicate_of")
    assert reason["2.wav"] == "silent" and reason["3.wav"] == "wrong_duration" and reason["4.wav"] == "not_mono"
    assert reason["5.wav"] == ""                                          # kept
    assert bool(res.set_index(res.path.str.split("/").str[-1]).loc["5.wav", "warn_clipping"])


def test_interleaved_properties(manifest, cfg):
    b = cfg["split"]["anomaly"]["buffer_files"]
    assert cfg["split"]["anomaly"]["strategy"] == "interleaved"
    for _, g in manifest.groupby("group"):
        assert not ((g.split_ad == "train") & (g.label == 1)).any()
        for s in ("val", "test"):
            x = g[g.split_ad == s]
            assert (x.label == 0).sum() == (x.label == 1).sum() > 0
            for lab in (0, 1):            # each set covers a wide part of the recording, not one block
                pos = x[x.label == lab].index_in_label / g[g.label == lab].index_in_label.max()
                assert pos.max() - pos.min() > 0.25
        tr = g[g.split_ad == "train"].index_in_label / g[g.label == 0].index_in_label.max()
        assert tr.min() < 0.1 and tr.max() > 0.9


@pytest.mark.parametrize("n_normal,n_abn", [(1011, 407), (1016, 359), (1033, 348), (1015, 361),
                                            (1006, 143), (1005, 111), (702, 100), (1036, 102)])
def test_interleaved_on_real_mimii_counts(n_normal, n_abn):
    from smart_listener.config import load_config
    cfg = load_config()
    rows = []
    for label, n in ((0, n_normal), (1, n_abn)):
        for k in range(n):
            rows.append({"file_id": f"x_{label}_{k}", "path": f"fan/id_00/{'ab' if label else ''}normal/{k:08d}.wav",
                         "machine_type": "fan", "machine_id": "id_00", "group": "fan_id_00",
                         "label_name": "abnormal" if label else "normal", "label": label, "index_in_label": k})
    df = assign_anomaly_splits(pd.DataFrame(rows), cfg)
    df["split_cls"] = "train"
    assert [p for p in validate_manifest(df, cfg) if "classifier" not in p] == []
    assert (df.split_ad == "train").sum() >= 300
