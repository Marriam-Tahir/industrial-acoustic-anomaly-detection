"""Tests for the configuration loader. Run with:  pytest"""
import copy

import pytest

from smart_listener.config import load_config, validate_config


def test_default_config_loads_with_expected_values():
    cfg = load_config()
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["features"]["n_mels"] == 64
    assert cfg["features"]["n_fft"] == 1024
    assert cfg["features"]["hop_length"] == 512
    assert cfg["dataset"]["machine_ids"] == ["id_00", "id_02", "id_04", "id_06"]
    assert cfg["dataset"]["zips"]["fan"]["file"] == "6_dB_fan.zip"


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_bad_value_is_rejected():
    cfg = load_config()
    bad = copy.deepcopy(cfg)
    bad["features"]["n_mels"] = -5
    with pytest.raises(ValueError):
        validate_config(bad)


def test_hop_larger_than_fft_is_rejected():
    cfg = load_config()
    bad = copy.deepcopy(cfg)
    bad["features"]["hop_length"] = 4096
    with pytest.raises(ValueError):
        validate_config(bad)


def test_fingerprint_changes_only_with_audio_or_feature_settings():
    from smart_listener.config import config_fingerprint
    cfg = load_config()
    base = config_fingerprint(cfg)
    other = copy.deepcopy(cfg)
    other["split"]["seed"] = 7
    assert config_fingerprint(other) == base
    other["features"]["n_mels"] = 80
    assert config_fingerprint(other) != base


def test_overlapping_classifier_ids_rejected():
    cfg = copy.deepcopy(load_config())
    cfg["split"]["classifier"]["val_ids"] = ["id_00"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_unknown_split_strategy_rejected():
    cfg = copy.deepcopy(load_config())
    cfg["split"]["anomaly"]["strategy"] = "magic"
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_checksum_table_is_complete_and_consistent():
    cfg = load_config()
    sums = cfg["dataset"]["checksums"]
    assert len(sums) == 12 and all(len(v) == 32 for v in sums.values())
    for t in cfg["dataset"]["machine_types"]:                  # the two files we use must agree with the table
        z = cfg["dataset"]["zips"][t]
        assert sums[z["file"]] == z["md5"]
