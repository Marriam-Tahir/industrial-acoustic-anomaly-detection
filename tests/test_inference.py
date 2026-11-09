"""Phase 5 tests: upload preparation, scoring (must equal the Phase 2 scoring), residual map and pictures."""
import numpy as np
import pytest
import soundfile as sf

from phase5_fixtures import SR, analyzer, synth, synth_clip  # noqa: F401  (fixtures)
from smart_listener import anomaly, audio, evidence, inference
from smart_listener.inference import AudioRejected, check_hint, filename_hint, prepare_upload

RNG = np.random.default_rng(1)


def wav(tmp_path, seconds=10.0, sr=SR, channels=1, amp=0.1, name="x.wav", freq=150.0):
    t = np.arange(int(seconds * sr)) / sr
    y = (amp * np.sin(2 * np.pi * freq * t) + 0.01 * RNG.standard_normal(len(t))).astype(np.float32)
    if channels > 1:
        y = np.stack([y] * channels, axis=1)
    p = tmp_path / name
    sf.write(p, y, sr, subtype="PCM_16")
    return p


# ----------------------------------------------------------------------------- file names
@pytest.mark.parametrize("name,expect", [
    ("fan_id_06_00000123.wav", ("fan", "id_06")), ("C:/x/Pump-04.wav", ("pump", "id_04")),
    ("PUMP id 00 test.WAV", ("pump", "id_00")), ("recording1.wav", None), ("fan_0123.wav", None)])
def test_filename_hint(name, expect):
    h = filename_hint(name)
    assert (h and (h["machine_type"], h["machine_id"])) == expect or (h is None and expect is None)


def test_hint_warning_only_when_different():
    assert check_hint({"machine_type": "fan", "machine_id": "id_06"}, "fan_id_06") is None
    assert check_hint(None, "fan_id_06") is None
    assert "pump id_04" in check_hint({"machine_type": "pump", "machine_id": "id_04"}, "fan_id_06")


# ----------------------------------------------------------------------------- preparing uploads
def test_exact_clip_gives_one_window(tmp_path, synth):
    p = prepare_upload(wav(tmp_path), synth["cfg"])
    assert len(p.windows) == 1 and len(p.windows[0]) == audio.expected_samples(synth["cfg"]) and p.warnings == []


def test_long_file_is_cut_into_consecutive_windows(tmp_path, synth):
    p = prepare_upload(wav(tmp_path, 30), synth["cfg"])
    assert len(p.windows) == 3 and p.offsets_s == [0.0, 10.0, 20.0]


def test_short_tail_dropped_and_nearly_full_tail_padded(tmp_path, synth):
    p = prepare_upload(wav(tmp_path, 25), synth["cfg"])
    assert len(p.windows) == 2 and any("too short" in w for w in p.warnings)
    p = prepare_upload(wav(tmp_path, 29.8), synth["cfg"])
    assert len(p.windows) == 3 and any("padded" in w for w in p.warnings)
    assert len(p.windows[-1]) == audio.expected_samples(synth["cfg"])


def test_rejections(tmp_path, synth):
    cfg = synth["cfg"]
    with pytest.raises(AudioRejected, match="only 5.0 s"):
        prepare_upload(wav(tmp_path, 5), cfg)
    with pytest.raises(AudioRejected, match="silent"):
        prepare_upload(wav(tmp_path, 10, amp=0.0, name="s.wav").with_name("s.wav"), cfg) if False else \
            prepare_upload(_silent(tmp_path), cfg)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not audio at all")
    with pytest.raises(AudioRejected, match="could not be read"):
        prepare_upload(junk, cfg)
    with pytest.raises(AudioRejected, match="limit"):
        prepare_upload(wav(tmp_path, 30, name="l.wav"), cfg, max_seconds=20)


def _silent(tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(p, np.zeros(SR * 10, dtype=np.float32), SR, subtype="PCM_16")
    return p


def test_stereo_uses_channel_one_and_resampling_is_reported(tmp_path, synth):
    p = prepare_upload(wav(tmp_path, 10, channels=2, name="st.wav"), synth["cfg"])
    assert len(p.windows) == 1 and any("channels" in w for w in p.warnings)
    p = prepare_upload(wav(tmp_path, 10, sr=8000, name="lo.wav"), synth["cfg"])
    assert len(p.windows) == 1 and any("8000 Hz" in w for w in p.warnings)


def test_clipping_is_warned_about(tmp_path, synth):
    y = np.clip(3 * np.sin(2 * np.pi * 100 * np.arange(SR * 10) / SR), -1, 1).astype(np.float32)
    p = tmp_path / "clip.wav"
    sf.write(p, y, SR, subtype="PCM_16")
    assert any("clipped" in w for w in prepare_upload(p, synth["cfg"]).warnings)


# ----------------------------------------------------------------------------- scoring
def test_upload_score_equals_phase2_score(synth, analyzer):
    """The same clip must get the same score from the app and from the Phase 2 evaluation code."""
    cfg, g = synth["cfg"], "fan_id_06"
    data = anomaly.load_group(cfg, g)
    model, _ = anomaly.load_checkpoint(analyzer.checkpoint_path(g), cfg)
    idx = anomaly.select_rows(data["rows"], "test")[:6]
    expected = anomaly.score_rows(model, data, idx, "dense", cfg, "cpu")[:, 0]
    for i, e in zip(idx, expected):
        fid = data["rows"].loc[i, "file_id"]
        path = synth["root"] / "mono" / synth["manifest"].set_index("file_id").loc[fid, "path"]
        y, _ = audio.fix_length(audio.load_mono_wav(path, cfg), audio.expected_samples(cfg))
        assert analyzer.analyze_waveform(g, y).score == pytest.approx(float(e), rel=1e-4)


def test_normal_and_abnormal_levels(analyzer):
    for g in ("fan_id_06", "pump_id_04"):
        kind = g.split("_")[0]
        normal = [analyzer.analyze_waveform(g, synth_clip(kind, False, RNG)) for _ in range(4)]
        bad = [analyzer.analyze_waveform(g, synth_clip(kind, True, RNG)) for _ in range(4)]
        assert max(r.z for r in normal) < analyzer.thresholds["z_warn"]
        assert all(r.clip_level == "above_alarm_limit" for r in bad)


def test_clip_level_boundaries():
    assert inference.clip_level(1.0, 3, 6) == "within_normal"
    assert inference.clip_level(3.0, 3, 6) == "within_normal"
    assert inference.clip_level(3.01, 3, 6) == "above_warning_limit"
    assert inference.clip_level(6.5, 3, 6) == "above_alarm_limit"


def test_availability_and_wrong_window_length(synth, tmp_path):
    from smart_listener.inference import Analyzer
    a = Analyzer(synth["cfg"], tmp_path / "nothing_here", synth["calibration"])
    ok, why = a.available("fan_id_06")
    assert not ok and "missing" in why
    with pytest.raises(FileNotFoundError):
        a.load("fan_id_06")
    assert a.available("pump_id_00")[0] is False
    b = Analyzer(synth["cfg"], synth["models_dir"], synth["calibration"])
    with pytest.raises(ValueError, match="exactly"):
        b.analyze_waveform("fan_id_06", np.zeros(1000, dtype=np.float32))


def test_checkpoint_for_another_group_is_refused(synth, tmp_path):
    import shutil
    from smart_listener.inference import Analyzer
    d = tmp_path / "dense"
    d.mkdir()
    shutil.copy(synth["models_dir"] / "dense" / "pump_id_04.pt", d / "fan_id_06.pt")
    a = Analyzer(synth["cfg"], tmp_path, synth["calibration"])
    with pytest.raises(ValueError, match="trained for pump_id_04"):
        a.load("fan_id_06")


# ----------------------------------------------------------------------------- evidence
def test_residual_map_points_at_the_odd_frequency(synth, analyzer):
    y = synth_clip("fan", True, RNG)                     # the abnormal test clip adds a 1.8 kHz tone
    ev = evidence.compute_evidence(analyzer, "fan_id_06", y)
    assert ev["resid"].shape == (64, audio.expected_frames(synth["cfg"])) and (ev["resid"] >= 0).all()
    top = ev["summary"]["top_bands"]
    assert any(b["hz_low"] <= 1800 <= b["hz_high"] for b in top), top
    assert "not identify a fault" in ev["sentence"]
    assert ev["result"].clip_level == "above_alarm_limit"


def test_residual_map_is_consistent_with_the_score_for_dense(analyzer, synth):
    """Dense inputs overlap, so the map is a re-arrangement of the same errors: its mean is close to the score."""
    y = synth_clip("pump", False, RNG)
    ev = evidence.compute_evidence(analyzer, "pump_id_04", y)
    assert ev["summary"]["mean_error"] == pytest.approx(ev["result"].score, rel=0.25)


def test_residual_map_for_conv_model_has_right_shape(synth):
    import torch
    cfg = synth["cfg"]
    model = anomaly.make_model("conv", cfg).eval()
    x = RNG.standard_normal((64, audio.expected_frames(cfg))).astype(np.float32)
    r = evidence.residual_map(model, "conv", x, cfg)
    assert r.shape == x.shape
    covered = ~np.isnan(r)
    assert covered[:, :288].all() and not covered[:, 288:].any()   # windows start at 0,32,...,224 and span 64 frames
    assert r.shape == x.shape and (r[covered] > 0).all()
    assert evidence.summarize(r, cfg)["mean_error"] > 0
