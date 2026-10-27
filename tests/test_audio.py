import numpy as np
import pytest

from smart_listener import audio
from conftest import write_wav


def test_shape_and_determinism(cfg):
    cfg["audio"]["expected_duration_s"] = 10.0
    y = np.random.default_rng(1).standard_normal(160000).astype("float32") * 0.1
    a, b = audio.log_mel(y, cfg), audio.log_mel(y, cfg)
    assert a.shape == (64, 313) == (cfg["features"]["n_mels"], audio.expected_frames(cfg))
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_loudness_is_preserved(cfg):
    y = np.random.default_rng(2).standard_normal(16000).astype("float32") * 0.05
    shift = (audio.log_mel(y * 2, cfg) - audio.log_mel(y, cfg)).mean()
    assert shift == pytest.approx(6.02, abs=0.05)   # doubling amplitude = +6 dB, nothing hides it


def test_windows(cfg):
    spec = np.zeros((64, 313), dtype="float32")
    w = audio.make_windows(spec, 64, 32)
    assert w.shape == (8, 64, 64)
    with pytest.raises(ValueError):
        audio.make_windows(spec[:, :10], 64, 32)


def test_running_stats_equals_numpy():
    rng = np.random.default_rng(3)
    specs = [rng.standard_normal((8, 30)) * 3 + 5 for _ in range(4)]
    st = audio.RunningStats(8)
    for s in specs:
        st.update(s)
    mean, std = st.finalize()
    big = np.concatenate(specs, axis=1)
    assert np.allclose(mean, big.mean(axis=1)) and np.allclose(std, big.std(axis=1))


def test_norm_uses_given_stats_only():
    train = np.full((4, 10), 10.0) + np.arange(10)
    st = audio.RunningStats(4)
    st.update(train)
    mean, std = st.finalize()
    test = np.full((4, 10), 100.0)
    out = audio.apply_norm(test, mean, std)
    assert out.mean() > 5          # test data is NOT re-centred on itself


def test_loader_rejects_wrong_format(tmp_path, cfg):
    write_wav(tmp_path / "s.wav", np.zeros((1600, 2), dtype="float32"))
    with pytest.raises(ValueError):
        audio.load_mono_wav(tmp_path / "s.wav", cfg)
    write_wav(tmp_path / "r.wav", np.zeros(1600, dtype="float32"), sr=8000)
    with pytest.raises(ValueError):
        audio.load_mono_wav(tmp_path / "r.wav", cfg)


def test_fix_length():
    assert audio.fix_length(np.zeros(5), 5)[1] == "ok"
    assert audio.fix_length(np.zeros(3), 5) [0].shape == (5,)
    assert audio.fix_length(np.zeros(9), 5)[1] == "trimmed"
