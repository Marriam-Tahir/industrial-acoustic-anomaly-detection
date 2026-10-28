"""Shared helpers: a tiny synthetic MIMII-like folder for tests."""
import numpy as np
import pytest
import soundfile as sf

from smart_listener.config import load_config


def write_wav(path, y, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def make_tree(root, n_normal=200, n_abn=60, types=("fan", "pump"), ids=("id_00", "id_02", "id_04", "id_06"), seconds=1.0):
    """Short clips (1 s) so tests are fast. Each file gets unique noise so no accidental duplicates."""
    rng = np.random.default_rng(0)
    for t in types:
        for i in ids:
            for label, n in (("normal", n_normal), ("abnormal", n_abn)):
                for k in range(n):
                    y = rng.standard_normal(int(16000 * seconds)) * 0.05
                    write_wav(root / t / i / label / f"{k:08d}.wav", y.astype("float32"))
    return root


@pytest.fixture
def cfg():
    c = load_config()
    c["audio"]["expected_duration_s"] = 1.0
    c["split"]["anomaly"]["min_train_files"] = 5
    c["split"]["anomaly"]["buffer_files"] = 2
    c["quality"]["drift_min_files"] = 10
    return c
