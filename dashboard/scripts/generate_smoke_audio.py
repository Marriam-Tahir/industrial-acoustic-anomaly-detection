"""Create an explicitly synthetic WAV for software demonstrations, not accuracy tests."""
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "test-results" / "synthetic_signal.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(160000) / 16000
    rng = np.random.default_rng(42)
    signal = 0.15 * (0.7 * np.sin(2 * np.pi * 173 * t) + 0.15 * np.sin(2 * np.pi * 901 * t) + 0.1 * rng.standard_normal(len(t)))
    wavfile.write(target, 16000, (signal * 32767).astype(np.int16))
    print(f"Synthetic software-check audio: {target}")
    print("This is generated test audio, not a real fan/pump recording or a labeled normal/abnormal example.")
