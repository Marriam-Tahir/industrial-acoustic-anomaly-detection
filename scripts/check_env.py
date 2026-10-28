"""Check that the smartlistener environment works.  Run:  python scripts/check_env.py"""
import importlib
import importlib.metadata as md
import platform
import sys

# (name used in "import ...", name used by pip)
MODULES = [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
           ("sklearn", "scikit-learn"), ("soundfile", "soundfile"), ("librosa", "librosa"),
           ("yaml", "PyYAML"), ("requests", "requests"), ("remotezip", "remotezip"), ("pytest", "pytest")]


def main() -> int:
    problems = 0
    print("Python  :", sys.version.split()[0], "|", platform.platform())
    if sys.version_info < (3, 12):
        print("  WARNING: this project expects Python 3.12 or newer")
        problems += 1
    print("Packages:")
    for name, dist in MODULES:
        try:
            importlib.import_module(name)
            print(f"  {name:<12} {md.version(dist)}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            print(f"  {name:<12} MISSING ({exc})")
            problems += 1

    try:
        import numpy as np
        from smart_listener.audio import expected_frames, log_mel
        from smart_listener.config import config_fingerprint, load_config

        cfg = load_config()
        f, a = cfg["features"], cfg["audio"]
        y = np.random.default_rng(0).standard_normal(a["sample_rate"] * 10).astype("float32") * 0.1
        mel = log_mel(y, cfg)
        print("Config  : loaded OK, fingerprint", config_fingerprint(cfg))
        print("Mel test: 10 s of test noise ->", mel.shape, "(expected (64, 313))")
        if mel.shape != (f["n_mels"], expected_frames(cfg)) or not np.isfinite(mel).all():
            problems += 1
    except Exception as exc:  # noqa: BLE001
        print("Config/mel test FAILED:", repr(exc))
        problems += 1

    print("RESULT  :", "ALL GOOD" if problems == 0 else f"{problems} problem(s) found - see above")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
