"""Pack everything Colab needs for Phase 2 into ONE zip (about 1 GB), so you upload a single file to Google Drive.

    python scripts/make_colab_bundle.py

Creates phase2_bundle.zip in the project folder. It contains the config, the small Python package, the two Phase 2
scripts, the manifest, and the Phase 1 feature files and normalisation statistics. It does NOT contain any WAV audio.
"""
import sys
import zipfile

from smart_listener.config import project_root


def main() -> int:
    root = project_root()
    items = [root / "config" / "audio_config.yaml", root / "scripts" / "train_ad.py", root / "scripts" / "evaluate_ad.py",
             root / "data" / "manifests" / "manifest.csv", root / "data" / "processed" / "features_meta.json"]
    items += sorted((root / "src" / "smart_listener").glob("*.py"))
    feat = root / "data" / "processed"
    items += sorted((feat / "logmel").glob("*.npy")) + sorted((feat / "logmel").glob("*_files.csv"))
    items += sorted((feat / "norm").glob("*_anomaly.json"))
    missing = [p for p in items if not p.exists()]
    if missing:
        print("Missing files:", *missing, sep="\n  ")
        return 1
    if len(list((feat / "logmel").glob("*.npy"))) != 8:
        print("Expected 8 feature files (4 fan + 4 pump). Run scripts/extract_features.py first.")
        return 1
    out = root / "phase2_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in items:
            z.write(p, p.relative_to(root).as_posix())
            print("  added", p.relative_to(root).as_posix())
    print(f"\nCreated {out} ({out.stat().st_size / 1e9:.2f} GB). Upload it to Google Drive: My Drive/smart_listener/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
