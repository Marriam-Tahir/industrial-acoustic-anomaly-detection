"""Phase 8: robustness scoring of noisier MIMII releases (read straight from a zip, test clips only)."""
import io
import zipfile

import numpy as np
import soundfile as sf

from phase5_fixtures import SR, analyzer, synth  # noqa: F401
from smart_listener import audio, robustness
from smart_listener.config import resolve_path


def _noisy_zip(synth, path, noise=0.05, drop=None):
    """A zip laid out like MIMII (top folder, 2 channels) with the synthetic clips plus extra background noise."""
    rng = np.random.default_rng(1)
    mono = resolve_path(synth["cfg"], "mono_dir")
    with zipfile.ZipFile(path, "w") as z:
        for r in synth["manifest"].to_dict("records"):
            if drop and r["path"].startswith(drop):
                continue
            y = audio.load_mono_wav(mono / r["path"], synth["cfg"])
            y = y + rng.normal(0, noise, len(y)).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, np.stack([y, y * 0.5], 1), SR, format="WAV")
            z.writestr(f"0_dB_fan_pump/{r['path']}", buf.getvalue())


def test_names_and_counts():
    assert robustness.snr_from_name("-6_dB_fan.zip") == -6 and robustness.snr_from_name("0_dB_pump.zip") == 0
    assert robustness.rel_key("x/fan/id_06/normal/00000001.wav") == "fan/id_06/normal/00000001.wav"
    assert robustness.rel_key("readme.txt") is None


def test_only_test_clips_are_read_from_the_zip_and_scored(synth, analyzer, tmp_path):
    z = tmp_path / "0_dB_test.zip"
    _noisy_zip(synth, z)
    man = synth["manifest"]
    clips = robustness.test_clips(man)
    assert set(clips["split_ad"]) == {"test"} and len(clips) < len(man)
    wanted = {r["key"]: r for r in clips.to_dict("records")}
    assert robustness.folder_counts(robustness.source_names(z)) == robustness.manifest_counts(man)
    df = robustness.score_rows(analyzer, robustness.iter_source(z, wanted), synth["cfg"], 0)
    assert len(df) == len(clips) and set(df["file_id"]) == set(clips["file_id"]) and set(df["snr_db"]) == {0}

    mono = resolve_path(synth["cfg"], "mono_dir")
    n = audio.expected_samples(synth["cfg"])
    base = robustness.score_rows(analyzer, ((r, audio.fix_length(audio.load_mono_wav(mono / r["path"], synth["cfg"]), n)[0])
                                            for r in clips.to_dict("records")), synth["cfg"], 6)
    t = synth["calibration"]["thresholds"]
    m = robustness.summarize(__import__("pandas").concat([base, df]), t["z_warn"], t["z_alarm"])
    assert set(m["snr_db"]) == {6, 0} and len(m) == 2 * len(synth["groups"])
    assert (m[m["snr_db"] == 6]["auc_change_vs_6db"] == 0).all() and m["auc"].between(0, 1).all()
    noisy = m[m["snr_db"] == 0].set_index("group")
    clean = m[m["snr_db"] == 6].set_index("group")
    assert (noisy["normal_median_z"] > clean["normal_median_z"]).all()      # extra noise looks less normal


def test_a_release_with_different_files_is_detected(synth, tmp_path):
    z = tmp_path / "-6_dB_x.zip"
    _noisy_zip(synth, z, drop="fan/id_06/abnormal")
    have = robustness.folder_counts(robustness.source_names(z))
    want = robustness.manifest_counts(synth["manifest"])
    assert have != want and ("fan", "id_06", "abnormal") not in have
