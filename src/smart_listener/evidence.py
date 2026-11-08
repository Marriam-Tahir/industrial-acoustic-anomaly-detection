"""Phase 5: visual evidence for an analysed clip.

What the pictures show: the waveform, the log-mel spectrogram, and a "residual heatmap" - where in time and
frequency the model could NOT reproduce the sound it was shown. A bright patch means "this part of the sound looks
least like this machine's normal sound". It points at where the model is surprised. It does NOT name a fault, and a
bright patch is not proof of a mechanical problem.

Matplotlib is used with the non-interactive Agg backend, so this works on a server with no screen.
"""
import io
import warnings
from typing import Optional

import numpy as np

from smart_listener import audio

STATE_COLORS = {"NORMAL": "#2a7f62", "WATCH": "#8a8a1f", "WARNING": "#d9822b", "HIGH": "#c0392b"}
CLIP_COLORS = {"within_normal": "#2a7f62", "above_warning_limit": "#d9822b", "above_alarm_limit": "#c0392b"}
CLIP_TEXT = {"within_normal": "within this machine's normal range",
             "above_warning_limit": "above the warning limit",
             "above_alarm_limit": "above the alarm limit"}


# ----------------------------------------------------------------------------- the residual map
def residual_map(model, kind: str, x_norm: np.ndarray, cfg: dict, device="cpu") -> np.ndarray:
    """Squared reconstruction error placed back on the spectrogram grid.

    x_norm : normalised log-mel spectrogram of ONE clip, shape (n_mels, frames).
    Returns an array of the same shape. Every cell is the average squared error of all model inputs that contain it
    (neighbouring inputs overlap, so cells near the edges are covered by fewer inputs). Cells that no model input
    covers are NaN (the conv model does not see the last ~25 frames, exactly as in its Phase 2 scoring)."""
    import torch
    from smart_listener.anomaly import CONTEXT, stack_frames
    n_mels, n_frames = x_norm.shape
    total = np.zeros((n_mels, n_frames), dtype=np.float64)
    count = np.zeros((n_mels, n_frames), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        if kind == "dense":
            X = torch.from_numpy(stack_frames(x_norm)).to(device)                       # (T', 5*64)
            err = ((model(X) - X) ** 2).cpu().numpy().reshape(-1, CONTEXT, n_mels)       # (T', 5, 64)
            for t in range(err.shape[0]):
                for c in range(CONTEXT):
                    total[:, t + c] += err[t, c]
                    count[:, t + c] += 1
        else:
            wf, hop = cfg["features"]["window_frames"], cfg["features"]["window_hop_frames"]
            W = audio.make_windows(x_norm, wf, hop)                                       # (k, 64, wf)
            X = torch.from_numpy(np.ascontiguousarray(W[:, None], dtype=np.float32)).to(device)
            err = ((model(X) - X) ** 2).cpu().numpy()[:, 0]
            for k in range(err.shape[0]):
                s = k * hop
                total[:, s:s + wf] += err[k]
                count[:, s:s + wf] += 1
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)   # NaN = not seen by the model (conv: the last frames)
    return out.astype(np.float32)


def summarize(resid: np.ndarray, cfg: dict, top_bands: int = 3) -> dict:
    """Short, numeric description of the residual map (this is what gets stored in the database)."""
    import librosa
    f, a = cfg["features"], cfg["audio"]
    n_mels, n_frames = resid.shape
    hz = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=f["fmin"], fmax=f["fmax"])
    low, high = hz[:-2], hz[2:]                                  # each band's lower / upper edge (triangular filter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns (frames the model did not see)
        per_band = np.nanmean(resid, axis=1)
        per_frame = np.nanmean(resid, axis=0)
    share = per_band / max(float(per_band.sum()), 1e-12)
    order = np.argsort(per_band)[::-1][:top_bands]
    frame_s = f["hop_length"] / a["sample_rate"]
    t_peak = int(np.nanargmax(per_frame))
    return {
        "mean_error": float(np.nanmean(resid)),
        "top_bands": [{"band": int(b), "hz_low": round(float(low[b])), "hz_high": round(float(high[b])),
                       "share_of_error": round(float(share[b]), 3)} for b in order],
        "peak_time_s": round(t_peak * frame_s, 2),
        "peak_time_error_ratio": round(float(per_frame[t_peak] / max(float(np.nanmean(per_frame)), 1e-12)), 2),
        "note": ("Where the model's reconstruction error is largest. It shows where the sound differs most from "
                 "this machine's normal sound; it is not a fault diagnosis."),
    }


def describe(summary: dict, result: dict) -> str:
    """One plain-language sentence for the dashboard / report / chatbot."""
    tb = summary["top_bands"][0]
    return (f"This 10-second clip scored z={result['z']:.1f} ({CLIP_TEXT[result['clip_level']]}). "
            f"The model had most trouble with {tb['hz_low']}-{tb['hz_high']} Hz "
            f"({tb['share_of_error'] * 100:.0f}% of the error) and around {summary['peak_time_s']} s. "
            f"This shows where the sound differs from normal; it does not identify a fault.")


def compute_evidence(analyzer, group: str, y: np.ndarray) -> dict:
    """Everything for one clip: spectrogram, residual map, score result and summary. One model pass for the score
    and one for the map (the clip is short, so this stays well under a second on a laptop)."""
    spec = analyzer.spectrogram(y)
    result = analyzer.result_from_score(group, analyzer.score_spectrogram(group, spec))
    model, kind, mean, std = analyzer.load(group)
    resid = residual_map(model, kind, audio.apply_norm(spec, mean, std), analyzer.cfg, analyzer.device)
    summ = summarize(resid, analyzer.cfg)
    return {"spec": spec, "resid": resid, "result": result, "summary": summ,
            "sentence": describe(summ, result.as_dict())}


# ----------------------------------------------------------------------------- pictures
def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def clip_figure(y: np.ndarray, ev: dict, cfg: dict, title: str = "") -> bytes:
    """Four panels: waveform, log-mel spectrogram, residual heatmap, and the score against the limits."""
    plt = _plt()
    sr, f = cfg["audio"]["sample_rate"], cfg["features"]
    spec, resid, res = ev["spec"], ev["resid"], ev["result"]
    t_end = len(y) / sr
    ext = [0, spec.shape[1] * f["hop_length"] / sr, f["fmin"] / 1000, f["fmax"] / 1000]
    fig = plt.figure(figsize=(9, 8.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 0.8], hspace=0.85, top=0.90)

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(len(y)) / sr, y, lw=0.4, color="#35566b")
    ax.set_xlim(0, t_end)
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(spec, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    ax.set_ylabel("kHz")
    ax.set_title("Log-mel spectrogram (dB)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(resid, origin="lower", aspect="auto", extent=ext, cmap="magma")
    ax.set_ylabel("kHz")
    ax.set_title("Where the model could not reproduce the sound (residual heatmap)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    ax.set_xlabel("seconds")

    ax = fig.add_subplot(gs[3])
    top = min(max(res.z, res.z_alarm) + 1.5, res.z_alarm * 3)      # far-off values are drawn at the edge with an arrow
    off_scale = res.z > top
    ax.axvspan(-3, res.z_warn, color="#2a7f62", alpha=0.10)
    ax.axvspan(res.z_warn, res.z_alarm, color="#d9822b", alpha=0.12)
    ax.axvspan(res.z_alarm, top, color="#c0392b", alpha=0.12)
    ax.axvline(res.z_warn, color="#d9822b", ls="--", lw=1.2)
    ax.axvline(res.z_alarm, color="#c0392b", ls="--", lw=1.2)
    ax.plot([min(res.z, top * 0.96)], [0.5], ">" if off_scale else "o", ms=11, color=CLIP_COLORS[res.clip_level], mec="black")
    ax.set_xlim(min(-3, res.z - 1), top)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"anomaly level z = {res.z:.1f}{' (off the scale)' if off_scale else ''}   (warning limit {res.z_warn:.1f}, alarm limit {res.z_alarm:.1f})")
    ax.set_title(f"This clip: {CLIP_TEXT[res.clip_level]}", loc="left", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left", y=0.985)
    return _fig_to_png(fig)


def history_figure(rows: list, z_warn: float, z_alarm: float, title: str = "") -> bytes:
    """z over time with the two limits and the severity state as a coloured band underneath.
    `rows` are analyses (dicts with window_utc, z, state), oldest first."""
    plt = _plt()
    from datetime import datetime
    fig, (ax, bar) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [5, 0.6], "hspace": 0.08})
    if not rows:
        ax.text(0.5, 0.5, "no analysed windows yet", ha="center", va="center", transform=ax.transAxes)
        return _fig_to_png(fig)
    t = [datetime.fromisoformat(r["window_utc"]) for r in rows]
    z = [r["z"] for r in rows]
    ax.plot(t, z, lw=1.1, color="#35566b")
    ax.axhline(z_warn, color="#d9822b", ls="--", lw=1.2, label=f"warning limit ({z_warn:.1f})")
    ax.axhline(z_alarm, color="#c0392b", ls="--", lw=1.2, label=f"alarm limit ({z_alarm:.1f})")
    ax.set_ylabel("anomaly level z")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    for i, r in enumerate(rows):
        t1 = t[i + 1] if i + 1 < len(t) else t[i]
        if i + 1 == len(t) and len(t) > 1:
            t1 = t[i] + (t[i] - t[i - 1])
        bar.axvspan(t[i], t1, color=STATE_COLORS.get(r.get("state") or "NORMAL", "#999999"), lw=0)
    bar.set_yticks([])
    bar.set_xlabel("time (simulated clock for simulated monitoring)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS.values()]
    bar.legend(handles, list(STATE_COLORS), loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=8, frameon=False)
    fig.autofmt_xdate()
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return _fig_to_png(fig)
