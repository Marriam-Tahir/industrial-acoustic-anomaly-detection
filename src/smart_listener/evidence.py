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

STATE_COLORS = {"NORMAL": "#3fb27f", "WATCH": "#c9a43a", "WARNING": "#f0a31c", "HIGH": "#e5534b"}
CLIP_COLORS = {"within_normal": "#3fb27f", "above_warning_limit": "#f0a31c", "above_alarm_limit": "#e5534b"}
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
    if summary.get("peak_time_error_ratio", 1.0) >= 2.0:
        when = f"strongest around {summary['peak_time_s']} s"
    else:
        when = "spread fairly evenly through the clip (a steady difference, not a short event)"
    return (f"This 10-second clip scored z={result['z']:.1f} ({CLIP_TEXT[result['clip_level']]}). "
            f"The model had most trouble with {tb['hz_low']}-{tb['hz_high']} Hz "
            f"({tb['share_of_error'] * 100:.0f}% of the error), {when}. "
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
# Two looks: "dark" (the control-room dashboard, the default) and "light" (for printing, if a report wants it).
PALETTES = {
    "dark": {"bg": "#0e151d", "panel": "#111a24", "text": "#e8edf3", "muted": "#8fa0b3", "grid": "#243242",
             "line": "#6fa8d6", "warn": "#f0a31c", "alarm": "#e5534b", "spec": "viridis", "resid": "inferno"},
    "light": {"bg": "#ffffff", "panel": "#ffffff", "text": "#1c2733", "muted": "#5d6b7a", "grid": "#d9e0e7",
              "line": "#2f6a9a", "warn": "#d98a00", "alarm": "#c0392b", "spec": "viridis", "resid": "magma"},
}
FONT = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _style(theme: str) -> dict:
    p = PALETTES.get(theme, PALETTES["dark"])
    return {"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 9, "figure.facecolor": p["bg"],
            "axes.facecolor": p["panel"], "axes.edgecolor": p["grid"], "axes.labelcolor": p["muted"],
            "xtick.color": p["muted"], "ytick.color": p["muted"], "text.color": p["text"], "axes.titlecolor": p["muted"],
            "axes.spines.top": False, "axes.spines.right": False, "grid.color": p["grid"], "grid.linewidth": 0.6}


def _panel_title(ax, text: str, p: dict) -> None:
    ax.set_title(text.upper(), loc="left", fontsize=8, color=p["muted"], fontweight="bold", pad=6)


def clip_figure(y: np.ndarray, ev: dict, cfg: dict, title: str = "", theme: str = "dark") -> bytes:
    """Four panels: waveform, log-mel spectrogram, residual heatmap, and the score against the limits."""
    plt = _plt()
    p = PALETTES.get(theme, PALETTES["dark"])
    sr, f = cfg["audio"]["sample_rate"], cfg["features"]
    spec, resid, res = ev["spec"], ev["resid"], ev["result"]
    t_end = len(y) / sr
    ext = [0, spec.shape[1] * f["hop_length"] / sr, f["fmin"] / 1000, f["fmax"] / 1000]
    with plt.rc_context(_style(theme)):
        fig = plt.figure(figsize=(10, 8.4))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 0.75], hspace=0.62, top=0.93)

        ax = fig.add_subplot(gs[0])
        ax.plot(np.arange(len(y)) / sr, y, lw=0.4, color=p["line"])
        ax.set_xlim(0, t_end)
        ax.set_ylabel("amplitude")
        ax.grid(axis="y", alpha=0.5)
        _panel_title(ax, "Waveform", p)

        ax = fig.add_subplot(gs[1])
        im = ax.imshow(spec, origin="lower", aspect="auto", extent=ext, cmap=p["spec"])
        ax.set_ylabel("kHz")
        _panel_title(ax, "Log-mel spectrogram (dB)", p)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.outline.set_edgecolor(p["grid"])

        ax = fig.add_subplot(gs[2])
        im = ax.imshow(resid, origin="lower", aspect="auto", extent=ext, cmap=p["resid"])
        ax.set_ylabel("kHz")
        ax.set_xlabel("seconds")
        _panel_title(ax, "Residual heatmap - where the sound differs most from normal", p)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.outline.set_edgecolor(p["grid"])

        ax = fig.add_subplot(gs[3])
        top = min(max(res.z, res.z_alarm) + 1.5, res.z_alarm * 3)      # far-off values are drawn at the edge with an arrow
        off_scale = res.z > top
        lo = min(-3, res.z - 1)
        ax.axvspan(lo, res.z_warn, color=CLIP_COLORS["within_normal"], alpha=0.16, lw=0)
        ax.axvspan(res.z_warn, res.z_alarm, color=CLIP_COLORS["above_warning_limit"], alpha=0.18, lw=0)
        ax.axvspan(res.z_alarm, top, color=CLIP_COLORS["above_alarm_limit"], alpha=0.18, lw=0)
        ax.axvline(res.z_warn, color=p["warn"], ls="--", lw=1.2)
        ax.axvline(res.z_alarm, color=p["alarm"], ls="--", lw=1.2)
        ax.plot([min(res.z, top * 0.96)], [0.5], ">" if off_scale else "o", ms=10, color=CLIP_COLORS[res.clip_level],
                mec=p["text"], mew=1.0)
        ax.set_xlim(lo, top)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"anomaly level z = {res.z:.1f}{' (off the scale)' if off_scale else ''}      "
                      f"warning limit {res.z_warn:.1f}   |   alarm limit {res.z_alarm:.1f}")
        _panel_title(ax, f"This clip: {CLIP_TEXT[res.clip_level]}", p)
        if title:
            fig.suptitle(title, fontsize=11, x=0.01, ha="left", y=0.985, color=p["text"], fontweight="bold")
        return _fig_to_png(fig)


def history_figure(rows: list, z_warn: float, z_alarm: float, title: str = "", theme: str = "dark") -> bytes:
    """z over time with the two limits, a dot per window coloured by the severity state, and the state as a band.
    `rows` are analyses (dicts with window_utc, z, state), oldest first. Very large z values are compressed
    (log scale above a few times the alarm limit) so the limits stay readable."""
    plt = _plt()
    from datetime import datetime
    p = PALETTES.get(theme, PALETTES["dark"])
    with plt.rc_context(_style(theme)):
        fig, (ax, bar) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [6, 0.5], "hspace": 0.06})
        if not rows:
            ax.text(0.5, 0.5, "No analysed windows yet", ha="center", va="center", transform=ax.transAxes, color=p["muted"])
            ax.set_xticks([])
            ax.set_yticks([])
            bar.axis("off")
            return _fig_to_png(fig)
        t = [datetime.fromisoformat(r["window_utc"]) for r in rows]
        z = [r["z"] for r in rows]
        compress = max(z) > 4 * z_alarm and z_alarm > 0
        if compress:
            ax.set_yscale("symlog", linthresh=max(z_alarm * 1.5, 1.0))
        ax.plot(t, z, lw=1.2, color=p["line"], zorder=2)
        ax.scatter(t, z, s=16, c=[STATE_COLORS.get(r.get("state") or "NORMAL", "#6b7785") for r in rows], zorder=3, lw=0)
        ax.axhline(z_warn, color=p["warn"], ls="--", lw=1.1, zorder=1)
        ax.axhline(z_alarm, color=p["alarm"], ls="--", lw=1.1, zorder=1)
        for zv, name, col in ((z_warn, "warning", p["warn"]), (z_alarm, "alarm", p["alarm"])):
            ax.text(0.995, zv, f"{name} {zv:.1f}", color=col, va="bottom", ha="right", fontsize=8,
                    transform=ax.get_yaxis_transform(), zorder=4)
        ax.set_ylabel("anomaly level z" + (" (compressed above limits)" if compress else ""))
        ax.grid(alpha=0.6)
        for i in range(len(rows)):
            t1 = t[i + 1] if i + 1 < len(t) else t[i]
            if i + 1 == len(t) and len(t) > 1:
                t1 = t[i] + (t[i] - t[i - 1])
            bar.axvspan(t[i], t1, color=STATE_COLORS.get(rows[i].get("state") or "NORMAL", "#6b7785"), lw=0)
        bar.set_yticks([])
        for sp in ("left", "bottom"):
            bar.spines[sp].set_visible(False)
        bar.set_xlabel("time (UTC; simulated clock for simulated monitoring)")
        present = [s for s in STATE_COLORS if any((r.get("state") or "NORMAL") == s for r in rows)]
        ax.legend([plt.Line2D([0], [0], marker="o", ls="", color=STATE_COLORS[s]) for s in present], present,
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(present), fontsize=8, frameon=False, labelcolor=p["muted"])
        fig.autofmt_xdate()
        if title:
            ax.set_title(title, loc="left", fontsize=10, color=p["text"], fontweight="bold")
        return _fig_to_png(fig)
