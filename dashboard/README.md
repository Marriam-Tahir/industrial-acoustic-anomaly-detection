# Smart Listener

A local, working audio anomaly-detection application using the supplied `convae_v1` export. The main dashboard is served by Flask; [app.py](./app.py) is an optional Streamlit client that calls the same API. Training and inference share the unchanged exported preprocessing module.

## Quick start from GitHub

Use Python 3.11 or later. Install this application in its own environment; it does not require installing the separate training package at the repository root.

```powershell
git clone --single-branch --branch tahira https://github.com/Marriam-Tahir/industrial-acoustic-anomaly-detection.git
cd industrial-acoustic-anomaly-detection\dashboard
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The six inference models, matching preprocessing code/configuration, and saved evaluation reports are included. Training-resume checkpoints, original recordings, local environments, and generated test outputs are excluded. To use the provided VS Code tasks, open this `dashboard` folder as the workspace.

## Start the application

Open PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe main.py
```

Open **http://127.0.0.1:5000**. Keep the terminal running. Alternatively run `powershell -ExecutionPolicy Bypass -File .\start.ps1` or select the Flask dashboard task in VS Code. Stop the server with Ctrl+C.

For the optional Streamlit dashboard, keep Flask running and open a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Then open **http://127.0.0.1:8502**. Both dashboards use the same Flask inference service. `SMART_LISTENER_API_URL` can point the Streamlit client to another instance.

## Use the dashboard

1. Upload a WAV containing audible machine sound. Inputs may contain 1–60 seconds, 4–192 kHz sampling, and up to eight channels. Maximum file size: 25 MiB.
2. Select **Fan** or **Pump** and the recording's known signal-to-noise condition: +6, 0, or −6 dB. These select the matching exported detector. Noise condition is not estimated from sound.
3. **Auto from filename** recognizes standalone `fan` or `pump` tokens, for example `fan_id_06_00000001.wav`. Ordinary MIMII filenames such as `00000001.wav` require a manual machine selection. The filename's normal/abnormal label never determines the decision.
4. Select **Analyze recording**. The dashboard displays the machine source, Normal/Abnormal decision, raw score and saved threshold, nineteen window scores, and a mel spectrogram.
5. Download the JSON report. The model library displays the actual saved test metrics. History contains only result metadata in your browser; uploaded audio is not retained in history.

Only the first ten seconds contribute windows; shorter recordings are padded with zeros by the exported code. The original code peak-normalizes before trimming, so normalization considers the complete input recording. Resampling, downmixing, short clips, trimming, and possible clipping produce visible messages. Silent, nearly silent, constant, or nonfinite signals are rejected.

## Machine identification and detection limits

**This export has no trained machine classifier.** Machine type is selected by the operator or read from the filename; the application labels this source. It does not infer a physical machine ID or fault type. A generic recording cannot be identified acoustically from these six autoencoders. Add a separately trained classifier with its own complete preprocessing and label mapping to support that capability.

The sibling project's older experimental classifier uses different features, and its model card says not to use it as an anomaly-model gate. It is deliberately not mixed into this application.

The provided test report also shows limited anomaly detection on the held-out machine ID: fan AUC is about 0.44–0.51 and pump AUC about 0.61–0.65. These are the supplied model's measured results, not application integration defects. The model-library screen exposes them, and results include a performance message. Normal means score ≤ threshold for the selected bundle; it does not prove a machine is healthy. The application demonstrates the real exported models without changing their thresholds or inventing confidence percentages.

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Main dashboard |
| `GET /api/health` | Readiness, available models, preprocessing checks, upload limit; 503 if degraded |
| `GET /api/models` | Model settings, saved test metrics, and preprocessing metadata |
| `POST /api/predict` | Multipart WAV inference |

Prediction fields are `file` (required WAV), `machine_type` (`fan`, `pump`, or `auto`, default `auto`), and `snr` (`6_dB`, `0_dB`, or `-6_dB`, default `6_dB`). Windows PowerShell example:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict `
  -F "file=@D:/path/to/recording.wav" `
  -F "machine_type=fan" `
  -F "snr=6_dB"
```

Success returns `request_id`, timestamp, filename, `machine_type`, `machine_identification`, `condition`, `is_anomaly`, `score`, `threshold`, `score_ratio`, `aggregation`, `model_id`, `snr`, `audio`, `window_scores`, `window_times`, `spectrogram`, `preprocessing`, `warnings`, and `elapsed_ms`. The score ratio is a score divided by threshold, not a probability. The response and `X-Request-ID` header share the same ID.

Errors use:

```json
{"error": {"code": "MACHINE_REQUIRED", "message": "Select the machine type or use a labeled filename."}, "request_id": "..."}
```

| HTTP status | Meaning |
| --- | --- |
| 400 | Missing, empty, malformed, or unreadable WAV |
| 413 | Oversized file or request |
| 415 | Unsupported file extension |
| 422 | Invalid selector, ambiguous machine, unsuitable duration/format, or unusable signal |
| 503 | Model not ready or invalid model output |
| 500 | Unexpected error; details stay in server logs |

## Preprocessing consistency

Source: [preprocessing_config.json](./models/convae_v1/export/preprocessing_config.json), fingerprint **55fac6205183**.

The engine verifies the exported preprocessing module's SHA-256 prefix, recomputes the parameter fingerprint, checks the implementation's frozen settings, and validates every model's fingerprint, shape, machine/noise metadata, aggregation, and threshold. Each model runs a forward pass before it is marked ready.

Inference calls `Preprocessor.preprocess_samples` and the unchanged `AnomalyScorer.score_array`: channel mean → resample to 16 kHz → peak normalization → trim/pad to 10 s → nineteen 1 s windows with 0.5 s hop → 64-band power mel → natural log with epsilon → per-window min/max scaling → float32 `(19, 64, 32)`. The original NumPy/SciPy implementation is used directly. Display spectrogram scaling does not feed back into model inputs.

Each model applies its exported aggregation (max or mean here) and **strictly `score > threshold`**. No threshold is fitted or adjusted in the app. Uploaded paths are never used to select files from the server.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest test_api.py test_inference.py test_ui.py -q
```

The tests cover the HTTP contract, real loading of all six bundles, direct parity with the exported scorer, padding, trimming, stereo, resampling, threshold equality, file validation, silent/nonfinite inputs, missing models, altered preprocessing artifacts, and Streamlit result/error/stale-result behavior. Synthetic signals and controlled UI replies test the software and are not evidence of machine classification accuracy. Browser verification is recorded in [TESTING.md](./TESTING.md).

To generate an explicitly synthetic file for trying the upload flow:

```powershell
.\.venv\Scripts\python.exe scripts/generate_smoke_audio.py
```

Choose the resulting `test-results/synthetic_signal.wav` and manually select a machine. This is a pipeline check, not a labeled MIMII recording. Use original WAV test recordings for your viva's normal/abnormal examples and discuss any mistakes according to the saved test metrics.

## Fresh installation

For another machine with Python 3.11+ and these exported models:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The dependencies are declared in [requirements.txt](./requirements.txt). The exported training versions are preserved in [the model metadata](./models/convae_v1/run_info.json); the app was verified with compatible installed versions for CPU inference. `SMART_LISTENER_EXPORT_DIR` can select another compatible complete export before startup. Restart the server after changing exports.

## Files and viva

- [api.py](./api.py): Flask routes, structured errors, dependency injection for tests.
- [main.py](./main.py): startup and WSGI app.
- [inference.py](./inference.py): export checks, input validation, preprocessing, real scoring, and response construction.
- [config.py](./config.py): application settings; feature settings stay in the export.
- [app.py](./app.py): optional Streamlit HTTP client.
- [templates/index.html](./templates/index.html), [static/app.js](./static/app.js), [static/styles.css](./static/styles.css): main browser UI.
- [test_api.py](./test_api.py), [test_inference.py](./test_inference.py), [test_ui.py](./test_ui.py): automated tests.
- [VIVA_GUIDE.md](./VIVA_GUIDE.md): demo script and explanation prompts.

The Flask development server is bound to loopback for a local demonstration. For network hosting, use a production WSGI server and an appropriate deployment configuration.
