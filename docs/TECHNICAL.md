# Smart Listener - technical notes

How the system works and why it was built this way. Measured numbers are in [RESULTS.md](RESULTS.md); they are
generated from the report files, so they are not repeated here.

## 1. Data

- **Source:** MIMII (Zenodo 3384388), fans and pumps, machine IDs 00/02/04/06, the **6 dB** recordings. Each clip
  is 10 s, 16 kHz, 8 microphones; channel 1 is used, as in the dataset's own baseline.
- **Audit:** files are checked for sample rate, duration (10 s within ±0.5 s), clipping, silence and loudness drift
  over the recording order. Settings are in `config/audio_config.yaml`.
- **Splits (one per machine, leakage-safe):**
  - Training uses normal clips only.
  - Validation and test take interleaved blocks spread across the whole recording.
  - A gap of 5 files separates blocks, so neighbouring recordings never fall in different splits.
  - Abnormal clips are divided between validation and test.
  - The manifest is `data/manifests/manifest.csv`.
- **Rule used everywhere:** test data is never used to choose a model, a limit, a rule or a setting. It is used once,
  to report results.

## 2. Features and models

- **Log-mel spectrogram:** 64 mel bands, n_fft 1024, hop 512, giving 313 frames per 10 s clip. Absolute dB values
  with no per-clip scaling, so loudness information is kept. Normalisation statistics come from training files only.
- **Two autoencoders per machine,** trained on normal sound only:
  - **dense:** the MIMII-paper baseline. 5 neighbouring frames (320 values) go through an 8-value bottleneck.
  - **conv:** a convolutional autoencoder on 64 × 64 windows (about 2 s).
- **Score:** the mean squared reconstruction error over the clip. Higher means less like this machine's normal
  sound. It is not a probability.
- **Model choice per machine:** made on validation ROC-AUC only. The dense model was chosen for every machine.
- **Loudness check:** `scripts/sanity_level_baseline.py` tests whether loudness alone separates normal from abnormal
  as well as the model. Where it does, the dashboard shows a caution for that machine.

## 3. From score to severity

1. **Calibration** (`scripts/calibrate_severity.py`): for each machine, the median and spread (1.4826 × MAD) of
   log(score) on normal validation clips give `z = (ln score − centre) / spread`. One machine has too few normal
   validation clips to read a rare percentile reliably, so the warning and alarm limits are the 95th and 99.5th
   percentiles of the **pooled** validation z-values of all 8 machines. Test normal clips are used only to check how
   often each machine crosses these limits.
2. **Severity rule** (`src/smart_listener/severity.py`): WATCH, WARNING and HIGH need several windows above a limit
   (persistence), and the level only goes down after several clean windows (hysteresis). The rule was tuned on
   validation streams (`scripts/evaluate_streams.py`): the fastest reaction with no false HIGH on normal validation
   streams.
3. **Incidents:** an incident opens when HIGH is reached. A HIGH that returns within the cooldown re-opens the same
   incident instead of creating a new one.

The data has no failure times, fault severities or maintenance records. That is why the system reports a sound
deviation level, not a failure probability, remaining life or fault diagnosis.

## 4. Application

| Part | File | Notes |
|---|---|---|
| Upload preparation | `inference.py` | Cut into 10 s windows. A final piece ≥ 9.5 s is padded, shorter ones are dropped. Resampling and channel selection produce warnings. |
| Scoring | `inference.py` | Reuses the training code (`audio.log_mel`, `anomaly.clip_inputs`); `scripts/verify_inference.py` proves the scores equal the evaluation scores. |
| Evidence | `evidence.py` | Waveform, log-mel, residual heatmap, top 3 frequency bands by share of error, peak time. |
| Simulated monitoring | `monitor.py` | Replays validation/test clips in order; scenarios normal / fault / fault then recovery; at most 4 sessions at once. |
| E-mail | `alerts.py`, `notifier.py` | Alert rows are written first; a background worker sends them (SMTP, or test mode into `data/app/outbox`). 3 attempts (waits of 1 and 4 min), hourly limit, reminders, one alert per incident and recipient. |
| Reports | `reports.py` | PDF made with matplotlib (no extra dependency), CSV. |
| Assistant | `assistant.py` | Keyword intents over the incident's records. Sections are labelled by source; physical fault, failure time and probability are always "not determined". |
| Identification | `identification.py` | Suggests a machine only if a validation run passes (98% correct on held-out test clips while answering a useful share). On the real models it did not, so it is off. |
| Database | `db.py`, `repo.py` | SQLite. Tables: admins, sessions, machines, monitoring_sessions, analyses, severity_events, incidents, incident_notes, alerts, settings, audit_log. |

### API routes (FastAPI, `src/smart_listener/api.py`; interactive list at http://127.0.0.1:8000/docs)

| Area | Routes |
|---|---|
| Health and login | `GET /health`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`, `GET /account`, `POST /auth/password` |
| Machines | `GET/POST /machines`, `GET/PUT/DELETE /machines/{id}`, `GET /model-profiles`, `GET /fleet` |
| Analysis | `POST /machines/{id}/analyze-upload`, `GET /machines/{id}/analyses`, `GET /analyses/{id}`, `GET /analyses/{id}/evidence.png`, `GET /machines/{id}/history.png`, `GET /machines/{id}/severity-events`, `GET /activity` |
| Monitoring | `GET /monitoring/scenarios`, `POST /machines/{id}/monitoring/start`, `GET /monitoring`, `GET /monitoring/{id}`, `POST /monitoring/{id}/stop` |
| Incidents | `GET /incidents`, `GET /incidents/{id}`, `POST /incidents/{id}/acknowledge`, `GET/POST /incidents/{id}/notes` |
| E-mail | `GET /alerts`, `POST /alerts/{id}/retry`, `GET /mail/status`, `POST /mail/test`, `POST /mail/process` |
| Reports | `GET /reports/incidents.csv`, `GET /reports/analyses.csv`, `GET /reports/incidents/{id}.pdf`, `GET /reports/machines/{id}.pdf` |
| Assistant | `GET /assistant/questions`, `POST /assistant/ask` |
| Identification | `GET /identification/status`, `POST /identify` |
| Administration | `GET/PUT /settings`, `GET /audit`, `GET /system` |

## 5. Machine identification (investigated, not used)

`scripts/investigate_identification.py` scores every validation and test clip with all 8 models.

- **Picking the lowest calibrated z** names the right machine much less often for abnormal clips than for normal
  ones. An abnormal recording is unfamiliar to its own model too, so another model often fits it better.
- **Answering only clear-cut cases** is accurate, but answers too few clips to be useful.
- **Unregistered machines** would often be accepted as one of the registered ones.

The application therefore uses manual selection, as an installed system would: each microphone is mounted on a
known machine. The numbers are in RESULTS.md, section 3.

## 6. Robustness to louder background noise (0 dB and −6 dB)

MIMII publishes the same machine recordings mixed with more factory noise at 0 dB and −6 dB signal-to-noise ratio
(SNR). The test checks how the 6 dB models and limits behave on those, **without retraining or re-tuning
anything**, using only the clips that are in the 6 dB **test** split (the file names are the same in every release).

```bash
python scripts/robustness_snr.py --baseline                          # the same test clips at 6 dB (from data/interim)
python scripts/robustness_snr.py --source Dataset/0_dB_fan.zip       # read straight from the zip, nothing unpacked
python scripts/robustness_snr.py --source Dataset/0_dB_pump.zip
python scripts/robustness_snr.py --source Dataset/-6_dB_fan.zip
python scripts/robustness_snr.py --source Dataset/-6_dB_pump.zip
python scripts/build_results.py                                      # adds the table to docs/RESULTS.md
```

Each run checks that the zip has the same number of files per machine and label as the 6 dB release. A machine
whose file list differs is left out, because its file numbers might not be the same recordings. Add `--verify-md5`
to also check the zip against the official checksum first (slow for a 10 GB file). Each run then writes `reports/robustness/` (`summary.txt`, `metrics.csv`, `auc_by_snr.png`). Reported per machine
and SNR:

- ROC-AUC;
- the share of normal clips above the warning and alarm limits (false alerts per window);
- the share of abnormal clips above the alarm limit (detections per window).

More noise is expected to lower ROC-AUC and push normal clips over the limits, because the limits were set on 6 dB
sound. That is a known limitation, not a defect: a real installation would be calibrated on its own normal sound.

## 7. Testing

`python -m pytest -q` runs the full suite on a synthetic MIMII-style project built inside the tests (so the dataset
and trained models are not needed). It covers:

- audio and features, models, calibration and the severity rule;
- database, API, upload, simulated monitoring and evidence;
- dashboard screens (Streamlit AppTest), reports, assistant, identification gating;
- e-mail delivery: test mode, SMTP with a local test server, retries, hourly limit, reminders;
- robustness scoring from a zip;
- security: authentication on every route, headers, lockout, hashing, injection.

`scripts/verify_inference.py` and `scripts/check_system.py` check the real installation.

## 8. Known limitations

- MIMII is laboratory data from 8 machines. Other machines, microphones or plants need their own normal recordings,
  training and calibration.
- Some machines' models detect abnormal sound much less reliably than others. The dashboard shows each machine's
  held-out test ROC-AUC and a caution where loudness alone does as well.
- Simulated monitoring replays clips; there is no live microphone input.
- One administrator role; HTTP on the local computer only (see SECURITY.md).
