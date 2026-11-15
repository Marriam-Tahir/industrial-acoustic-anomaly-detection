# Smart Listener - Industrial Acoustic Anomaly Detection

A system that analyses industrial machinery sounds to detect unusual acoustic patterns, assess operational severity
and monitor machine incidents.

Capstone project for the NAVTTC-sponsored AI/ML training programme at NetSol Technologies.
Team: Yusra Nazish, Marriam Tahir, Tahira Kainat.

---

## What it does

- **Learns each machine's normal sound.** One autoencoder per machine (MIMII fans and pumps, IDs 00/02/04/06,
  6 dB recordings), trained on normal recordings only.
- **Scores new sound.** Every 10-second window gets an anomaly score, converted to a calibrated level `z` for that
  machine and compared with shared warning and alarm limits.
- **Assesses operational severity.** NORMAL / WATCH / WARNING / HIGH from several consecutive windows, so a single
  odd sound does not raise an alarm.
- **Manages incidents.** A HIGH incident opens, the owner is e-mailed once (SMTP, or test mode), reminders follow
  until someone acknowledges it, and operator notes are kept.
- **Shows evidence.** Waveform, log-mel spectrogram, residual heatmap (where the model could not reproduce the sound)
  and the frequency bands that differ most.
- **Reports.** Incident and machine PDF reports and CSV exports, all built from the database.
- **Maintenance assistant.** Answers questions about an incident from its records only, with no language model,
  and says "not determined" when the data cannot answer.

### What it does not do

- It does **not** diagnose the physical fault, predict failure time or give a probability of failure.
- It does **not** identify which machine a recording came from. This was investigated and rejected
  (docs/RESULTS.md, section 3); the user selects the machine.
- **Simulated monitoring** replays recorded MIMII clips and is labelled SIMULATED everywhere. It is not a live
  microphone connection.

## Architecture

```
 WAV upload / simulated replay
            |
            v
 FastAPI backend (src/smart_listener/api.py)  ---- SQLite database (data/app/smartlistener.db)
   inference.py   log-mel -> autoencoder -> score -> calibrated z
   severity.py    NORMAL / WATCH / WARNING / HIGH, incidents
   evidence.py    evidence pictures and summaries
   monitor.py     simulated monitoring sessions (background threads)
   notifier.py    e-mail delivery (SMTP or test mode), retries, hourly limit, reminders
   reports.py     PDF / CSV        assistant.py  rule-based incident assistant
            |
            v  HTTP + login token
 Streamlit dashboard (dashboard/app.py): Overview, Analyze Audio, Machines, Incidents, Reports, Assistant, Settings
```

The dashboard talks only to the API; it never opens the database. The API listens on 127.0.0.1 only.

## Technology stack

Python 3.12 (conda environment `smartlistener`), NumPy, pandas, librosa, soundfile, PyTorch, matplotlib, FastAPI,
Uvicorn, httpx, Streamlit, SQLite, pytest. Heavy training runs on Google Colab. There is no Docker, cloud service or
JavaScript framework.

## Quick start (Windows, Anaconda Prompt or Git Bash)

```bash
cd /d E:\Yusra\IndustrialAcousticAnamolyDetection        # Git Bash: cd /e/Yusra/IndustrialAcousticAnamolyDetection
conda activate smartlistener
pip install -r requirements-app.txt                      # once
python scripts/check_system.py                           # everything in place? (models, data, database, e-mail)
python scripts/init_db.py --create-admin admin           # once; asks for a password
python scripts/start_all.py                              # starts the API and the dashboard together
```

Open http://localhost:8501 and sign in. Press Ctrl+C in the same window to stop both programs.

You can also start them separately: `python scripts/run_api.py`, then in a second window
`streamlit run dashboard/app.py`, run from the project folder so the theme in `.streamlit/config.toml` is used.

### E-mail alerts

Without configuration the system runs in **TEST MODE**: alert e-mails are saved in `data/app/outbox` and are NOT
sent. To send real e-mail, copy `config/smtp.env.example` to `config/smtp.env`, fill in the account details (for
Gmail use an app password), and restart the API. `config/smtp.env` holds a password, so it is listed in `.gitignore`
and must never be committed. Settings > E-mail delivery > Send test confirms delivery.

## Rebuilding the results (pipeline)

| Step | Command | Output |
|---|---|---|
| 1. Dataset preparation | Phase 1 scripts (download check, audit, leakage-safe splits) | `data/manifests/manifest.csv`, `data/interim/mono16k/` |
| 2. Features | `python scripts/extract_features.py` | `data/processed/` |
| 3. Training (Colab) | `python scripts/make_colab_bundle.py`, then `notebooks/colab/02_phase2_train.ipynb` | `models/ad/<model>/<machine>.pt` |
| 4. Evaluation | `python scripts/evaluate_ad.py` | `reports/phase2/` |
| 5. Loudness baseline | `python scripts/sanity_level_baseline.py` | `reports/phase3/` |
| 6. Calibration | `python scripts/calibrate_severity.py` | `config/severity_calibration.json` |
| 7. Stream behaviour | `python scripts/evaluate_streams.py` | `reports/phase3/` |
| 8. App scoring check | `python scripts/verify_inference.py` | same scores as step 4 |
| 9. Identification study | `python scripts/investigate_identification.py` | `reports/identification/` |
| 10. Noise robustness | `python scripts/robustness_snr.py` (see docs/TECHNICAL.md) | `reports/robustness/` |
| 11. Results page | `python scripts/build_results.py` | `docs/RESULTS.md` |

Test recordings are never used to choose a model, a limit or a setting.

## Tests

```bash
python -m pytest -q
```

The tests build a small synthetic MIMII-style project, so they need neither the dataset nor the trained models. They
never send real e-mail, even when `config/smtp.env` exists.

## Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md): using the dashboard, screen by screen
- [docs/TECHNICAL.md](docs/TECHNICAL.md): data, models, calibration, severity, API, design decisions
- [docs/RESULTS.md](docs/RESULTS.md): measured results, generated from the report files
- [docs/SECURITY.md](docs/SECURITY.md): what is protected and how, and what is not
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): the presentation walkthrough

## Data

MIMII dataset (Purohit et al., 2019), Zenodo record 3384388, licence CC BY-SA 4.0. The dataset files are not stored
in this repository.
