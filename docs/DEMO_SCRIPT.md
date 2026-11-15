# Smart Listener - demo script (about 12 minutes)

Use **PUMP-04** and **FAN-06** for the live parts. In the identification study (section 7 of
reports/identification/summary.txt) their models put the most abnormal test clips above the alarm limit, so they
show the system at its clearest. Say so if asked: other machines' models are weaker, and the dashboard shows that
openly.

## The day before

1. `python scripts/check_system.py`: no FAIL lines. A WARN for test-mode e-mail is fine if you are not showing
   e-mail.
2. Settings: replay speed **1.0** s per clip (a fault run then takes about 40 s). Make sure the owner e-mail of
   PUMP-04 is an inbox you can show, for example on your phone.
3. Run one **fault then recovery** simulation on PUMP-04 so Incidents, Reports and the Assistant already have data.
   Acknowledge it with a note.
4. Pick an abnormal **test** clip of PUMP-04 for the upload demo and copy it to the desktop:
   ```bash
   python -c "import pandas as pd; m=pd.read_csv('data/manifests/manifest.csv', keep_default_na=False); print(m[(m.group=='pump_id_04')&(m.split_ad=='test')&(m.label_name=='abnormal')].path.head(3).to_string())"
   ```
   The files are under `data/interim/mono16k/`. Rename the copy to `pump_id_04_demo.wav` so the dashboard suggests
   the machine from the file name.
5. Backup in case something fails: export one incident PDF, and take screenshots of Overview, an analysis result and
   an incident detail.

## On the day

`python scripts/start_all.py`, sign in, and keep the browser at 100% zoom.

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Login | Sign in | "Smart Listener listens to machines and tells maintenance when one sounds unlike its normal self." |
| 0:45 | Overview | Point to the summary numbers, machine status and needs attention | "One autoencoder per machine learned its normal sound. Severity is judged over several 10-second windows, so one odd noise does not raise an alarm." |
| 2:00 | Analyze Audio | Upload `pump_id_04_demo.wav`, show the suggested machine, Run analysis | "We choose the machine. We tested identifying it from the sound: abnormal recordings were usually matched to the wrong machine, so we did not use it." |
| 3:00 | Result | Point to z against the limits, the evidence picture and the top frequency bands | "This is how far it is from normal, and where in the sound. It does not say which part is broken. The data has no fault labels, so we do not claim that." |
| 4:30 | Machines > PUMP-04 > Simulated monitoring | Start the **fault** scenario | "This replays recorded MIMII clips as if a microphone were listening. It is labelled SIMULATED everywhere." |
| 5:00 | (while it runs) | Watch the chart climb: NORMAL, then WATCH, WARNING, HIGH | "The level rises only when deviations repeat, and comes down only after several clean windows." |
| 6:00 | Banner / Incidents | Open the new incident, show the timeline and notification | "One e-mail per incident went to the owner." Show the inbox, or the TEST MODE record. |
| 7:00 | Incidents | Add an operator note, acknowledge | "Acknowledging stops reminders, and the audit trail records who did it." |
| 7:45 | Assistant | Ask: "What did the model detect?", then "Can you identify the exact fault?", then "When will it fail?" | "Model findings, records and general advice are labelled separately. When the data cannot answer, it says 'not determined'." |
| 9:00 | Reports | Generate the incident PDF and open it | "Reports come straight from the database." |
| 9:45 | Settings | E-mail delivery, machine registry (test ROC-AUC, cautions) | "We show how reliable each machine's model is, including the weak ones." |
| 10:30 | Close | Show docs/RESULTS.md | Summarise results and limitations (below). |

## Closing points

- Per-machine detectors with calibrated, shared limits. Test data was never used to tune anything.
- Severity with persistence and hysteresis; incidents, e-mail with retries and reminders, reports and an audit trail.
- Honest scope: no fault diagnosis, no failure prediction, no probability of failure. Automatic machine
  identification was tested and rejected. Robustness to louder noise was measured on 0 dB and −6 dB recordings
  (RESULTS.md, section 4).
- Next steps for a real plant: record each machine's own normal sound, connect live microphones, recalibrate on site,
  and add HTTPS and user roles.

## Likely questions

| Question | Answer |
|---|---|
| Why an autoencoder, not a classifier? | Faults are rare and varied; we have few labelled faults. A model of normal sound flags anything unusual without needing examples of every fault. |
| How do you avoid false alarms? | Limits come from normal validation sound, and HIGH needs several windows above the alarm limit. The rule was tuned for no false HIGH on normal validation streams. |
| Why are some machines weaker? | Their abnormal sounds are close to normal or mostly differ in loudness. We measured this, and the dashboard shows a caution. |
| What happens with more factory noise? | Measured on the 0 dB and −6 dB releases (RESULTS.md, section 4). More noise lowers separation and raises false alerts with 6 dB limits, so a real site needs its own calibration. |
| Could it run on a real factory floor? | The pipeline is the same. It needs on-site normal recordings per machine, a microphone feed instead of the replay, and recalibration. |
| Is the data leakage-safe? | Splits are per machine in interleaved blocks with gaps between them; normalisation and limits are fitted without test data. |

## If something goes wrong

| Problem | Fix |
|---|---|
| Dashboard says it cannot reach the API | The API window shows the error; restart with `python scripts/start_all.py`. |
| Port already in use | Close the old windows, or `python scripts/start_all.py --api-port 8001 --dashboard-port 8502`. |
| E-mail not arriving | Settings > E-mail delivery shows the reason (for example a wrong app password). Show the record instead; the delivery process is still visible. |
| Simulation seems stuck | Each clip takes the replay speed in seconds; check Settings. Stop and restart the session. |
| Anything else | Use the backup PDF and screenshots, and explain from them. |
