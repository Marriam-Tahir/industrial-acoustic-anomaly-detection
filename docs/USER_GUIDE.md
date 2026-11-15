# Smart Listener - user guide

For the people who use the dashboard: maintenance managers, operators and the administrator. No technical background
is needed.

## The idea in one paragraph

Each registered machine has its own model that has learned what that machine sounds like when it is working
normally. A recording is cut into 10-second pieces ("windows"), and each window gets an **anomaly level z**: how far
the sound is from that machine's normal sound. Small numbers are normal. Above the **warning limit** the sound is
unusual; above the **alarm limit** it is very unusual. One odd window is not enough to worry about, so the
**severity** of a machine is judged from several windows in a row.

| Severity | Meaning | What to do |
|---|---|---|
| NORMAL | Within the machine's normal range | Nothing |
| WATCH | One window above the warning limit | Keep an eye on it |
| WARNING | Repeated windows above the warning limit, or one above the alarm limit | Plan a check |
| HIGH | Several windows above the alarm limit: a sustained deviation. An **incident** opens and the owner is e-mailed | Inspect the machine, then acknowledge the incident |

The system tells you **that** a machine sounds different and **where** in the sound the difference is. It does not
tell you **which part is faulty**, **when** the machine will fail, or **how likely** a failure is.

## Signing in

Use the username and password the administrator gave you. After five wrong passwords in a row, sign-in is paused for
a few minutes. A forgotten password is reset by the administrator with `scripts/init_db.py`.

## Overview

- **Summary numbers:** registered machines, how many are normal, how many need attention, and open incidents.
- **Machine status:** every machine with its severity, its latest anomaly level and when it was last analysed.
  Machines needing attention are listed first. **Details** opens the machine.
- **Needs attention:** open HIGH incidents (with **Review incident**) and machines at WATCH, WARNING or HIGH.
- **Recent activity:** the latest severity changes. Events from simulated monitoring are marked SIMULATED.

A red banner at the top of every screen shows an open HIGH incident that nobody has acknowledged yet.

## Analyze Audio

1. **Choose a WAV file**, at least 10 seconds long. A file longer than 10 seconds is cut into 10-second windows.
2. **Choose the machine** it was recorded from. Each machine has its own model and alert limits, so this matters. If
   the file name contains the machine ID (for example `pump_id_04_...wav`), that machine is suggested. The suggestion
   comes from the name, not from the sound.
3. **Run analysis.** The result shows:
   - the anomaly score and level `z`, with the warning and alarm limits;
   - the severity over the recording, when it has enough windows (one window alone is shown as a window result);
   - the evidence picture: waveform, spectrogram, residual heatmap and the frequency bands that differ most;
   - every window in a table.

Analysing an uploaded file **never** changes a machine's status and never opens an incident. Uploaded audio is not
stored.

**E-mail after an upload.** When the result is not normal, the machine's owner is e-mailed automatically, once per
upload, with three attachments: a PDF report, a CSV with every 10-second window, and the evidence picture.

| File length | E-mail is sent when |
|---|---|
| 60 seconds or longer (6+ windows, severity is judged) | severity is WATCH, WARNING or HIGH (never for NORMAL) |
| shorter (severity "not assessed") | at least one window is ABOVE ALARM LIMIT |

The result card says whether an e-mail was sent. Without a mail server (TEST MODE) the message is saved in
`data/app/outbox` instead. The attachments are kept in `data/app/attachments`. The hourly e-mail limit in Settings
also applies to these messages.

## Machines

- **Registered machines:** every machine, its model, its current severity and its last analysis.
- **Machine detail:** facts about the machine, its owner (who receives alerts) and how reliable its model was on
  held-out test data. A yellow note means the model is less trustworthy for this machine.
  - **History:** the anomaly level over time, with the limits drawn in, and every severity change.
  - **Simulated monitoring:** replays recorded MIMII clips one after another, as if a microphone were listening.
    Pick a scenario (normal, fault, fault then recovery) and press **Start simulated monitoring**. The header shows
    SIMULATED MONITORING while it runs. This is how the incident process is demonstrated; it is not live data.
  - **Analysed windows:** every analysed window, with its evidence picture.
  - **Edit machine:** change the name, location or owner, or mark the machine inactive.
- **Add machine:** choose the machine ID (it selects the trained model; the type follows from it), then enter a
  name, location and owner e-mail.

## Incidents

An incident opens when a machine reaches HIGH. It closes on its own when the machine is normal again, and re-opens if
HIGH returns soon after.

- Filter by status (open, closed, not acknowledged) and by machine.
- **Details** shows the machine, start and end time, peak anomaly level, the evidence for the window that opened it,
  severity changes, e-mails sent and operator notes.
- **Acknowledge incident:** records that someone has reviewed it (with an optional note) and stops reminders.
- **Add operator note:** adds an observation, for example what the inspection found.
- **Download report:** opens Reports with this incident selected.
- **Ask assistant:** opens the assistant for this incident.

## Reports

| Report | Contents |
|---|---|
| Incident report (PDF) | One incident: facts, severity changes, e-mails, notes, chart and evidence picture |
| Machine report (PDF) | One machine over a date range: windows analysed, highest level, incidents, chart |
| Analysis data (CSV) | One row per analysed window, for Excel |
| Incident list (CSV) | One row per incident |

Choose the report, press **Generate report**, then **Download**.

## Assistant

Choose an incident and ask a question in plain words, or press one of the suggested questions. Each part of an
answer says where it comes from:

- **From the model:** what the model measured.
- **From the records:** times, status, acknowledgements, e-mails.
- **Not determined:** the data cannot answer this (for example "which part is broken?" or "when will it fail?").
- **General practice:** ordinary maintenance advice, not based on this incident's data.

## Settings (administrator)

- **Administrator account:** change your password. Your other sessions are signed out.
- **Email notifications:** switch alert e-mails on or off, set the most e-mails per hour, and set reminders for
  unacknowledged incidents.
  - **TEST MODE** means no mail server is configured: messages are saved, not sent.
  - **SMTP CONFIGURED** means real e-mails are sent.
- **E-mail delivery:** waiting, sent and failed messages. **Send test** checks delivery, **Send pending now** sends
  without waiting, and a failed message can be put back in the queue.
- **Machine registry:** each machine's model status, limits, test ROC-AUC and alert address.
- **Audit trail:** who did what and when.

## Times

All times are shown in Pakistan Standard Time (PKT, time zone `Asia/Karachi`, UTC + 5 hours): on the website, in reports, CSV files and e-mails. The database still stores every time in UTC; it is converted only for display. CSV files keep the original `_utc` columns and add `_pkt` columns. Report date filters are Pakistan calendar days.
