# Integration verification

Verified locally on 9 November 2026 using the supplied `convae_v1` weights and unchanged export code.

## Automated tests

```text
65 passed in 19.45s
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest test_api.py test_inference.py test_ui.py -q
```

| Area | Tests | Evidence |
| --- | ---: | --- |
| Flask HTTP boundary | 25 | Success/error contract, request IDs, selector validation, malformed/unsupported/oversized uploads, readiness, static UI routes, sanitized unexpected errors |
| Real model integration | 33 | All six CPU models load; direct score/window parity with original scorer; stereo/resampling/padding/trimming; silence, constant and nonfinite audio; preprocessing tampering; missing bundles; strict threshold equality |
| Streamlit behavior | 7 | Normal/abnormal result rendering with controlled API replies, disabled states, offline/timeout errors, and clearing a result after an input changes or a request fails |

The editor's problems tool reported no errors for the application, test, and browser JavaScript files.

## Live service checks

- `GET http://127.0.0.1:5000/api/health`: 200, `ready`, six loaded models, preprocessing fingerprint `55fac6205183`.
- A real multipart upload through the running Flask server returned 200, nineteen window scores, a real model decision, and a matching response/header request ID.
- A malformed WAV returned 400 with `INVALID_AUDIO` and the structured error contract.
- The optional Streamlit server at http://127.0.0.1:8502 accepted an actual browser upload and displayed the Flask-derived result, metrics, nineteen-window plot, and JSON report control. AppTest also initialized the client with no exceptions.

## Main dashboard browser checks

A browser session exercised the live application through its real API:

1. Initial empty state and disabled Analyze button.
2. WAV upload, waveform preview, and actual Fan anomaly result.
3. Nineteen scores on the timeline and a populated spectrogram.
4. JSON report content: real machine, all nineteen scores, and the verified fingerprint; download link generated a `.json` filename and was activated.
5. All six model cards and saved test metrics.
6. API/pipeline screen and configuration fingerprint.
7. Result metadata added to history; saved fields exclude audio and feature arrays.
8. Generic filename in Auto mode produces a machine-selection error and clears stale results.
9. Explicit fan token in a filename is reported as filename identification.
10. Corrupt WAV produces an actionable error, followed by a successful subsequent Pump analysis.
11. No unhandled JavaScript errors during the exercised flow.
12. Narrow viewport of 390 CSS pixels: body content width 375 pixels, recording card about 343 pixels after the mobile navigation adjustment, and no horizontal document overflow. A desktop layout was also inspected with two analysis columns.

The integrated browser did not expose a completed native download event. The report Blob's JSON content and anchor activation were verified; the browser-managed save-to-disk completion was not confirmed through that event.

## Interpretation and remaining model work

Generated signals were used for software checks because the provided folder contains exports and reports, not original WAV examples. Synthetic signals and mocked UI replies are not labeled real machines and do not measure statistical detection performance. Use held-out original recordings for model evaluation and your viva examples.

The six new autoencoders have no acoustic machine classifier. Selection/filename metadata is explicit; machine IDs and fault types are not predicted. The supplied anomaly report shows limited test AUC on unseen machines, displayed in the UI. No accuracy claim, confidence percentage, threshold adjustment, or substitute model was introduced.

See [README.md](./README.md) for startup and the HTTP contract, and [VIVA_GUIDE.md](./VIVA_GUIDE.md) for the demonstration script.
