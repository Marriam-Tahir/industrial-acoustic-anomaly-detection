# Smart Listener viva guide

## Five-minute live demonstration

1. Start the Flask dashboard using the command in [README.md](./README.md). Open http://127.0.0.1:5000 and show **System ready**. Show the six bundles in Model library, including their actual saved thresholds and test metrics.
2. Upload an original 10-second fan or pump WAV, select the correct type and known noise condition, and play the audio. Select Analyze recording.
3. Explain the result: machine type and its source, strict score/threshold comparison, aggregation, window-error timeline, frequency display, and timing. Download the response JSON.
4. Upload a second original recording with a known normal/abnormal ground truth. Discuss the model decision honestly, including false positives/negatives. Do not choose a different threshold to make the demonstration look correct.
5. Show an error: upload a corrupt WAV, a silent file, or use Auto with a generic filename. Explain the actionable message and request ID. The app remains usable after a failed request.
6. Open API & pipeline to explain the request fields and configuration checks. Optionally show the Streamlit client at http://127.0.0.1:8502 making the same API call.

Prepare original WAV examples before the viva. The locally supplied package includes model artifacts and processed feature arrays, but no original WAV recordings were available during integration. The generated synthetic file is clearly a software check; it is not evidence of normal/abnormal machine accuracy.

## What each part owns

Browser/Streamlit → Flask multipart request → input validation → machine/model selection → shared preprocessing → PyTorch autoencoder → exported aggregation/threshold → JSON response → plots and report.

The backend owns model selection and scoring. The dashboard does not calculate the verdict, fit thresholds, or replace missing models with mock predictions. All six models run on CPU and are cached in memory. Requests use a scoring lock to avoid overlapping CPU forwards in this local demo.

## Request and response explanation

`POST /api/predict` accepts a WAV in `file`, machine type in `machine_type`, and known signal-to-noise condition in `snr`. The result includes the model ID, machine source, Normal/Abnormal, raw reconstruction score, validation-derived threshold, all nineteen window errors, preprocessing fingerprint, warnings, processing time, and request ID. `X-Request-ID` lets you correlate a response with server errors.

Abnormal is the positive class and means **score > threshold**. Equality is normal. Max aggregation means one high-error window can determine the decision. Mean aggregation averages all windows; compare the recording's aggregate, not every individual window, to the threshold for the final decision.

`score_ratio` is a visualization aid, not confidence. These models were not calibrated to output defect probabilities. The spectrogram summarizes signal frequencies; it is not a localization of a specific mechanical fault.

## How consistency is guaranteed

The model export includes [preprocessing_config.json](./models/convae_v1/export/preprocessing_config.json) and the same preprocessing module used for training. Startup checks:

1. Compare the module bytes with the saved SHA-256 prefix.
2. Recompute the parameter fingerprint **55fac6205183** and compare it with the JSON and implementation settings.
3. Require every model's fingerprint and input shape to match. Verify machine/noise metadata, aggregation, and threshold.
4. Load real weights and run a CPU forward pass before reporting the model ready.

At inference, `Preprocessor.preprocess_samples` from that export performs every audio transformation. Tests compare the application's scores and all window errors directly with the original exported scorer for every bundle, plus short/long/stereo/resampled input. This checks that the integration preserves training features and scoring; it does not improve the model's statistical accuracy.

The fixed pipeline is mean-channel mixdown, resampling, peak normalization, trim/zero-pad to 10 seconds, nineteen one-second windows, mel power, natural log, per-window min/max, float32. Peak normalization happens before trimming, exactly as in the training implementation. Changing the feature parameters requires rebuilding features and retraining/exporting the model; editing the app settings cannot silently change them.

## Error handling

Missing/corrupt audio returns 400, unsupported extensions 415, oversized uploads 413, invalid selectors or unusable signal 422, unavailable models 503, unexpected failures 500. Responses share `{error: {code, message}, request_id}`. Internal exceptions are logged on the server, not shown as tracebacks in the UI. An error clears the previous result so an old decision cannot be mistaken for the failed upload.

Inputs are capped by size, channel count, sample-rate range, and duration. Silence/near-constant sound is rejected before a health verdict. Short inputs are padded and visibly flagged; long recordings show that only the first ten seconds provide model windows. Choosing the correct machine and noise condition remains an operator responsibility.

## Machine recognition: say this clearly

“The provided stage-one export contains machine-specific anomaly detectors, not an acoustic type classifier. I identify the machine through the operator's selection or an explicit fan/pump token in the filename, and label that source. I never infer the type from the lowest autoencoder error. Automatic sound-based identification needs a separate classifier, its label mapping, its matching preprocessing, and validation on unseen machine IDs.”

No serial number or machine ID is predicted from the audio. The older sibling classifier was not integrated because its own model card says not to use it as a gate, and it belongs to a different feature pipeline.

## Explain the observed model limits

The new `convae_v1` report has fan test AUC about 0.44–0.51 and pump about 0.61–0.65. Do not quote the older sibling project's much higher per-machine metrics for these six exported models. The app shows the new export's actual results and keeps the saved validation thresholds unchanged. A correct integration can still expose a weak model. Improving that requires model/data work, broader validation, or retraining; it is separate from the API/UI implementation.

## Checks to mention

Automated coverage includes HTTP success/error contracts, all-six-model loading, original-scorer parity, stereo/resampling, padding/trimming, threshold equality, filename ambiguity, unsupported/malformed/oversized audio, silence/nonfinite values, missing models, and preprocessing tampering. Run the tests with the exact command in [README.md](./README.md). Browser checks exercise the live UI through the actual Flask service.
