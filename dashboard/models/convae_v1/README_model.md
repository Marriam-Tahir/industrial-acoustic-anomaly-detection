# Smart Listener - Stage-1 convolutional autoencoder (run convae_v1)

Trained only on NORMAL clips of the TRAIN machines (id_00, id_02) of the Member 1 data; validation machine id_04
chooses the aggregation and threshold; test machine id_06 gives the final numbers (`reports/test_results.csv`).
Preprocessing fingerprint: `55fac6205183` (the scorer refuses to run with a different one).

## Files
- `experiments/<name>/`: `best.pt`, `last.pt` (resume), `history.csv`, `done.json`, `val_threshold.json`, `val_scores.csv`,
  `test_metrics.json`, `test_scores.csv`, `test_plots.png`
- `export/<name>/`: `model.pt`, `model_config.json` (architecture, aggregation, threshold, fingerprint)
- `export/`: `preprocessing_config.json`, `smart_listener_preprocess.py`, `smart_listener_scoring.py`, `smart_listener_model.py`

## Member 3: score new audio
```python
import sys; sys.path.insert(0, "export")
from smart_listener_model import AnomalyScorer
scorer = AnomalyScorer.from_dir("export/fan_6_dB", "export/preprocessing_config.json")
print(scorer.score_file("clip.wav"))   # {'score': ..., 'threshold': ..., 'is_anomaly': ..., 'window_scores': [...]}
```
Pick the bundle that matches the machine type and noise level of the audio. A recording is anomalous when
`score > threshold`; the score is the max/mean (stored per bundle) reconstruction error over the 19 one-second windows.

## Limits
Validation and test each contain one machine ID (the public MIMII release has four per machine type), so threshold and
test numbers describe a single unseen machine; they are not a general guarantee.
