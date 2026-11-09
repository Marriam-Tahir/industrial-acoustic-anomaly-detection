# industrial-acoustic-anomaly-detection
AI-powered industrial predictive maintenance system using PyTorch Convolutional Autoencoders for anomaly detection, ResNet-18 for fault classification, FastAPI, and Streamlit.

## Run the Smart Listener dashboard

The runnable Flask dashboard and optional Streamlit client are in [dashboard/](./dashboard/README.md). This application includes six `convae_v1` inference bundles, the exact exported preprocessing implementation, audio upload/playback, normal/abnormal results, window-score and spectrogram plots, model metrics, JSON reports, and tests.

```powershell
cd dashboard
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Open **http://127.0.0.1:5000**. Follow the [application guide](./dashboard/README.md) for the Streamlit client and API contract, the [viva guide](./dashboard/VIVA_GUIDE.md) for the live demonstration, and the [test report](./dashboard/TESTING.md) for verification details.

Machine type is selected by the operator or read from a labeled filename. The provided autoencoders do not perform acoustic machine classification. The model library displays the actual saved test metrics and their limitations.

The training scripts, existing API, reports, and `src/smart_listener` package remain available in their existing locations.
