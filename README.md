# Beat2Band — Real-Time Beatbox to Instrument Classifier

A real-time system that listens to beatbox sounds through a microphone, classifies them into instrument categories (kick, snare, hi-hat, k-snare, silence), and triggers corresponding drum samples — all running locally through a Flask web interface.

## Requirements

- Python 3.11.x (Python 3.14 is **not** supported — `pygame` and `simpleaudio` have no prebuilt wheels for it on Windows)
- A working microphone
- Speakers or headphones for output

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If any package fails to build on Windows, confirm your Python version is 3.11.x:

```bash
python --version
```

### 4. Verify required files are present

The repository should already include:

```
project/
├── app.py
├── svm_model_200ms.pkl
├── scaler_200ms.pkl
├── label_encoder_200ms.pkl
├── samples/
│   ├── _kick.wav
│   ├── _snare.wav
│   ├── _hi hat.wav
│   ├── _k_snare.wav
│   └── _silence.wav
├── templates/
│   └── index.html
└── static/
    └── style.css
```

If `samples/*.wav` are missing, the app will print a warning for each missing class and that class will not produce sound (classification will still work, playback will not).

## Running the app

```bash
python app.py
```

Open your browser to:

```
http://127.0.0.1:5000
```

## Usage

1. Click **Start** to begin listening through your microphone.
2. Beatbox into the mic — kick, snare, hi-hat, or k-snare sounds.
3. The page shows the detected class and confidence score, and the corresponding drum sample plays through your speakers.
4. Click **Stop** to release the microphone.

## Notes on behavior

- Detection only triggers when input energy crosses a threshold (`ENERGY_THRESHOLD = 0.02` in `app.py`). If sounds aren't being detected, try beatboxing louder or closer to the mic.
- Predictions below a per-class confidence threshold are silently rejected — this is intentional to avoid false triggers. Thresholds are defined in `CONFIDENCE_THRESHOLDS` in `app.py`.
- The `_silence` class is used internally to suppress background noise and is never played back.
- Audio is captured and played back **on the machine running the Flask server** — this is a local-only setup. If you open `http://127.0.0.1:5000` from a different device on your network, microphone and speaker access still happens on the server machine, not the client's.

## Troubleshooting

| Issue | Likely cause | Fix |
|---|---|---|
| No sound on detection | Missing `.wav` file in `samples/` | Check console for "Warning: no sample for ..." |
| Classifier never triggers | `ENERGY_THRESHOLD` too high for your mic | Lower the value in `app.py` |
| Classifier triggers on background noise | `ENERGY_THRESHOLD` too low | Raise the value in `app.py` |
| Wrong class detected frequently | Confidence thresholds too permissive | Raise values in `CONFIDENCE_THRESHOLDS` |
| `pygame` or `sounddevice` fails to install | Unsupported Python version | Use Python 3.11.x, not 3.13+/3.14 |

## Project structure summary

- **`app.py`** — Flask server: handles microphone streaming, onset detection, feature extraction, SVM classification, and sample playback.
- **`templates/index.html`** — Frontend UI with Start/Stop controls and live status display.
- **`static/style.css`** — Basic styling for the UI.
- **`*.pkl` files** — Pretrained SVM model, feature scaler, and label encoder. These must match the feature extraction logic in `app.py` (200ms windows, 40 MFCCs, hop length 64).
- **`samples/`** — One `.wav` file per class, used for audible playback on detection.
