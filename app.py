from flask import Flask, render_template, jsonify
import sounddevice as sd
import numpy as np
import librosa
import pickle
import pygame
import threading
from collections import deque
import os

app = Flask(__name__)

pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
pygame.mixer.set_num_channels(32)

SR = 22050
CHUNK = int(SR * 0.04)
WINDOW = int(0.2 * SR)
ENERGY_THRESHOLD = 0.02
COOLDOWN_FRAMES = 8
PRE_ONSET_FRAMES = 1
FRAMES_NEEDED = int(np.ceil(WINDOW / CHUNK))

with open("svm_model_200ms.pkl", "rb") as f:
    svm = pickle.load(f)
with open("scaler_200ms.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder_200ms.pkl", "rb") as f:
    le = pickle.load(f)

CONFIDENCE_THRESHOLDS = {
    "_kick": 0.50,
    "_hi hat": 0.40,
    "_snare": 0.60,
    "_k_snare": 0.60,
    "_silence": 0.99,
}

wav_cache = {}
for cls in le.classes_:
    path = os.path.join("samples", f"{cls}.wav")
    if os.path.exists(path):
        wav_cache[cls] = pygame.mixer.Sound(path)

# Shared state for the frontend to poll
state = {"label": None, "confidence": 0.0, "running": False}

pre_buffer  = deque(maxlen=PRE_ONSET_FRAMES)
post_buffer = []
collecting  = False
frames_collected = 0
cooldown = 0
stream = None

def classify_and_play(audio):
    audio = np.array(audio, dtype=np.float32)
    if len(audio) < WINDOW:
        audio = np.pad(audio, (0, WINDOW - len(audio)))
    else:
        audio = audio[:WINDOW]

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40, hop_length=64)
    vec  = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)

    pred  = svm.predict(vec_scaled)[0]
    prob  = svm.predict_proba(vec_scaled).max()
    label = le.inverse_transform([pred])[0]

    threshold = CONFIDENCE_THRESHOLDS.get(label, 0.65)

    if prob < threshold or "silence" in label:
        return

    state["label"] = label
    state["confidence"] = float(prob)

    if label in wav_cache:
        wav_cache[label].play()

def audio_callback(indata, frames, time, status):
    global collecting, frames_collected, cooldown

    chunk = indata[:, 0].copy()
    rms   = np.sqrt(np.mean(chunk ** 2))

    if cooldown > 0:
        cooldown -= 1
        pre_buffer.append(chunk)
        return

    if not collecting:
        pre_buffer.append(chunk)
        if rms > ENERGY_THRESHOLD:
            collecting = True
            frames_collected = 0
            post_buffer.clear()
            post_buffer.extend(list(pre_buffer))
            post_buffer.append(chunk)
            frames_collected = len(post_buffer)
    else:
        post_buffer.append(chunk)
        frames_collected += 1

        if frames_collected >= FRAMES_NEEDED:
            audio_window = np.concatenate(post_buffer)
            threading.Thread(
                target=classify_and_play,
                args=(audio_window,),
                daemon=True
            ).start()
            collecting = False
            post_buffer.clear()
            cooldown = COOLDOWN_FRAMES

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start_stream():
    global stream
    if not state["running"]:
        stream = sd.InputStream(samplerate=SR, channels=1,
                                  blocksize=CHUNK, callback=audio_callback)
        stream.start()
        state["running"] = True
    return jsonify({"status": "started"})

@app.route("/stop")
def stop_stream():
    global stream
    if state["running"] and stream:
        stream.stop()
        stream.close()
        state["running"] = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def status():
    return jsonify({
        "label": state["label"],
        "confidence": state["confidence"],
        "running": state["running"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)