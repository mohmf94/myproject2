# streamlitapp.py  (improved)
import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import wavio
import os
from tensorflow.image import resize

# ===== CONFIG =====
# Put Trained_model.h5 in the same folder or set MODEL_PATH to its location.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Trained_model.h5")
TARGET_SR = 22050               # use same samplerate everywhere
TARGET_SHAPE = (150, 150)       # as used during training
LABELS = ['belly_pain', 'burping', 'discomfort', 'hungry']

# ===== Load model (cached) =====
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed loading model at {MODEL_PATH}: {e}")
    st.stop()

st.title("👶 Baby Cry Prediction (Local)")

st.markdown(
    """
    **Notes**
    - Recording works only when you run this app **locally** (sounddevice accesses the server machine's mic).
    - If you deploy to a cloud service (Streamlit Cloud), use file upload instead (microphone input from the visitor's browser won't be available here).
    """
)

# ===== Helpers =====
def extract_features(file_path, sr=TARGET_SR, target_shape=TARGET_SHAPE):
    try:
        y, sr_loaded = librosa.load(file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = resize(np.expand_dims(mel_db, axis=-1), target_shape)
        # repeat to 3 channels if model expects 3
        if model.input_shape[-1] == 3:
            mel_resized = np.repeat(mel_resized, 3, axis=-1)
        return mel_resized.astype(np.float32)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def record_audio(duration=5, samplerate=TARGET_SR):
    st.info(f"Recording for {duration} s (local machine)...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(tmp.name, recording, samplerate, sampwidth=2)
    return tmp.name

def predict_cry(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, None
    preds = model.predict(np.expand_dims(features, axis=0))[0]
    idx = int(np.argmax(preds))
    return LABELS[idx], preds

# ===== UI: Recording =====
col1, col2 = st.columns(2)

with col1:
    if st.button("🎤 Record 5 seconds (local only)"):
        file_path = record_audio(5)
        st.audio(file_path, format="audio/wav")
        with st.spinner("Analyzing..."):
            label, scores = predict_cry(file_path)
        if label:
            st.success(f"Predicted: **{label}**")
            st.write({lab: float(scores[i]) for i, lab in enumerate(LABELS)})

with col2:
    uploaded = st.file_uploader("Or upload WAV/MP3 file", type=["wav", "mp3"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.read())
        tmp.flush()
        st.audio(tmp.name, format="audio/wav")
        with st.spinner("Analyzing..."):
            label, scores = predict_cry(tmp.name)
        if label:
            st.success(f"Predicted: **{label}**")
            st.write({lab: float(scores[i]) for i, lab in enumerate(LABELS)})
