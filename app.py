import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
from tensorflow.image import resize
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "Trained_model.h5"       # Upload using Git LFS
TARGET_SR = 22050
TARGET_SHAPE = (150, 150)
LABELS = ['belly_pain', 'burping', 'discomfort', 'hungry']

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_resized = resize(np.expand_dims(mel_db, -1), TARGET_SHAPE)

        # If model expects 3 channels
        if model.input_shape[-1] == 3:
            mel_resized = np.repeat(mel_resized, 3, axis=-1)

        return mel_resized
    except Exception as e:
        st.error(f"Feature error: {e}")
        return None

# -----------------------------
# Prediction
# -----------------------------
def predict(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, None

    preds = model.predict(np.expand_dims(features, 0))_
