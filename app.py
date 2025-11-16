import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import wavio
import os
from tensorflow.python.ops.image_ops_impl import resize_images as resize
# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Trained_model.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def extract_features(file_path, sr=48000, target_shape=(150, 150)):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_s = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_s = np.repeat(mel_s, 3, axis=-1)  # Convert to 3-channel
        return mel_s
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def record_audio(duration=5, samplerate=48000):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp_audio = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
    wavio.write(temp_audio.name, recording, samplerate, sampwidth=2)
    return temp_audio.name

def predict_cry(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error"
    prediction = model.predict(np.expand_dims(features, axis=0))
    labels = ['belly_pain', 'burping', 'discomfort', 'hungry']
    return labels[np.argmax(prediction)], prediction[0]

# Streamlit UI
st.title("👶 Baby Cry Prediction App")

if st.button("🎤 Record 5 Seconds"):
    file_path = record_audio()
    st.audio(file_path, format="audio/wav")
    with st.spinner("Analyzing cry..."):
        result, confidences = predict_cry(file_path)
    st.success(f"Predicted Cry Type: **{result}**")
    st.write("Confidence Scores:", dict(zip(labels, confidences)))

uploaded_file = st.file_uploader("📤 Upload a Recording (WAV)", type=["wav", "mp3"])
if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
    temp_file.write(uploaded_file.read())
    st.audio(temp_file.name, format="audio/wav")
    with st.spinner("Analyzing cry..."):
        result, confidences = predict_cry(temp_file.name)
    st.success(f"Predicted Cry Type: **{result}**")
    st.write("Confidence Scores:", dict(zip(labels, confidences)))