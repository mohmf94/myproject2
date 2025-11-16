import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import wavio
from tensorflow.image import resize

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\babycry\\donateacry_corpus_cleaned_and_updated_data\\Trained_model.h5")

model = load_model()

# Debugging: Print model input shape
print("Model Input Shape:", model.input_shape)  # Expected: (None, 150, 150, 3)

# Function to extract spectrogram features with proper audio loading
def extract_features(file_path, sr=48000, target_shape=(150, 150)):
    y, sr = librosa.load(file_path, sr=sr)  # Load the audio file correctly
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Ensure correct dtype conversion and resizing
    mel_s = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return mel_s

# Function to record audio for 5 seconds
def record_audio(duration=5, samplerate=22050):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_audio.name, recording, samplerate, sampwidth=2)
    return temp_audio.name

# Function to predict baby cry type
def predict_cry(file_path):
    features = extract_features(file_path, sr=48000)

    print("Final Input Shape for Prediction:", features.shape)  # Debugging step

    prediction = model.predict(np.expand_dims(features, axis=0))  # Ensure correct input shape
    labels = ['belly_pain', 'burping', 'discomfort', 'hungry']
    return labels[np.argmax(prediction)]

# Streamlit App UI
st.title("👶 Baby Cry Prediction App")

# Record Button
if st.button("🎤 Record 5 Seconds"):
    file_path = record_audio()
    st.audio(file_path, format="audio/wav")
    result = predict_cry(file_path)
    st.success(f"Predicted Cry Type: **{result}**")

# Upload Section
uploaded_file = st.file_uploader("📤 Upload a Recording (WAV)", type=["wav", "mp3"])
if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(uploaded_file.read())
    st.audio(temp_file.name, format="audio/wav")
    result = predict_cry(temp_file.name)  # Ensure you pass the correct file path
    st.success(f"Predicted Cry Type: **{result}**")
