import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile
import os
import pandas as pd
import plotly.express as px

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load ML components
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature extraction (modified to take audio array and sr)
def extract_features(y, sr):
    try:
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        target_length = sr * 30
        if len(y) == 0:
            raise ValueError("Audio data is empty or corrupted")
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        elif len(y) > target_length:
            y = y[:target_length]

        harmonic, _ = librosa.effects.hpss(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=harmonic, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

        def stats(x):
            return np.hstack([np.mean(x, axis=1),
                              np.std(x, axis=1),
                              np.median(x, axis=1)]) if x.size != 0 else np.array([])

        features = np.hstack([
            stats(mfcc), stats(mfcc_delta), stats(mfcc_delta2),
            stats(chroma), stats(mel),
            stats(contrast), stats(tonnetz)
        ])

        expected_feature_length = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 258
        if features.shape[0] != expected_feature_length:
            raise ValueError(f"Extracted features have incorrect dimension: {features.shape[0]} (expected {expected_feature_length})")

        return features
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Header
st.markdown("""
    <div class="header-container">
        <h1>Music Genre Classification System</h1>
        <p>AI-powered platform for precise audio genre detection</p>
    </div>
""", unsafe_allow_html=True)

# File uploader with drag & drop (Streamlit supports it natively)
uploaded_file = st.file_uploader("Upload a WAV file (Drag & Drop supported)", type=["wav"], label_visibility="collapsed")

# Process
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    try:
        # Validate and load audio
        with sf.SoundFile(temp_file_path) as audio_file:
            if audio_file.format != 'WAV':
                raise ValueError("Uploaded file is not a valid WAV file")
            sr = audio_file.samplerate
            y = audio_file.read(dtype='float32')

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        duration = len(y) / sr

        # Audio trimming controls
        st.markdown("<h3>Audio Trimming & Preview</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        start_time = col1.slider("Start time (seconds)", 0.0, duration, 0.0, 0.1)
        end_time = col2.slider("End time (seconds)", start_time, duration, duration, 0.1)

        # Trim audio
        y_trim = y[int(start_time * sr):int(end_time * sr)]

        # Save trimmed audio to temp for preview
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_trim_file:
            sf.write(temp_trim_file.name, y_trim, sr)
            temp_trim_path = temp_trim_file.name

        # Audio preview with controls
        st.audio(temp_trim_path, format="audio/wav")

        # Feature extraction and prediction
        features = extract_features(y_trim, sr)

        if features is not None:
            with st.spinner("Analyzing audio and predicting genre..."):
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)
                predicted_genre = label_encoder.inverse_transform(prediction)[0]

                # Get probabilities (assuming model supports predict_proba)
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(features_scaled)[0]
                else:
                    prediction_proba = np.zeros(len(label_encoder.classes_))  # Fallback
                    prediction_proba[prediction[0]] = 1.0

            # Display prediction card
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="card-header">Predicted Genre</div>
                    <div class="card-body">
                        <span class="genre-text">{predicted_genre}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("<h3>Genre Probability Distribution</h3>", unsafe_allow_html=True)
            genres = label_encoder.classes_
            df = pd.DataFrame({'Genre': genres, 'Probability (%)': prediction_proba * 100})
            fig = px.bar(df, x='Genre', y='Probability (%)',
                         color='Probability (%)',
                         color_continuous_scale='agsunset',
                         template='plotly_dark')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E0E0E0',
                title_font_color='#FFD700',
                xaxis_title_font_color='#E0E0E0',
                yaxis_title_font_color='#E0E0E0'
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("<div class='error-box'>Feature extraction failed. Please use a valid WAV file.</div>", unsafe_allow_html=True)

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_trim_path' in locals() and os.path.exists(temp_trim_path):
            os.remove(temp_trim_path)

# Footer
st.markdown("""
    <div class='footer'>
        © 2025 Sigmoid Innovations
    </div>
""", unsafe_allow_html=True)