import streamlit as st
import whisper
import torch
import numpy as np
import io
import librosa
import soundfile as sf

# Copyright (c) 2024 SavioLazarus
# Licensed under MIT License - see LICENSE file for details

# Page configuration
st.set_page_config(
    page_title="Whisper Transcription App",
    page_icon="🎙️",
    layout="wide"
)

# App title
st.title("🎙️ Whisper Transcription App")
st.write("Upload an audio file to transcribe it")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Transcribe"):
        with st.spinner("Loading model..."):
            try:
                # Load the model
                model = whisper.load_model("tiny")
                
                # Read audio file using librosa
                audio_bytes = audio_file.read()
                
                # Create a file-like object from bytes
                audio_io = io.BytesIO(audio_bytes)
                
                # Load audio with librosa
                audio, sr = librosa.load(audio_io, sr=16000)
                
                st.write(f"Audio loaded: {len(audio)} samples at {sr} Hz")
                
                # Transcribe directly from numpy array
                with st.spinner("Transcribing..."):
                    result = model.transcribe(audio)
                
                # Show result
                st.success("Transcription complete!")
                st.write(result["text"])
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.write("Debug info:")
                st.write(f"File type: {audio_file.type}")
                st.write(f"File size: {audio_file.size} bytes")
