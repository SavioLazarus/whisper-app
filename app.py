import streamlit as st
import whisper
import torch
import numpy as np
import io
import librosa

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

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection with memory warnings
    model_options = ["tiny", "base", "small"]
    model_info = {
        "tiny": "Fastest, ~39MB model",
        "base": "Good balance, ~74MB model",
        "small": "Better quality, ~244MB model"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=1,  # Default to "base"
        format_func=lambda x: f"{x} - {model_info[x]}"
    )
    
    # Memory warning
    if selected_model == "small":
        st.warning("⚠️ 'small' model may be slow on free tier. Consider 'base' for faster processing.")
    
    # Language selection
    language = st.selectbox(
        "Language",
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
        index=0
    )
    
    # Show device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device.upper()}")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    
    # Show file info
    file_size = audio_file.size / (1024 * 1024)  # Convert to MB
    st.info(f"File: {audio_file.name} ({file_size:.2f} MB)")
    
    # Check file size warning
    if file_size > 10:
        st.warning("⚠️ Large files may take longer to process. Consider using a shorter clip.")
    
    if st.button("Transcribe"):
        with st.spinner(f"Loading {selected_model} model..."):
            try:
                # Check available memory
                if device == "cpu":
                    if selected_model == "small":
                        st.info("Loading small model on CPU... This may take a moment.")
                
                # Load model with error handling
                try:
                    model = whisper.load_model(selected_model, device=device)
                except Exception as model_error:
                    st.error(f"Failed to load {selected_model} model: {model_error}")
                    st.info("Try using a smaller model (tiny or base)")
                    return
                
                # Read audio file
                audio_bytes = audio_file.read()
                audio_io = io.BytesIO(audio_bytes)
                
                # Load audio with error handling
                try:
                    audio, sr = librosa.load(audio_io, sr=16000)
                    st.write(f"Audio loaded: {len(audio)} samples at {sr} Hz")
                except Exception as audio_error:
                    st.error(f"Failed to load audio: {audio_error}")
                    st.info("Try uploading a different audio file")
                    return
                
                # Transcribe with progress
                with st.spinner("Transcribing..."):
                    try:
                        options = {
                            "task": "transcribe",
                            "fp16": device
