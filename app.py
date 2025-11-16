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
    
    # Model selection
    model_options = ["tiny", "base", "small", "medium"]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=1,  # Default to "base"
        help="Larger models are more accurate but slower"
    )
    
    # Language selection
    language = st.selectbox(
        "Language",
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
        index=0
    )

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    
    # Show file info
    file_size = audio_file.size / (1024 * 1024)  # Convert to MB
    st.info(f"File: {audio_file.name} ({file_size:.2f} MB)")
    
    if st.button("Transcribe"):
        with st.spinner(f"Loading {selected_model} model..."):
            try:
                # Load model
                model = whisper.load_model(selected_model)
                
                # Read audio file
                audio_bytes = audio_file.read()
                audio_io = io.BytesIO(audio_bytes)
                audio, sr = librosa.load(audio_io, sr=16000)
                
                st.write(f"Audio loaded: {len(audio)} samples at {sr} Hz")
                
                # Transcribe
                with st.spinner("Transcribing..."):
                    options = {"task": "transcribe"}
                    
                    if language != "Auto-detect":
                        options["language"] = language.lower()
                    
                    result = model.transcribe(audio, **options)
                
                # Show result
                st.success("Transcription complete!")
                
                # Display transcription
                if "text" in result:
                    st.subheader("Transcription:")
                    st.write(result["text"])
                    
                    # Show language info
                    if "language" in result:
                        st.info(f"Detected language: {result['language'].upper()}")
                
                # Show segments with timestamps
                if st.checkbox("Show timestamps"):
                    if "segments" in result:
                        for segment in result["segments"]:
                            start = segment["start"]
                            end = segment["end"]
                            text = segment["text"]
                            st.write(f"[{start:.2f}s - {end:.2f}s] {text}")
                
                # Download button
                st.download_button(
                    label="Download transcription",
                    data=result["text"],
                    file_name=f"transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.write("Debug info:")
                st.write(f"File type: {audio_file.type}")
                st.write(f"File size: {audio_file.size} bytes")

# Tips
st.markdown("""
### Tips for Bett
