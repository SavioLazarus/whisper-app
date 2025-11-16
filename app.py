import streamlit as st
import whisper
import torch
import os
import tempfile
from io import BytesIO
import time

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
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Transcribe"):
        with st.spinner("Loading model..."):
            try:
                # Load the smallest model to start
                model = whisper.load_model("tiny")
                
                # Save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name
                
                st.write(f"Processing file: {tmp_path}")
                
                # Transcribe
                with st.spinner("Transcribing..."):
                    result = model.transcribe(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                # Show result
                st.success("Transcription complete!")
                st.write(result["text"])
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.write("This is a detailed error message to help debug the issue.")
