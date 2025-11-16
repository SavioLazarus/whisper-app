import streamlit as st
import whisper
import torch
import io
import numpy as np

# Copyright (c) 2024 SavioLazarus
# Licensed under MIT License - see LICENSE file for details

st.title("🎙️ Whisper Transcription App")

audio_file = st.file_uploader("Upload audio", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Transcribe"):
        try:
            # Read the audio file
            audio_bytes = audio_file.read()
            
            # Load model
            model = whisper.load_model("tiny")
            
            # Create a temporary file in memory
            with io.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Transcribe
            result = model.transcribe(tmp_path)
            
            # Clean up
            import os
            os.unlink(tmp_path)
            
            st.success(result["text"])
            
        except Exception as e:
            st.error(f"Error: {e}")
