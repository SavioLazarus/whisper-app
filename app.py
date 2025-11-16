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

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection with quality info
    model_options = ["tiny", "base", "small", "medium", "large"]
    model_info = {
        "tiny": "Fastest, lowest quality",
        "base": "Good balance of speed and quality",
        "small": "Better quality, slower",
        "medium": "High quality, much slower",
        "large": "Best quality, very slow"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=1,  # Default to "base" for better quality
        format_func=lambda x: f"{x} - {model_info[x]}"
    )
    
    # Language selection
    language = st.selectbox(
        "Language",
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic", "Hindi", "Portuguese", "Italian", "Dutch"],
        index=0
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Higher temperature makes output more random but may reduce accuracy"
        )
        
        best_of = st.slider(
            "Number of candidates",
            min_value=1,
            max_value=5,
            value=1,
            help="Generate multiple candidates and select the best one"
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
                # Load the selected model
                model = whisper.load_model(selected_model)
                
                # Read audio file using librosa
                audio_bytes = audio_file.read()
                audio_io = io.BytesIO(audio_bytes)
                audio, sr = librosa.load(audio_io, sr=16000)
                
                st.write(f"Audio loaded: {len(audio)} samples at {sr} Hz")
                
                # Transcribe with better settings
                with st.spinner("Transcribing..."):
                    options = {
                        "task": "transcribe",
                        "fp16": torch.cuda.is_available(),
                        "temperature": temperature,
                        "best_of": best_of
                    }
                    
                    # Add language if specified
                    if language != "Auto-detect":
                        options["language"] = language.lower()
                    
                    result = model.transcribe(audio, **options)
                
                # Show result
                st.success("Transcription complete!")
                
                # Display transcription with confidence
                if "text" in result:
                    st.subheader("Transcription:")
                    st.write(result["text"])
                    
                    # Show language info
                    if "language" in result:
                        st.info(f"Detected language: {result['language'].upper()}")
                
                # Show segments with timestamps if available
                if st.checkbox("Show detailed segments"):
                    if "segments" in result:
                        for i, segment in enumerate(result["segments"]):
                            start = segment["start"]
                            end = segment["end"]
                            text = segment["text"]
                            
                            # Show confidence if available
                            if "avg_logprob" in segment:
                                confidence = np.exp(segment["avg_logprob"]) * 100
                                st.write(f"[{start:.2f}s - {end:.2f}s] ({confidence:.1f}% confidence) {text}")
                            else:
                                st.write(f"[{start:.2f}s - {end:.2f}s] {text}")
                
                # Download button
                st.download_button(
                    label="Download transcription",
                    data=result["text"],
                    file_name=f"transcription_{audio_file.name}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.write("Debug info:")
                st.write(f"File type: {audio_file.type}")
                st.write(
