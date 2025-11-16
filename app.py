import streamlit as st
import whisper
import torch
import os
import tempfile
from io import BytesIO
import time
from datetime import datetime
import requests

# Copyright (c) 2024 SavioLazarus
# Licensed under MIT License - see LICENSE file for details

# Page configuration
st.set_page_config(
    page_title="Whisper Transcription App",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .transcription-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🎙️ Whisper Transcription App</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Transcribe audio files using OpenAI\'s Whisper model</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    # Model selection
    model_options = ["tiny", "base", "small", "medium", "large"]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    # Task selection
    task = st.radio(
        "Task",
        options=["Transcribe", "Translate"],
        help="Transcribe: Same language | Translate: To English"
    )
    
    # Language selection
    language = st.selectbox(
        "Language",
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic"],
        index=0
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Audio")
    
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        help="Supported: MP3, WAV, M4A, FLAC, OGG"
    )
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        
        # Example audio
        st.markdown("### Or Try Example")
        if st.button("Load English Example"):
            try:
                response = requests.get("https://storage.googleapis.com/aai-web-samples/english.mp3")
                audio_file = BytesIO(response.content)
                st.audio(audio_file, format="audio/mp3")
                st.success("Example loaded!")
            except:
                st.error("Could not load example")

with col2:
    st.markdown("### Results")
    
    if st.button("🚀 Process Audio", type="primary"):
        if audio_file is not None:
            with st.spinner(f"Loading {selected_model} model..."):
                try:
                    # Load model
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model(selected_model, device=device)
                    
                    # Save audio temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp:
                        temp.write(audio_file.read())
                        temp_path = temp.name
                    
                    # Process
                    with st.spinner("Processing audio..."):
                        start_time = time.time()
                        
                        options = {
                            "task": task.lower(),
                            "fp16": device == "cuda"
                        }
                        
                        if language != "Auto-detect":
                            options["language"] = language.lower()
                        
                        result = model.transcribe(temp_path, **options)
                        processing_time = time.time() - start_time
                        
                        # Clean up
                        os.unlink(temp_path)
                    
                    # Results
                    st.success(f"Done in {processing_time:.2f} seconds!")
                    
                    if "language" in result:
                        st.info(f"Detected: {result['language'].upper()}")
                    
                    if "text" in result:
                        transcription = result["text"]
                        st.markdown('<div class="transcription-box"><h3>Transcription</h3><p>' + transcription + '</p></div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            label="Download Text",
                            data=transcription,
                            file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                        if st.checkbox("Show timestamps"):
                            if "segments" in result:
                                for segment in result["segments"]:
                                    start = segment["start"]
                                    end = segment["end"]
                                    text = segment["text"]
                                    st.write(f"[{start:.2f}s - {end:.2f}s] {text}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try using a smaller model or shorter audio file")
        else:
            st.error("Please upload an audio file first")

# Footer
st.markdown('<p style="text-align: center; margin-top: 2rem; color: #888;">Made with ❤️ using OpenAI Whisper</p>', unsafe_allow_html=True)
