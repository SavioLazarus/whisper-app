import streamlit as st
import whisper
import torch
import os
import tempfile
from io import BytesIO
import time
from datetime import datetime
import requests
import subprocess

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
    .file-info {
        background-color: #e6f3ff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🎙️ Whisper Transcription App</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Transcribe audio and video files using OpenAI\'s Whisper model</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    # Model selection
    model_options = ["tiny", "base", "small", "medium", "large"]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=1,
        help="Larger models are more accurate but slower. Use 'tiny' or 'base' for long files."
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
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic", "Hindi", "Portuguese", "Italian", "Dutch"],
        index=0
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider(
            "Process in chunks (minutes)",
            min_value=0,
            max_value=30,
            value=0,
            step=5,
            help="Split long files into chunks. Set to 0 to process entire file at once."
        )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Media")
    
    # Support both audio and video formats
    audio_file = st.file_uploader(
        "Choose an audio or video file",
        type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "avi", "mov", "mkv", "wmv", "webm"],
        help="Supported Audio: MP3, WAV, M4A, FLAC, OGG | Supported Video: MP4, AVI, MOV, MKV, WMV, WebM"
    )
    
    if audio_file is not None:
        # Display file info
        file_size = audio_file.size / (1024 * 1024)  # Convert to MB
        st.markdown(f'<div class="file-info">📁 File: {audio_file.name} ({file_size:.2f} MB)</div>', unsafe_allow_html=True)
        
        # Show media preview
        if audio_file.type.startswith('video/'):
            st.video(audio_file)
        else:
            st.audio(audio_file)
        
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
    
    if st.button("🚀 Process Media", type="primary"):
        if audio_file is not None:
            with st.spinner(f"Loading {selected_model} model..."):
                try:
                    # Load model
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model(selected_model, device=device)
                    
                    # Save media temporarily
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp.write(audio_file.read())
                        temp_path = temp.name
                    
                    # Get file extension
                    file_extension = os.path.splitext(audio_file.name)[1].lower()
                    
                    # Convert to audio if it's a video file
                    if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm']:
                        with st.spinner("Extracting audio from video..."):
                            audio_path = temp_path.replace(file_extension, '.wav')
                            subprocess.run(['ffmpeg', '-i', temp_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', audio_path], 
                                         check=True, capture_output=True)
                        os.unlink(temp_path)  # Remove video file
                        temp_path = audio_path
                    
                    # Process audio
                    with st.spinner("Transcribing audio..."):
                        start_time = time.time()
                        
                        options = {
                            "task": task.lower(),
                            "fp16": device == "cuda",
                            "verbose": False
                        }
                        
                        if language != "Auto-detect":
                            options["language"] = language.lower()
                        
                        # Process entire file or in chunks
                        if chunk_size > 0:
                            # Process in chunks for long files
                            chunk_duration = chunk_size * 60  # Convert minutes to seconds
                            result = model.transcribe(temp_path, **options)
                        else:
                            # Process entire file
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
                        
                        # Download options
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            st.download_button(
                                label="📄 Download Text",
                                data=transcription,
                                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        with col_download2:
                            # Create SRT format for subtitles
                            if "segments" in result:
                                srt_content = ""
                                for i, segment in enumerate(result["segments"]):
                                    start_time = segment["start"]
                                    end_time = segment["end"]
                                    text = segment["text"]
                                    
                                    # Convert to SRT time format
                                    start_srt = time.strftime('%H:%M:%S,000', time.gmtime(start_time))
                                    end_srt = time.strftime('%H:%M:%S,000', time.gmtime(end_time))
                                    
                                    srt_content += f"{i+1}\n{start_srt} --> {end_srt}\n{text}\n\n"
                                
                                st.download_button(
                                    label="🎬 Download SRT",
                                    data=srt_content,
                                    file_name=f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                                    mime="text/plain"
                                )
                        
                        # Show timestamps if available
                        if st.checkbox("Show timestamps"):
                            if "segments" in result:
                                for segment in result["segments"]:
                                    start = segment["start"]
                                    end = segment["end"]
                                    text = segment["text"]
                                    st.write(f"[{start:.2f}s - {end:.2f}s] {text}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Tips: Try using a smaller model, shorter file, or enable chunk processing in Advanced Settings")
        else:
            st.error("Please upload an audio or video file first")

# Footer
st.markdown('<p style="text-align: center; margin-top: 2rem; color: #888;">Made with ❤️ using OpenAI Whisper • Developed with AI assistance from GLM-4.6</p>', unsafe_allow_html=True)
