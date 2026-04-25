import torch
from transformers import pipeline
import librosa
import numpy as np
import os
import gradio as gr

# --- 1. CONFIGURATION (Core logic preserved) ---
try:
    print("✅ Loading libraries and model...")
    # Detect GPU (CUDA) or fallback to CPU
    device = 0 if torch.cuda.is_available() else -1 
    
    transcriber = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny", 
        device=device
    )
    print(f"🚀 Model loaded successfully on: {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"❌ Initialization error: {e}")

# --- 2. PROCESSING FUNCTION ---
def process_audio_interface(audio_input):
    if audio_input is None:
        return "Please upload an audio file first."
    
    try:
        # Load audio at 16000Hz as required by Whisper
        audio_data, sr = librosa.load(audio_input, sr=16000)
        
        # Long-form processing: chunking into 30s segments for stability
        print("🎙️ Transcribing audio... please wait.")
        result = transcriber(audio_data, chunk_length_s=30, return_timestamps=True)
        
        return result["text"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- 3. GRAPHICAL INTERFACE (Gradio 6.0+) ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Professional AI Audio Transcriber")
    gr.Markdown("### Instantly convert speech to text using state-of-the-art Whisper AI.")
    
    with gr.Row():
        # Main input/output column
        with gr.Column(scale=2):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (.mp3, .wav, .m4a)")
            transcribe_btn = gr.Button("🚀 Start Transcription", variant="primary")
            output_text = gr.Textbox(label="Transcription Result", lines=12, placeholder="Your transcribed text will appear here...")
        
        # Technical specifications and limits column
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Specifications & Limits")
            gr.Markdown("""
            **Service Limits:**
            * **Max File Size:** 500MB (Industry Standard).
            * **Length:** Unlimited (files over 30 min may take longer to process).
            * **Formats:** MP3, WAV, M4A, AAC, OGG.
            
            **Best Practices:**
            1. Use audio with clear speech and low background noise.
            2. For very long files, stay on the page while the AI segments the audio.
            3. If the process times out, try refreshing or using a smaller file.
            """)

    # Connecting the button to the logic
    transcribe_btn.click(
        fn=process_audio_interface,
        inputs=audio_input,
        outputs=output_text
    )

# --- 4. LAUNCH ---
if __name__ == "__main__":
    # Theme is applied here in Gradio 6.0
    # Use share=True to generate a public link if running locally
    demo.launch(theme=gr.themes.Soft(), share=True)