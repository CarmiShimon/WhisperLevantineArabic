import streamlit as st
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, WhisperTokenizer
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import sys


# Load pre-trained ASR model and processor
@st.cache_resource
def load_model(device, model_path):
    # Load Whisper model and processor
    # tokenizer = WhisperTokenizer.from_pretrained(f'{model_path}/tokenizer', language="Arabic", task="transcribe")
    processor = WhisperProcessor.from_pretrained(f'{model_path}/processor', language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.eval()
    # model_name = "openai/whisper-base"
    # processor = WhisperProcessor.from_pretrained(model_name)
    # model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def plot_waveform(waveform, sample_rate):
    fig, ax = plt.subplots()
    ax.set_title("Waveform")
    librosa.display.waveshow(waveform.squeeze().numpy(), sr=sample_rate, ax=ax)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(fig)

def plot_spectrogram(waveform, sample_rate, transcription):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform.squeeze().numpy())), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
    ax.set_title("Spectrogram with Transcription")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)
    st.write(f"### {transcription}")


def main():
    model_path = '/home/user/Desktop/arb_stt/best_models/medium/checkpoint-3300'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fs = 16000

    processor, model = load_model(device, model_path)

    # Streamlit app title
    st.title("Speech-to-Text Transcription for Levantine Arabic")
    # File upload or microphone input
    mode = st.radio("Select input mode:", ('Upload Audio File', 'Use Microphone'))
    if mode == 'Upload Audio File':
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

        if audio_file is not None:
            waveform, sample_rate = torchaudio.load(audio_file)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != fs:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=fs)
                waveform = resampler(waveform)
            plot_waveform(waveform, fs)
            inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=fs)
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            st.write("### Transcription:")
            st.write(transcription[0])


    elif mode == 'Use Microphone':
        duration = st.slider("Recording duration (seconds):", min_value=1, max_value=30, value=5)

        if st.button("Start Recording"):
            st.write("Recording...")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            st.write("Recording complete.")
            waveform = torch.tensor(recording).T
            plot_waveform(waveform, fs)
            inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=fs)
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"].to(device))
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            plot_spectrogram(waveform, fs, transcription)
            st.write("### Transcription:")
            st.write(transcription[0])

main()


