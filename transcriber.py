import os
import argparse
from datetime import timedelta

import librosa
import torch
from faster_whisper import WhisperModel


def seconds_to_timestamp(seconds):
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    t = timedelta(seconds=seconds)
    return str(t)[:-3].rjust(11, '0').replace('.', ',')


def write_vtt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_ts = seconds_to_timestamp(segment.start)
            end_ts = seconds_to_timestamp(segment.end)
            f.write(f"{start_ts} --> {end_ts}\n{segment.text}\n\n")


def transcribe_audio(model, audio_path, word_timestamps=True, vad_filter=True):
    print(f"\nProcessing {audio_path}...")
    with torch.no_grad():
        audio_data, sr = librosa.load(audio_path, sr=None)
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        segments, _ = model.transcribe(
            audio_data,
            language='ar',
            word_timestamps=word_timestamps,
            vad_filter=vad_filter
        )

        for segment in segments:
            if segment.words:
                for word in segment.words:
                    print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

        vtt_path = os.path.splitext(audio_path)[0] + ".vtt"
        write_vtt(segments, vtt_path)
        print(f"VTT written to: {vtt_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Faster-Whisper.")
    parser.add_argument("--model_path", required=True, help="Path to the model directory or file")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files (wav/mp3)")
    parser.add_argument("--word_timestamps", type=bool, default=True, help="Enable word timestamps (default: True)")
    parser.add_argument("--vad_filter", type=bool, default=True, help="Enable VAD filtering (default: True)")
    args = parser.parse_args()

    model = WhisperModel(args.model_path)

    for file in os.listdir(args.audio_dir):
        if file.endswith(".wav") or file.endswith(".mp3"):
            audio_path = os.path.join(args.audio_dir, file)
            transcribe_audio(
                model,
                audio_path,
                language="ar",
                beam_size=5,
                task="transcribe",
                word_timestamps=args.word_timestamps,
                vad_filter=args.vad_filter,
                vad_parameters=dict(min_silence_duration_ms=2000)
            )


if __name__ == "__main__":
    main()
