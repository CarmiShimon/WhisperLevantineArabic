import glob
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperTokenizer
import torchaudio


def transcribe_audio(files_dir_path):
    """
    Transcribe an audio file using the Whisper model.
    Args:
        files_dir_path (str): The path to the audio files directory.
    """
    for file_path in glob.glob(files_dir_path + '/*.wav'):
        audio_input, samplerate = torchaudio.load(file_path)
        inputs = processor(audio_input.squeeze(), return_tensors="pt", sampling_rate=samplerate)
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"].to(device))
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print(transcription[0])


if __name__ == '__main__':
    wav_dir_path = '/home/user/Desktop/arb_stt/test/'
    checkpoint_path = '/home/user/Desktop/arb_stt/best_models/medium/checkpoint-3300'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and processor
    tokenizer = WhisperTokenizer.from_pretrained(f'{checkpoint_path}/tokenizer', language="Arabic", task="transcribe")
    processor = WhisperProcessor.from_pretrained(f'{checkpoint_path}/processor', language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.eval()

    transcribe_audio(wav_dir_path)