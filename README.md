# WhisperLevantineArabic
Fine-tuned Whisper model for the Levantine Dialect (Israeli-Arabic)
## Description
This repository contains a fine-tuned version of the Whisper medium model, specifically optimized for transcribing Levantine Arabic with a focus on the Israeli dialect. This model aims to improve automatic speech recognition (ASR) performance for this specific variant of Arabic.

## Model Details
- **Base Model**: [Whisper Medium](https://github.com/openai/whisper)
- **Fine-tuned for**: Levantine Arabic (Israeli Dialect)
- **Performance Metrics**: 10% WER on test-set

## Dataset Description
This dataset contains transcribed audio data of spoken Levantine Arabic, with a focus on the Israeli dialect. It is designed to support research and development in speech recognition, linguistic analysis, and natural language processing for Levantine Arabic. The dataset comprises human-transcribed and annotated audio recordings, making it a valuable resource for training and evaluating speech recognition models and conducting linguistic studies.

## Dataset Composition
The dataset consists of three main components:

1. **Self-maintained Collection**: 2,000 hours of audio data, collected and maintained by our team. This forms the core of the dataset and represents a wide range of Israeli Levantine Arabic speech.

2. **Multi-Genre Broadcast [(MGB-2)](https://huggingface.co/datasets/BelalElhossany/mgb2_audios_transcriptions_preprocessed)-Filtered**: 200 hours of audio data sourced from the MGB-2 corpus, which includes broadcast news and other media content in Arabic.

3. **[CommonVoice18](https://huggingface.co/datasets/fsicoli/common_voice_18_0) (Filtered)**: An additional portion of data from the CommonVoice18 dataset.
   ##### Both MGB-2 and commonvoice18 filtered using [AlcLaM](https://arxiv.org/abs/2407.13097) (Arabic Language Model), ensuring relevance to Levantine Arabic.

- **Total Duration**: Approximately 2,200 hours of transcribed audio
- **Dialects**: Primarily Israeli Levantine Arabic, with some general Levantine Arabic content
- **Annotation**: Human-transcribed and annotated for high accuracy
- **Diverse Sources**: Includes self-collected data, broadcast media, and crowd-sourced content
- **Sampling Rate**: 16kHz

## Usage
The model was trained using a 16kHz sample rate, so ensure your audio files are also at 16kHz for optimal performance.

```python
# Example code for using the model
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





