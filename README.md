# WhisperLevantineArabic
Fine-tuned Whisper model for the Levantine Dialect (Israeli-Arabic)
## Description
This repository contains a fine-tuned version of the Whisper medium model, specifically optimized for transcribing Levantine Arabic with a focus on the Israeli dialect. This model aims to improve automatic speech recognition (ASR) performance for this specific variant of Arabic.

## Model Details
- **Base Model**: Whisper Medium
- **Fine-tuned for**: Levantine Arabic (Israeli Dialect)
- **Performance Metrics**: [Provide key metrics, e.g., Word Error Rate (WER), on your test set]
- **Training Dataset**: 

## Dataset Description
This dataset contains transcribed audio data of spoken Levantine Arabic, with a focus on the Israeli dialect. It is designed to support research and development in speech recognition, linguistic analysis, and natural language processing for Levantine Arabic. The dataset comprises human-transcribed and annotated audio recordings, making it a valuable resource for training and evaluating speech recognition models and conducting linguistic studies.

## Dataset Composition
The dataset consists of three main components:

1. **Self-maintained Collection**: 2,000 hours of audio data, collected and maintained by our team. This forms the core of the dataset and represents a wide range of Israeli Levantine Arabic speech.

2. **Multi-Genre Broadcast (MGB-2)Filtered**: 200 hours of audio data sourced from the MGB-2 corpus, which includes broadcast news and other media content in Arabic.

3. **CommonVoice18 (Filtered)**: An additional portion of data from the CommonVoice18 dataset.
   Both MGB-2 and commonvoice18 filtered using [AlcLaM](https://arxiv.org/abs/2407.13097) (Arabic Language Model), ensuring relevance to Levantine Arabic.

- **Total Duration**: Approximately 2,200 hours of transcribed audio
- **Dialects**: Primarily Israeli Levantine Arabic, with some general Levantine Arabic content
- **Annotation**: Human-transcribed and annotated for high accuracy
- **Diverse Sources**: Includes self-collected data, broadcast media, and crowd-sourced content
- **Sampling Rate**: 16kHz

## Usage
[Provide instructions on how to use the model, including code snippets if possible]

```python
# Example code for using the model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("your-username/your-model-name")
processor = WhisperProcessor.from_pretrained("your-username/your-model-name")





