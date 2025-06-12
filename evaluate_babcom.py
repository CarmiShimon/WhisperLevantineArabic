import faster_whisper
import os
import re
import torch
import pandas as pd
from normalize import ArabicNormalizer, ArabicNormalizer2
import evaluate
import librosa
import glob
import csv
from datetime import timedelta

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
metric = evaluate.load("wer")
normalizer2 = ArabicNormalizer2()

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

def extract_text_from_tsv(tsv_file):
    # Load the TSV file
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
    full_text = ' '.join(df['text'].astype(str))
    return full_text


def clean_text(text):
    text = text.replace('.', '')
    text = text.replace(' اه ', ' ')
    text = text.replace(' اه ', ' ')
    text = text.replace(' ممم ', ' ')
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Normalize alef variants
    text = re.sub('[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = re.sub('ة', 'ه', text)
    # Normalize yaa and dots
    text = re.sub('[ىي]', 'ي', text)
    text = re.sub(r'[.،؟:;]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Remove exaggerated letters (e.g., "مممممم")
    text = text.strip()
    text_norm = ArabicNormalizer(text).normalize_all().text
    text_norm = normalizer2.basic_unicode_fix(text_norm)
    text_norm = normalizer2.normalize_letters(text_norm)
    text_norm = normalizer2.strip_diacritics_punct(text_norm)
    text_norm = "".join(normalizer2.token_filters(text_norm))
    return text_norm

audio_path = '/shares/rndsounds/wake_up_word/genAI/ac/stt/bab_test/audio'
# model = faster_whisper.WhisperModel('/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/ct2_v3')
model = faster_whisper.WhisperModel('/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/ct2_v3')
wer_data = []
with torch.no_grad():
    for audio_file in glob.glob(f'{audio_path}/*.wav'):
        gt_file = os.path.dirname(os.path.dirname(audio_file)) + '/transcripts/' + os.path.basename(audio_file)[:-4] + '.tsv'
        if os.path.isfile(gt_file):
            gt = extract_text_from_tsv(gt_file)
        audio_data, sample_rate = librosa.load(audio_file)

        audio_data = librosa.resample(
            audio_data, orig_sr=sample_rate, target_sr=16000
        )
        segments, info = model.transcribe(
            audio_data,
            language="ar",
            beam_size=5,
            vad_filter=True,
            task="transcribe",
            word_timestamps=True,
            chunk_length=30,
            vad_parameters=dict(min_silence_duration_ms=2000)
        )
        write_vtt(segments, f"{audio_file[:-4]}.vtt")
        # transcript = ' '.join(s.text for s in segments)
        # transcription_norm = clean_text(transcript)
        # gt_norm = clean_text(gt)
        # WER = metric.compute(predictions=[transcription_norm], references=[gt_norm])
        # wer_data.append((audio_file, WER))
        # print(f'WER per {audio_file} = {WER}')
        # with open(os.path.dirname(os.path.dirname(audio_file)) + f'/transcripts/transcript_wer_{WER}_' + os.path.basename(audio_file)[:-4] + '.txt', 'w', encoding='utf-8') as f:
        #     f.write(transcript)
        # with open(os.path.dirname(os.path.dirname(audio_file)) + '/transcripts/gt_' + os.path.basename(audio_file)[:-4] + '.txt', 'w', encoding='utf-8') as f:
        #     f.write(gt)
    #
    # csv_path = os.path.join('/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/ct2_v3', 'wer_results.csv')
    # with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['audio_file', 'WER'])
    #     writer.writerows(wer_data)


