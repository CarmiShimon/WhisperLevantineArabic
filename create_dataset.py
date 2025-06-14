import json
from random import shuffle
import logging
from pathlib import Path
from typing import Iterator
import uuid
import glob
import subprocess
from tqdm import tqdm
from stable_whisper.result import WhisperResult, Segment
from stable_whisper.audio import AudioLoader
from audiosample import AudioSample
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    Audio as AudioColumnType,
    Value as ValueColumnType,
    Features,
)
from huggingface_hub import DatasetCard, DatasetCardData, upload_file

logger = logging.getLogger(__name__)


def _load_data_manifest(
        input_folder: str,
        audio_filename_glob: str,
        segments_glob: str,
        metadata_glob: str,
):

    segments_files = glob.glob(str(input_folder) + '/milestone*/stm_clean/*.json', recursive=True)  # [:100]
    audio_files = []
    metadata_files = []
    segments_files_clean = []
    for segments_file in segments_files:
        # find the audio file that has matches the glob and within the same

        segments_file = Path(segments_file)
        audio_dir = str(segments_file.parent.parent / 'audio')
        filename = segments_file.name
        audio_file = glob.glob(
            audio_dir + '/*/*' + filename[:-5] + '.wav')  # + glob.glob(audio_dir + '/*/*' + filename[:-5] + '.mp3')
        if not audio_file:
            continue
        if 'mp3' in audio_file[0]:
            continue
        segments_files_clean.append(segments_file)  # keep the segments file for later processing
        found_audio_files = [Path(audio_file[0])]
        # expect only one audio file
        assert (
                len(found_audio_files) == 1
        ), f"Expected 1 audio file, found {len(found_audio_files)} for {segments_file} (taking first)"
        audio_files.extend(found_audio_files)
        # print(segments_file)
        # print(audio_file)
        # expect only one metadata file
        metadata_files_ = glob.glob(audio_dir + '/*' + 'metadata.json')
        # print(metadata_files_)
        found_metadata_files = [Path(metadata_files_[0])]
        assert (
                len(found_metadata_files) == 1
        ), f"Expected 1 metadata file, found {len(found_metadata_files)} for {segments_file} (taking first)"
        metadata_files.extend(found_metadata_files[:1])
    segments_files = [Path(p) for p in segments_files_clean]
    return list(zip(audio_files, segments_files, metadata_files))


from subprocess import CalledProcessError, run, PIPE, DEVNULL

import numpy as np

WHISPER_EXPECTED_SAMPLE_RATE = 16000


def load_audio_with_ffmpeg(file: str, sr: int = 16000):
    cmd = [
        "ffmpeg",
        "-v", "error",             # Hide banner, show only errors
        "-nostdin",
        "-threads", "1",           # Use 1 thread to reduce concurrency issues
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "pipe:1"                   # Explicitly write to stdout
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Larger buffer can help avoid broken pipes
        )
        out, err = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{err.decode()}")

        audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    except Exception as e:
        raise RuntimeError(f"Audio loading failed: {e}")

def get_segment_word_scores(segment: Segment) -> list[float]:
    """
    Get the word scores for a segment.
    This is a helper function to extract the word scores from a segment.
    """
    if not segment.has_words:
        return []

    # Extract word scores from the segment
    word_scores = []
    for word in segment.words:
        if hasattr(word, "probability"):
            word_scores.append(word.probability)
    return word_scores


def calculate_median_quality_score(scores: list[float]) -> float:
    """
    Calculate the median quality score for a list of scores.
    This is a helper function to calculate the median quality score for a list of scores.
    """
    # Calculate the median probability of all words in the segment
    quality_score = float(np.median(scores)) if scores else 0.0
    return quality_score


def calculate_segments_quality_score(segments: list[Segment]) -> float:
    if not segments:
        return 0.0

    """Calculate the quality score based on the median word probabilities for a single segment."""
    try:
        all_word_probs = []
        for segment in segments:
            all_word_probs.extend(get_segment_word_scores(segment))
        # Calculate the median probability of all words in the segment
        quality_score = calculate_median_quality_score(all_word_probs)
        return quality_score

    except Exception:
        return 0.0


def generate_slices(
        input_segments: list[Segment],
        audio_duration: float,
        slice_length: int,
        per_segment_quality_threshold: float = 0,
):
    next_slice_start = 0
    curr_input_segment_idx = 0
    slices = []
    while next_slice_start < audio_duration:
        slice_start = next_slice_start

        # Ensure current segment exists
        # and validate it's duration.
        if curr_input_segment_idx < len(input_segments):
            curr_input_segment_duration = (
                    input_segments[curr_input_segment_idx].end - input_segments[curr_input_segment_idx].start
            )
            # If the first segment to work on is too long for a single slice or of 0 length we must skip it.
            if curr_input_segment_duration > slice_length or curr_input_segment_duration == 0:
                # skip if any segment ahead
                if curr_input_segment_idx + 1 < len(input_segments):
                    next_slice_start = input_segments[curr_input_segment_idx + 1].start
                    curr_input_segment_idx += 1
                # or break since nothing more to work on
                else:
                    next_slice_start = audio_duration

                continue

        curr_slice_source_segment_idxs = []
        curr_slice_source_segments = []
        curr_slice_segments = []
        curr_slice = {"segments": curr_slice_segments, "seek": slice_start}
        slices.append(curr_slice)
        # normal slice length is the expected slice hop - but this could be overridden below. See comments.
        next_slice_start = slice_start + slice_length
        # clip the slice end to the total audio duration
        slice_end = min(next_slice_start, audio_duration)

        # While more segments to work on and the current segment start is within the slice
        while curr_input_segment_idx < len(input_segments) and input_segments[curr_input_segment_idx].start < slice_end:
            curr_input_segment = input_segments[curr_input_segment_idx]

            # track the source segments used in this slice for quality analysis after slice completion
            curr_slice_source_segments.append(curr_input_segment)
            curr_slice_source_segment_idxs.append(curr_input_segment_idx)

            # Add it to the slice
            slice_segment = {
                "start": max(0, curr_input_segment.start - slice_start),  # relative to slice
            }
            curr_slice_segments.append(slice_segment)

            # Clip the segment end to the entire audio duration
            # This is meant to prevent small segment timing overflows over audio
            # duration which stems from arbitrary rounding errors in the data prep
            # and subtitles alignment logic.
            curr_input_segment_end = min(curr_input_segment.end, audio_duration)

            # If this input segment ends within the slice
            # It would be entirely contained including it's text and timestamps
            if curr_input_segment_end <= slice_end:
                #   s   e   s         e
                #  /    \  /          \??????
                # |_________________________|
                #                     ^
                slice_segment["end"] = min(slice_length, curr_input_segment_end - slice_start)  # relative to slice
                slice_segment["text"] = curr_input_segment.text
                slice_segment["word_scores"] = get_segment_word_scores(curr_input_segment)

                # entire segment is included - no need to reference it again on the next slice.
                curr_input_segment_idx += 1

            # Else - we cannot complete this segment on this slice.
            # The "start" of the segment is kept in the slice to mark it's crossing onto the next
            # slice but the next slice will also need to start at the **end** of the previous segment
            # to allow proper "restart" of the overflowing segment
            else:
                # This slice ends - close this slice

                # remove the last added source segment - it was not used
                curr_slice_source_segments = curr_slice_source_segments[:-1]
                curr_slice_source_segment_idxs = curr_slice_source_segment_idxs[:-1]

                # Special case - If the "start only" segment is the only one - don't include it at all.
                # Instead, this slice would be left empty.
                if len(curr_slice_segments) == 1:
                    #           s                    e
                    #          /                     \
                    # |_________________________||........
                    #                                ^
                    curr_slice_segments.clear()

                    # In this special case, the current segment starts within
                    # the slice and ends outside of it. But it is the only segment.
                    # We need to start the next slice on the **start** of this segment
                    # and not at the end of the previous one (which is not within this slice
                    # at all
                    next_slice_start = input_segments[curr_input_segment_idx].start
                else:
                    #   s    e  s                    e
                    #  /     \ /                     \
                    # |_________________________||........
                    #                                ^
                    # This is the normal cross-over case.
                    # The current segment starts within this slice
                    # and ends outside of it and other segments within this slice were closed normally.
                    # We need to start the next slice on the **end** of prev segment before the "start-only" one.
                    next_slice_start = input_segments[curr_input_segment_idx - 1].end

                # Break, this slice is done.
                break

        # Slice Quality Control
        slice_quality_score = calculate_segments_quality_score(curr_slice_source_segments)

        # Check if the slice quality is below threshold to abandon it and force a new slice
        if curr_slice_source_segments and slice_quality_score < per_segment_quality_threshold:
            # This slice is suspected as low quality

            # Look for a segment with good quality to start the next slice
            # skip the first segment in the slice (otherwise we probably are going
            # to just repeat the same slice)
            found_good_segment = False
            for seg_idx_within_slice, seg_of_slice in enumerate(curr_slice_source_segments):
                if seg_idx_within_slice == 0:
                    continue

                segment_score = calculate_segments_quality_score([seg_of_slice])

                if segment_score >= per_segment_quality_threshold:
                    # Found a good enough segment, start next slice from here
                    next_slice_start = seg_of_slice.start
                    curr_input_segment_idx = curr_slice_source_segment_idxs[seg_idx_within_slice]
                    found_good_segment = True
                    break

            # If no good segment found, start from the end of the last checked segment
            if not found_good_segment:
                next_segment_idx_after_slice_segments = curr_slice_source_segment_idxs[-1] + 1
                # if any segment ahead
                if next_segment_idx_after_slice_segments < len(input_segments):
                    next_slice_start = input_segments[next_segment_idx_after_slice_segments].start
                    curr_input_segment_idx = next_segment_idx_after_slice_segments
                # or there are more segments - stop slicing
                else:
                    next_slice_start = audio_duration

            # Clear the current slice content as we're abandoning it
            curr_slice_segments.clear()

    return slices


def merge_slice_segments(slices: list[dict], merge_below_gap_threshold: float = 0.3) -> list[dict]:
    """
    Merge segments within each slice that are close together.

    Args:
        slices: List of slices, each containing a list of segments
        merge_below_gap_threshold: Merge segments if gap between them is less than this threshold (in seconds)

    Returns:
        List of slices with merged segments
    """
    if not slices:
        return slices

    result_slices = []

    for slice_data in slices:
        # Create a new slice with the same properties as the original, but copy it to avoid modifying the original
        new_slice = {key: value for key, value in slice_data.items() if key != "segments"}
        new_slice["segments"] = []

        segments = slice_data.get("segments", [])

        # If no segments or only one segment, no merging needed
        if len(segments) <= 1:
            new_slice["segments"] = [segment.copy() for segment in segments]
            result_slices.append(new_slice)
            continue

        # Create a copy of segments to process
        result_segments = [segment.copy() for segment in segments]

        # Process segments in reverse order
        i = len(result_segments) - 1
        while i > 0:  # Stop at index 1 (second segment)
            current_segment = result_segments[i]
            prev_segment = result_segments[i - 1]

            # Check if we can merge the current segment with the previous one
            can_merge = False

            # Current segment must have start, end, and text to be mergeable
            # Note: No "end" cases means an open-only slice where the last segment
            # mark a segment which could not end within the same slice. we need
            # to keep it as is.
            if all(key in current_segment for key in ["start", "end", "text"]):

                # Calculate the gap between segments
                gap = current_segment["start"] - prev_segment["end"]

                # Check if the gap is small enough
                if gap < merge_below_gap_threshold:
                    can_merge = True

            if can_merge:
                # Merge current segment into previous segment
                prev_segment["end"] = current_segment["end"]
                prev_segment["text"] = prev_segment["text"] + current_segment["text"]

                # Remove the current segment as it's now merged
                result_segments.pop(i)

            # Move to previous segment
            i -= 1

        # Add all processed segments to the new slice
        new_slice["segments"] = result_segments
        result_slices.append(new_slice)

    return result_slices

import soundfile as sf
import io

def get_slice_audio_data(audio_loader: AudioLoader, slice, slice_length):
    audio_start_sec = slice["seek"]
    seek_sample = int(audio_start_sec * WHISPER_EXPECTED_SAMPLE_RATE)
    slice_length_samples = int(slice_length * WHISPER_EXPECTED_SAMPLE_RATE)
    audio_data = audio_loader.next_chunk(seek_sample, slice_length_samples)
    # slice_audio_data_as_mp3 = AudioSample(
    #     audio_data.numpy(), force_read_format="s16le", force_read_sample_rate=WHISPER_EXPECTED_SAMPLE_RATE
    # ).as_data(no_encode=False, force_out_format="wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data.numpy(), samplerate=WHISPER_EXPECTED_SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return buffer.getvalue()
    # return slice_audio_data_as_mp3


def get_timestamp_token_text(seconds: float) -> str:
    """
    Get the timestamp token text for a given seconds.
    This is a helper function to encode the timestamp tokens for the Whisper model.
    It is specific to Whisper and should be moved to a proper util that handles
    timestamp tokens encoding/decoding for any ASR model.
    """
    if 0 <= seconds <= 30:
        # round to precision of .02
        rounded = 0.02 * round(seconds / 0.02)
        return f"<|{rounded:.2f}|>"
    else:
        raise ValueError("Timestamp token out of range.")


def generate_examples_from_slices(
        slices, slice_length, audio_loader, metadata: dict, copy_metadata_fields: list[str] = []
) -> Iterator[dict]:
    source_id = metadata.get("source_id", "unknown")
    source_entry_id = metadata.get("source_entry_id", str(uuid.uuid4()))
    logger.debug(f"Generating dataset from {source_id}/{source_entry_id}")

    # No slices - nothing to do
    if not slices:
        logger.debug(f"No slices in {source_id}/{source_entry_id}")
        return None

    # At least one segments we can work on is expected
    if next(iter([seg for s in slices for seg in s["segments"]]), None) is None:
        logger.debug(f"No segments in {source_id}/{source_entry_id}")
        return None

    prev_example = None
    # correct end segments
    for slice in slices:
        segments = slice.get("segments", [])
        for seg in segments:
            if "end" in seg and isinstance(seg["end"], (int, float)):
                new_end = seg["end"] - 0.15
                # clamp at zero so you don’t get negative times
                seg["end"] = max(new_end, 0.0)

    for slice in slices:
        if slice["segments"]:
            try:
                slice_text = ""
                for segment in slice["segments"]:
                    slice_text += get_timestamp_token_text(segment["start"])
                    if "text" in segment:
                        slice_text += f'{segment["text"]}{get_timestamp_token_text(segment["end"])}'
                all_word_scores = [score for segment in slice["segments"] for score in segment.get("word_scores", [])]
                if '-100' in slice_text:
                    continue
                segments_quality_score = calculate_median_quality_score(all_word_scores)
                slice_audio_data = get_slice_audio_data(audio_loader, slice, slice_length)
                example = {
                    "audio": {
                        "bytes": slice_audio_data,
                        "path": source_entry_id,
                    },
                    "transcript": slice_text,
                    "metadata": {
                        "seek": float(slice["seek"]),
                        "duration": slice_length,
                        "source": source_id,
                        "entry_id": source_entry_id,
                        "quality_score": segments_quality_score,
                    },
                    "has_prev": False,
                    "has_timestamps": True,
                    "prev_transcript": "",
                }
                if prev_example:
                    example["prev_transcript"] = prev_example["transcript"]
                    example["has_prev"] = True
                if copy_metadata_fields:
                    for field_to_copy in copy_metadata_fields:
                        example["metadata"][field_to_copy] = metadata.get(field_to_copy, None)
                yield example
                prev_example = example
            except Exception as e:
                logger.error(
                    f'Error processing slice seek {float(slice["seek"]):.2f} in {source_id}:{source_entry_id}: {e}'
                )
        else:
            prev_example = None
    logger.debug(f"Done with samples from {source_id}/{source_entry_id}")


def prepare_training_dataset(
        input_folder: Path,
        slice_length: int = 30,
        max_source_entries: int = None,
        audio_filename_glob: str = "audio.*",
        segments_filename_glob: str = "transcript.*.json",
        metadata_glob: str = "metadata.json",
        num_proc: int = 1,
        per_proc_per_chunk_size: int = 10,
        per_sample_quality_threshold: float = 0,
        per_segment_quality_threshold: float = 0,
        copy_metadata_fields: list[str] = [],
) -> Dataset:
    """
    Prepare captioned datasets from the input folder.
    Produce audio slices and corresponding text including previous text when available
    Returns a HuggingFace Dataset. Splitting (if needed) should be applied outside this function.
    """
    input_folder = Path(input_folder)
    input_manifest = _load_data_manifest(
        input_folder,
        segments_glob=f"**/{segments_filename_glob}",
        audio_filename_glob=audio_filename_glob,
        metadata_glob=metadata_glob,
    )

    # Shuffle source entries to approach more homogenous shard sizes
    # shuffle(input_manifest)

    # Limit the number of source entries to process
    if max_source_entries:
        input_manifest = input_manifest[:max_source_entries]

    # Aim for reasonable entries per worker within each chunk
    manifest_processing_chunk_size = num_proc * per_proc_per_chunk_size

    def examples_from_entry_generator(input_manifest_shards):
        for audio_file, segments_data_file, metadata_file in input_manifest_shards:
            try:
                # Load captions
                segments_data = WhisperResult(str(segments_data_file))
                segments = segments_data.segments

                # Load metadata
                sample_quality_score = None
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    sample_quality_score = metadata.get("quality_score", None)

                if (
                        sample_quality_score is not None
                        and per_sample_quality_threshold > 0
                        and sample_quality_score < per_sample_quality_threshold
                ):
                    logger.debug(
                        f"Skipping sample {audio_file} with quality score {sample_quality_score} (threshold: {per_sample_quality_threshold})"
                    )
                    continue

                # Load Audio (streams output from an FFMPEG process for memory efficiency)
                audio_loader = AudioLoader(
                    str(audio_file),
                    stream=True,
                    sr=WHISPER_EXPECTED_SAMPLE_RATE,
                    buffer_size=int(3 * slice_length * WHISPER_EXPECTED_SAMPLE_RATE),
                )
                try:
                    audio_duration = audio_loader.get_duration()

                    # Create slices of the captions with the intended slice
                    slices = generate_slices(segments, audio_duration, slice_length, per_segment_quality_threshold)
                    slices = merge_slice_segments(slices)

                    # Generate the dataset
                    for example in generate_examples_from_slices(
                            slices,
                            slice_length,
                            audio_loader,
                            metadata,
                            copy_metadata_fields,
                    ):
                        yield example
                finally:
                    audio_loader.terminate()
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")

    input_manifest_chunks = [
        input_manifest[i: i + manifest_processing_chunk_size]
        for i in range(0, len(input_manifest), manifest_processing_chunk_size)
    ]

    # Why? Dataset.from_generator does not properly release memory from the generator
    # after completion. To avoid OOM, we:
    # 1. Generate multiple smaller datasets in chunks
    # 2. Let each chunk's generator get GC'd after completion
    # 3. Concatenate the memory-mapped datasets at the end
    # This trades off some disk I/O for better memory usage, while still
    # maintaining parallel generation within each chunk.
    all_datasets = []
    for input_manifest_chunk in tqdm(input_manifest_chunks, desc="Generating input manifest chunks"):
        try:
            dataset_features = Features(
                {
                    "audio": AudioColumnType(),
                    "transcript": ValueColumnType(dtype="string"),
                    "metadata": {
                        "seek": ValueColumnType(dtype="float32"),
                        "duration": ValueColumnType(dtype="float32"),
                        "source": ValueColumnType(dtype="string"),
                        "entry_id": ValueColumnType(dtype="string"),
                        "quality_score": ValueColumnType(dtype="float32"),
                    },
                    "has_prev": ValueColumnType(dtype="bool"),
                    "has_timestamps": ValueColumnType(dtype="bool"),
                    "prev_transcript": ValueColumnType(dtype="string"),
                }
            )
            if copy_metadata_fields:
                for field_to_copy in copy_metadata_fields:
                    dataset_features["metadata"][field_to_copy] = ValueColumnType(dtype="string")
            generated_dataset = Dataset.from_generator(
                examples_from_entry_generator,
                num_proc=num_proc,
                gen_kwargs={"input_manifest_shards": list(input_manifest_chunk)},
                features=dataset_features,
            )
        except ValueError as e:
            if "corresponds to no data" in str(e):
                print("Skipping dataset creation because no data was found.")
                continue
            else:
                raise  # Re-raise unexpected errors

        all_datasets.append(generated_dataset)

    if not all_datasets:
        return None

    examples_dataset = concatenate_datasets(all_datasets)
    examples_dataset = examples_dataset.cast_column(
        "audio", AudioColumnType(sampling_rate=WHISPER_EXPECTED_SAMPLE_RATE)
    )

    return examples_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI to prepare a training dataset from the input folder")
    # parser.add_argument(
    #     "input_folder",
    #     help="Path to the folder containing audio, transcript, and metadata data in the normalized structure",
    # )
    parser.add_argument(
        "--max_source_entries", type=int, default=None, help="Maximum number of source entries to process"
    )
    parser.add_argument("--audio_filename_glob", default="audio.*", help="Glob pattern for audio files")
    parser.add_argument("--segments_filename_glob", default="transcript.*.json", help="Glob pattern for segments files")
    parser.add_argument(
        "--validation_split_size", type=float, default=0, help="Split size for evaluation (between 0 and 1)"
    )
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use")
    parser.add_argument(
        "--per_proc_per_chunk_size",
        type=int,
        default=10,
        help=(
            "Number of entries per process per chunk. "
            "This is a memory usage consideration. This number times the number of processes will define the "
            "amount of memory kept around during the generation of a sub-dataset. "
            "If each sample is large (minutes of audio), this number should be decreased. "
            "If each sample is small (seconds of audio), this number can be increased to increase parallelism efficiency. "
        ),
    )
    parser.add_argument(
        "--per_sample_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-sample quality filtering (0-1 below this threshold the entire sample is dropped)",
    )
    parser.add_argument(
        "--per_segment_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-segment quality filtering (0-1 below this threshold a segment and it's surrounding slice are dropped)",
    )
    parser.add_argument(
        "--copy_metadata_fields",
        nargs="*",
        default=[],
        help="specify dataset specific metadata fields to copy into output segments from souce entries",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to the hub")
    parser.add_argument(
        "--output_dataset_name", type=str, help="Name of the dataset, Omit to not store any dataset (dry-run)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to set in dataset info",
    )
    parser.add_argument(
        "--dataset_license_file",
        type=str,
        help="A license file to upload as the dataset license",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Version of the dataset to set in dataset info",
    )
    parser.add_argument("--dataset_card_language", type=str, help="Language of the dataset for the dataset card")
    parser.add_argument("--dataset_card_license", type=str, help="License of the dataset for the dataset card")
    parser.add_argument(
        "--dataset_card_language_creators", type=str, nargs="+", help="Language creators type for the dataset card"
    )
    parser.add_argument(
        "--dataset_card_task_categories", type=str, nargs="+", help="Task categories for the dataset card"
    )
    parser.add_argument("--dataset_card_pretty_name", type=str, help="Pretty name for the dataset card")
    parser.add_argument("--push_as_public", action="store_true", help="Push the dataset as public")
    parser.add_argument(
        "--clear_output_dataset_cache_files",
        action="store_true",
        help="Clear the HF cache for the output dataset on disk",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level of the general logger.")

    args = parser.parse_args()
    args.input_folder = '/shares/rndsounds/wake_up_word/genAI/ac/stt'
    args.dataset_card_language = "Arabic"
    args.dataset_card_pretty_name = "Verbit Arabic STT Dataset"
    args.output_dataset_name = "verbit"
    args.validation_split_size = 0.01
    args.num_proc = 32
    # Configure Logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    else:
        logger.setLevel(level=numeric_level)

    # Prepare the dataset
    output_dataset = prepare_training_dataset(
        input_folder=args.input_folder,
        max_source_entries=args.max_source_entries,
        audio_filename_glob=args.audio_filename_glob,
        segments_filename_glob=args.segments_filename_glob,
        num_proc=args.num_proc,
        per_proc_per_chunk_size=args.per_proc_per_chunk_size,
        per_sample_quality_threshold=args.per_sample_quality_threshold,
        per_segment_quality_threshold=args.per_segment_quality_threshold,
        copy_metadata_fields=args.copy_metadata_fields,
    )

    if output_dataset:
        if args.dataset_name:
            output_dataset.info.dataset_name = args.dataset_name
        if args.dataset_version:
            output_dataset.info.version = args.dataset_version

        # Create dataset card if any of the card-related arguments are provided
        dataset_card = None
        if any(
                [
                    args.dataset_card_language,
                    args.dataset_card_license,
                    args.dataset_card_language_creators,
                    args.dataset_card_task_categories,
                    args.dataset_card_pretty_name,
                ]
        ):
            card_data = DatasetCardData(
                language=args.dataset_card_language,
                license=args.dataset_card_license,
                language_creators=args.dataset_card_language_creators,
                task_categories=args.dataset_card_task_categories,
                pretty_name=args.dataset_card_pretty_name,
            )
            dataset_card = DatasetCard.from_template(card_data, template_path="assets/ivritai_dataset_card_template.md")

        if args.validation_split_size > 0:
            # If a validation split is requested, split the dataset in main
            assert args.validation_split_size < 1.0, "validation_split_size must be a float between 0 and 1"
            temp = output_dataset.train_test_split(test_size=args.validation_split_size)
            output_dataset = DatasetDict({"train": temp["train"], "eval": temp["test"]})

        if args.output_dataset_name:
            if args.push_to_hub:
                if not args.push_as_public:
                    logger.warning("Pushing the dataset to the hub as private")
                output_dataset.push_to_hub(args.output_dataset_name, private=not args.push_as_public)
                # Push dataset card if it was created
                if dataset_card:
                    dataset_card.push_to_hub(repo_id=args.output_dataset_name, repo_type="dataset")

                if args.dataset_license_file and Path(args.dataset_license_file).exists():
                    upload_file(
                        path_or_fileobj=args.dataset_license_file,
                        repo_id=args.output_dataset_name,
                        path_in_repo="LICENSE",
                        repo_type="dataset",
                    )
            else:
                output_dataset.save_to_disk(args.output_dataset_name)
                # Save dataset card if it was created
                if dataset_card:
                    logger.warning("Dataset card will be saved locally since push_to_hub is not enabled")
                    dataset_card.save(f"{args.output_dataset_name}/README.md")

            # report the created dataset sizes per split
            if isinstance(output_dataset, DatasetDict):
                for split, ds in output_dataset.items():
                    logger.info(f"{split}: {ds.num_rows} samples")
            else:
                logger.info(f"Dataset created with {output_dataset.num_rows} samples")

        if args.clear_output_dataset_cache_files and output_dataset:
            logger.info("Clearing output dataset cache files")
            output_dataset.cleanup_cache_files()
    else:
        logger.warning("No dataset was created")