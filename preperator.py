import os
import numpy as np
import torch
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_from_disk,
)
from scipy.stats import beta
from torchaudio.transforms import Resample
from transformers import BatchFeature, WhisperProcessor

# from preprocess.augmentation import shift_audio_forward

# This is defined as part of the model config
# and should match the loaded model.
# For all whisper models so far this is the same value
# Ideally we will take this from the WhisperConfig but we need to
# Process the dataset before loading the model in some use cases.
whisper_max_target_positions = 448


class DatasetPreparator:
    """
    This class is responsible for preparing the dataset for training.
    It will:
    - Resample the audio to the target sampling rate if needed
    - Extract audio features from the audio
    - If audio was padded, store the padding in a less wasteful format
    - If requested, augment the audio with a random shift forward
    - Randomly decide whether to include timestamps with a sample
    - Randomly decide whether to include previous text with a sample
    - If timestamps are not included, inject a start-end timestamp token pair according to duration and shift augmentation, if injection was enabled
    - If timestamps are included, remove them in case decision was to not train on them for that sample
    - Tokenize the prev_text and text prefixing with the proper Whisper prefix tokens
    - Return the prepared example as a BatchFeature object dropping original dataset columns
    - If the features "has_timestamps" and "has_prev" are not present, assume no timestamps and no previous text

    Notes:
    - This process runs nicely in parallel (use proc_num > 1) but not when the device is set to GPU
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        # Don't change this for Whisper
        tokenizer_time_precision=0.02,
        # Below can be adapted
        timestamp_sample_prob=0.5,
        condition_on_prev_sample_prob=0.5,
        proc_num: int = 1,
        # Multi worker only supports cpu atm (crashes on cuda)
        device: str = "cpu",
        seed: np.random.RandomState = None,
        # Experimental
        inject_synthetic_timestamps=False,
        audio_shift_augmentation=False,
    ):
        if proc_num > 1:  # Parallel processing will not work in multi threaded env.
            torch.set_num_threads(1)

        # Stability of the seed will allow reusing cached processing
        self.seed = np.random.default_rng(998) if seed is None else np.random.default_rng(seed)
        self.device = device
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.proc_num = proc_num
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.tokenizer_time_precision = tokenizer_time_precision

        orig_predict_timestamps = self.processor.tokenizer.predict_timestamps
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=False)
        self.prefix_tokens_no_ts = self.processor.tokenizer.prefix_tokens
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=True)
        self.prefix_tokens_with_ts = self.processor.tokenizer.prefix_tokens
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=orig_predict_timestamps)

        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.start_of_prev_token_id = self.tokenizer.convert_tokens_to_ids("<|startofprev|>")
        self.no_timestamp_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.timestamp_begin_token_id = self.no_timestamp_token_id + 1
        self.last_timestamp_token = self.tokenizer.total_vocab_size - 1
        self.total_timestamp_tokens = self.last_timestamp_token - self.timestamp_begin_token_id + 1
        self.max_allowed_tokenized_timestamp = (self.total_timestamp_tokens - 1) * self.tokenizer_time_precision
        self.prev_ids_max_length = whisper_max_target_positions // 2
        self.timestamp_sample_prob = timestamp_sample_prob
        self.condition_on_prev_sample_prob = condition_on_prev_sample_prob
        self.inject_synthetic_timestamps = inject_synthetic_timestamps
        self.audio_shift_augmentation = audio_shift_augmentation
        self.max_shifted_audio_ends_at = 29.6

        # Prepare the output features - to ensure optimal storage during mapping (uses disk cache for mapped content)
        self.output_features = Features(
            {
                "input_features": Sequence(feature=Sequence(feature=Value(dtype="float32"))),
                "labels": Sequence(feature=Value(dtype="int32")),
                "pad_value": Sequence(feature=Value(dtype="float32")),
                "pad_amount": Value("int32"),
                "input_length": Value("float64"),
            }
        )

    def _validate_dataset_features(self, dataset):
        if "audio" not in dataset.features:
            raise ValueError("Dataset must contain an 'audio' feature")
        elif not isinstance(dataset.features["audio"], Audio):
            raise ValueError("Dataset must contain an 'audio' feature of type Audio")

        if "transcript" not in dataset.features:
            raise ValueError("Dataset must contain a 'transcript' feature")

        if not "has_timestamps" in dataset.features:
            print("Dataset does not contain a 'has_timestamps' feature. Assuming no timestamps.")

        if not "has_prev" in dataset.features:
            print("Dataset does not contain a 'has_prev' feature. Assuming no previous transcript.")
        elif not "prev_transcript" in dataset.features:
            raise ValueError("Dataset must contain a 'prev_transcript' feature if 'has_prev' is True")

    def _get_token_timestamp_for_time(self, time: float) -> int:
        # Sanity - time cannot be more than max timestamp token.
        if self.max_allowed_tokenized_timestamp < time:
            raise ValueError(f"Time {time} is too large. Max allowed time is {self.max_allowed_tokenized_timestamp}")

        # Round to nearest multiple of timestamp_token_time_precision
        return self.timestamp_begin_token_id + int(round(time / self.tokenizer_time_precision))

    def _select_legal_entries(self, processed_dataset):
        return processed_dataset.filter(
            lambda labels: len(labels) <= whisper_max_target_positions, input_columns="labels"
        )

    def _decide_example_augmentation(self, example, ancillary_features):
        """
        Decide whether to augment the audio with a shift (0 if no augmentation)
        """
        example_shift = 0
        audio = example["audio"]
        audio_duration = audio["array"].shape[0] / audio["sampling_rate"]
        if self.audio_shift_augmentation:
            max_shift = self.max_shifted_audio_ends_at - audio_duration

            if max_shift > 0:  # ony shift if there is room for it
                example_shift = round(
                    self.seed.beta(2, 3) * max_shift, 2
                )  # Skew toward zero. rand to whisper audio timestamp precision

        ancillary_features["audio_shift_augmentation"] = example_shift
        ancillary_features["original_audio_duration"] = audio_duration

    def _prepare_example_audio(self, example, ancillary_features, result_example: BatchFeature) -> None:
        audio = example["audio"]
        example_audio_shift_augmentation = ancillary_features["audio_shift_augmentation"]
        original_sampling_rate = audio["sampling_rate"]
        target_sampling_rate = self.target_sampling_rate

        if original_sampling_rate != target_sampling_rate:
            resampler = Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
            audio_array = torch.tensor(audio["array"]).float()
            resampled_audio_array = resampler(audio_array).numpy()
        else:
            resampled_audio_array = audio["array"]

        if example_audio_shift_augmentation > 0:
            resampled_audio_array = shift_audio_forward(
                resampled_audio_array, example_audio_shift_augmentation, target_sampling_rate
            )

        # We want to use the device kwargs - we call the feature extractor directly
        # to avoid warning from the tokenizer (which does not know how to consume a device kwarg)
        feature_extraction_result = self.processor.feature_extractor(
            resampled_audio_array, sampling_rate=target_sampling_rate, return_attention_mask=True, device=self.device
        )

        input_feat = feature_extraction_result["input_features"][0]
        attn_mask = feature_extraction_result["attention_mask"][0]

        pad_value = None
        pad_amount = None

        # Only makes sense to compress when we have at least 2 paddings to replace with 1
        if attn_mask[-3] == 0:
            # +1 because the STFT window overflows slightly into the first silence padding frame
            # And the model might learned to use that signal to recognize start of silence.
            padding_starts_at_index = attn_mask.argmin() + 1
            padding_vals = input_feat.T[padding_starts_at_index]
            pad_value = padding_vals
            input_feat_keep = input_feat.T[:padding_starts_at_index].T
            orig_out_features_length = input_feat.shape[-1]
            pad_amount = np.int32(orig_out_features_length - padding_starts_at_index)
        else:
            input_feat_keep = input_feat
            pad_value = np.array([], dtype=input_feat.dtype)
            pad_amount = np.int32(0)

        result_example["input_features"] = input_feat_keep
        result_example["pad_value"] = pad_value
        result_example["pad_amount"] = pad_amount
        result_example["input_length"] = len(resampled_audio_array) / target_sampling_rate

    def _prepare_example_text(self, example, ancillary_features, result_example, relative_sampling_ratios) -> None:
        text = example["transcript"]
        has_timestamps = example["has_timestamps"] if "has_timestamps" in example else False
        has_prev = example["has_prev"] if "has_prev" in example else False
        prev_text = example["prev_transcript"] if has_prev else None

        tokenizer_result = self.processor.tokenizer(
            text,
            # We will take care of prefixes/suffixes manually
            add_special_tokens=False,
            # Motivation: sometimes post-training models glue words together.
            # BUT - if timestamps are present we do not need to add a prefix space to
            # the text since it was already pre-formatted when timestamps tokens were interleaved into it.
            add_prefix_space=not has_timestamps,
            return_attention_mask=False,
        )
        token_ids = tokenizer_result["input_ids"]

        # Should we use and how we sample attributes
        if relative_sampling_ratios is None:
            use_timestamps_sampling_prob = self.timestamp_sample_prob
            use_prev_sampling_prob = self.condition_on_prev_sample_prob
        else:
            use_timestamps_sampling_prob = relative_sampling_ratios["has_timestamps"]
            use_prev_sampling_prob = relative_sampling_ratios["has_prev"]

        should_train_on_timestamps = bool(self.seed.binomial(1, use_timestamps_sampling_prob))
        should_condition_on_prev = has_prev and bool(self.seed.binomial(1, use_prev_sampling_prob))

        if has_timestamps and not should_train_on_timestamps:
            # Remove all timestamp tokens
            token_ids = [token_id for token_id in token_ids if token_id < self.timestamp_begin_token_id]
            # Note - no-timestamp token id is prepended as part of the prefix later.

        prev_ids = []
        if should_condition_on_prev and has_prev:
            prev_ids = self.processor.tokenizer(
                prev_text,
                add_special_tokens=False,
                # See why above
                add_prefix_space=not has_timestamps,
                return_attention_mask=False,
            )["input_ids"]

            if not should_train_on_timestamps and has_timestamps:
                prev_ids = [token_id for token_id in prev_ids if token_id < self.timestamp_begin_token_id]

            # Calculate how many prev_ids we want to take
            max_prev_ids_len_to_take = min(
                whisper_max_target_positions - len(token_ids)
                # 3 - Prefix for transcription (sot+lang+task)
                # 1 - eot token
                # 1 - Prefix for prev (prev)
                - 5,
                # And anyway no more than half the max size
                self.prev_ids_max_length,
            )

            # Take as much as we can from the prev_ids
            prev_ids = prev_ids[-max_prev_ids_len_to_take:]

            # prepend a prev token
            prev_ids = [self.start_of_prev_token_id] + prev_ids

        # If we don't have timestamps and we don't use prev text, but want to train on timestamps,
        # and allowed to perform timestamp augmentation, then we need to inject ts tokens.
        # Know this - prev text labels should include timestamps if the transcription labels do.
        # Since we are unable to inject timestamps into prev text, we cannot accomplish the injection
        # in those cases. Hence, the below check of "prev_ids"
        # TODO - When this feature is used - sampling probs are skewed since we "add" timestamp attributes on the fly.
        # this is a bug, and not compatible with the sampling ratios atm.
        if should_train_on_timestamps and not has_timestamps and not prev_ids and self.inject_synthetic_timestamps:
            # Injected timestamps may be "shift forward" augmented.
            # Audio features would have been augmented accordingly.
            example_shift_augmentation = ancillary_features["audio_shift_augmentation"]
            duration = ancillary_features["original_audio_duration"]

            start_at_ts_id = self._get_token_timestamp_for_time(example_shift_augmentation)
            ends_at = example_shift_augmentation + duration
            ends_at_ts_id = self._get_token_timestamp_for_time(ends_at)
            # wrap the text segment with the synthetic injected timestamp tokens
            # and append prefix/suffix for timestamp decoding
            token_ids = [start_at_ts_id] + token_ids + [ends_at_ts_id]
            has_timestamps = True  # So downstream processing handles proper prefixing

        with_timestamps = has_timestamps and should_train_on_timestamps
        prefix_tokens = self.prefix_tokens_with_ts if with_timestamps else self.prefix_tokens_no_ts
        labels_input_ids = prev_ids + prefix_tokens + token_ids + [self.eot_token_id]

        result_example["labels"] = labels_input_ids

    def _prepare_example_fn(self, input_example, relative_sampling_ratios=None):
        try:
            # A container for the output features
            result_example = BatchFeature({})
            # A container for ancillary features that are not part of the output features
            ancillary_features = BatchFeature({})

            self._decide_example_augmentation(input_example, ancillary_features)
            self._prepare_example_audio(input_example, ancillary_features, result_example)
            self._prepare_example_text(
                input_example,
                ancillary_features,
                result_example,
                relative_sampling_ratios,
            )

            return result_example
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def estimate_attribute_ratios(
        self,
        dataset,
        discriminators: dict[str, callable],
        target_sampling_error_range: float = 0.05,
        target_sampling_error_confidence: float = 0.95,
        max_to_sample: int = 4000,
    ):
        """Estimate the ratio of positive samples for attributes in the dataset.
        This function uses a beta distribution to calculate the confidence interval for the estimated ratio.
        The function will stop sampling when either the confidence interval is within the target range or
        the maximum number of samples has been reached.

        Args:
            dataset (Dataset): The input dataset to sample from.
            discriminators (dict[str, callable]): A dictionary where keys are attribute names and values are functions
                that take a sample and return True if the attribute is present, False otherwise.
            target_sampling_error_range (float): The target range for the sampling error.
            0.05 means 5% error. (2.5% on each side)
            target_sampling_error_confidence (float): The target confidence level for the sampling error.
                0.95 means 95% confidence.
            max_to_sample (int): The maximum number of samples to draw from the dataset.

        Returns:
            estimation (dict[str, dict[float, float, int]]): A dictionary where keys are attribute names and values are dictionaries
                containing the estimated ratio, confidence interval, and number of samples.
        """
        attr_states = {
            name: {
                "total_sampled": 0,
                "total_positive": 0,
                "done": False,
                "conf_interval": (0.0, 1.0),
            }
            for name in discriminators
        }

        for sample in dataset.to_iterable_dataset():
            all_done = True

            for attr, disc_fn in discriminators.items():
                state = attr_states[attr]

                positive = bool(disc_fn(sample))
                state["total_sampled"] += 1
                if positive:
                    state["total_positive"] += 1

                total = state["total_sampled"]
                pos = state["total_positive"]

                a, b = 1 + pos, 1 + (total - pos)
                conf_int = beta.interval(target_sampling_error_confidence, a, b)
                state["conf_interval"] = conf_int

                # Only mark as done if not already, and if either condition is met
                if not state["done"]:
                    if conf_int[1] - conf_int[0] < target_sampling_error_range:
                        state["done"] = True
                    elif total >= max_to_sample:
                        state["done"] = True

                if not state["done"]:
                    all_done = False

            if all_done:
                break

        results = {
            attr: {
                "estimated_ratio": state["total_positive"] / state["total_sampled"],
                "confidence_interval": state["conf_interval"],
                "samples": state["total_sampled"],
            }
            for attr, state in attr_states.items()
        }

        return results

    def prepare_dataset(self, dataset: Dataset):
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))
        self._validate_dataset_features(dataset)

        columns_to_remove = dataset.column_names
        # If a DatasetDict was passed in, it contains multiple splits.
        # Take the names of the columns from any of the splits
        if isinstance(dataset, DatasetDict):
            any_split_name = next(iter(columns_to_remove.keys()))
            columns_to_remove = dataset[any_split_name].column_names

        # If we need to sub-sample some of the attributes of the dataset
        # we will first estimate the ratios of the attributes in the dataset
        # and then sample the dataset to match the ratios
        relative_sampling_ratios = None
        if self.timestamp_sample_prob < 1.0 or self.condition_on_prev_sample_prob < 1.0:
            print(f"Estimating attribute frequencies for relative sub-sampling.")

            # Estimate the ratios of the attributes in the dataset
            estimations = self.estimate_attribute_ratios(
                dataset,
                {
                    "has_timestamps": lambda example: "has_timestamps" in example and example["has_timestamps"],
                    "has_prev": lambda example: "has_prev" in example and example["has_prev"],
                },
                target_sampling_error_range=0.05,
                target_sampling_error_confidence=0.95,
                max_to_sample=4000,
            )

            relative_sampling_ratios = {
                attr: (
                    min(1.0, self.timestamp_sample_prob / attr_estimation["estimated_ratio"])
                    if attr_estimation["estimated_ratio"] > 0
                    else 1.0
                )
                for attr, attr_estimation in estimations.items()
            }

            estimated_ratios = {attr: est["estimated_ratio"] for attr, est in estimations.items()}
            print(f"Estimated attribute frequencies: {estimated_ratios}")
            print(f"Relative sampling ratios: {relative_sampling_ratios}")

        processed_dataset = dataset.map(
            self._prepare_example_fn,
            fn_kwargs={
                "relative_sampling_ratios": relative_sampling_ratios,
            },
            remove_columns=columns_to_remove,
            num_proc=self.proc_num,
            features=self.output_features,
        )

        processed_dataset = self._select_legal_entries(processed_dataset)

        return processed_dataset


def process_datasets(datasets, preparator: DatasetPreparator):
    processed_datasets = [preparator.prepare_dataset(dataset=dataset_) for dataset_ in datasets]
    return concatenate_datasets(processed_datasets) if len(processed_datasets) > 1 else processed_datasets[0]


if __name__ == '__main__':
    # arrow_dir = "/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/final_dataset"  # Directory containing .arrow files
    # shutil.copytree(arrow_dir, "/tmp/verbit/train")
    # shutil.copytree(arrow_dir, "/tmp/verbit/eval")
    # dataset = load_from_disk("/tmp/verbit")
    save_path = "/shares/rndsounds/wake_up_word/genAI/ac/stt/verbit_dataset/large_v3_turbo"
    prep = DatasetPreparator(processor=WhisperProcessor.from_pretrained('openai/whisper-large-v3-turbo'), proc_num=32, seed=21)
    # prep = DatasetPreparator(processor=WhisperProcessor.from_pretrained('openai/whisper-medium'), proc_num=32,
    #                          seed=21)
    # Load each .arrow file as a Dataset
    dataset_list = ['eval', 'train']
    dataset_list_ = []
    for dat in dataset_list:
        dataset_dir = f"/home/shimonc/code/whisper/arb_stt/code/asr-training/verbit/{dat}/"  # Directory containing .arrow files
        dataset = [load_from_disk(dataset_dir)]
        # dataset_list_.append(dataset)
        processed_dataset = process_datasets(dataset, prep)
        chunk_size = 5000  # adjust as needed
        num_chunks = len(processed_dataset) // chunk_size + 1
        for i in range(num_chunks):
            if os.path.exists(f"{save_path}/{dat}_part_{i}"):
                continue
            chunk = processed_dataset.select(range(i * chunk_size, min((i + 1) * chunk_size, len(processed_dataset))))
            chunk.save_to_disk(f"{save_path}/{dat}_part_{i}")
        # processed_dataset.save_to_disk(save_path + '/' + dat)
    # shutil.rmtree(f"/tmp/verbit/")
    # final_dataset.save_to_disk(save_path)