import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, ProcessorMixin


def convert_to_standard_chat(example, num_images=0):
    if isinstance(example, list) and example and isinstance(example[0], dict) and "role" in example[0] and "content" in example[0]:
        return example

    standard_chat = []

    for message in example:
        standard_chat.append({"role": "user", "content": [{"text": message["user"], "type": "text"}]})
        standard_chat.append({"role": "assistant", "content": [{"text": message["assistant"], "type": "text"}]})
    
    if num_images > 0:
        for _ in range(num_images):
            standard_chat[0]["content"] = [{"type": "image"}] + standard_chat[0]["content"]

    return standard_chat


# Keys that are aligned with `input_ids` (one entry per token) and therefore have to be
# padded to the batch length instead of being concatenated. Gemma 4 returns
# `mm_token_type_ids`, which the model uses to build the bidirectional image-block mask.
TOKEN_ALIGNED_KEYS = ("token_type_ids", "mm_token_type_ids")


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    processor: Optional[ProcessorMixin] = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        input_ids = [input_id.squeeze(0) for input_id in input_ids]
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        inputs.pop("attention_mask", None)
        batched_inputs = {}
        for key, values in inputs.items():
            if key in TOKEN_ALIGNED_KEYS:
                batched_inputs[key] = self.pad_sequence(
                    [value.squeeze(0) for value in values],
                    batch_first=True,
                    padding_value=0,
                )
            else:
                batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["attention_mask"] = attention_mask

        return batched_inputs


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Union[HFDataset, str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_key: str,
        image_key: Optional[str] = None,
        video_key: Optional[str] = None,
        audio_key: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = HFDataset.from_parquet(dataset)

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.dataframe = dataset

    def __getitem__(self, index):
        row = self.dataframe[index]

        if self.processor is not None:
            multi_modal_inputs = {}
            images = None
            if self.image_key in row:
                images = [image for image in row[self.image_key]]
                multi_modal_inputs["images"] = images

            # TODO
            # Implement the load logic for video and audios later
            if self.video_key in row:
                videos = [video for video in row[self.video_key]]
                multi_modal_inputs["videos"] = videos

            if self.audio_key in row:
                audios = [audio for audio in row[self.audio_key]]
                multi_modal_inputs["audios"] = audios

            text = self.processor.apply_chat_template(
                convert_to_standard_chat(row[self.text_key], num_images=len(multi_modal_inputs.get("images", []))),
                tokenize=False,
                add_generation_prompt=False
            )

            # `text` already carries every special token written by the chat template
            # (Gemma's template emits `bos_token` while its tokenizer would add another one).
            model_inputs = self.processor(
                text=[text],
                return_tensors="pt",
                add_special_tokens=False,
                **multi_modal_inputs,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.tokenizer(
                [text], return_tensors="pt", add_special_tokens=False
            )

        return model_inputs

    def get_collator(self):
        return DataCollator(self.tokenizer, self.processor)

    def __len__(self):
        return len(self.dataframe)


class CacheIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Union[IterableDataset, str], # Expects an iterable or path
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_key: str,
        image_key: Optional[str] = None,
        video_key: Optional[str] = None,
        audio_key: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = load_dataset(dataset, streaming=True, split="train")
        else:
            dataset = dataset
        
        if isinstance(dataset, HFDataset):
            dataset = dataset.to_iterable_dataset()

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.dataset = dataset

    def process_item(self, row):
        """
        Logic extracted from the old __getitem__. 
        Processes a single row into model inputs.
        """
        if self.processor is not None:
            multi_modal_inputs = {}
            if self.image_key in row and row[self.image_key]:
                for img in row[self.image_key]:
                    w, h = img.size
                    ar = max(w / h, h / w)
                    # Skip images with extreme aspect ratios
                    if ar >= 200:
                        return None
                multi_modal_inputs["images"] = row[self.image_key]
            # Only process single-image examples for now
            num_images = len(multi_modal_inputs.get("images", []))
            if num_images != 1:
                return None

            if self.video_key in row and row[self.video_key]:
                multi_modal_inputs["videos"] = row[self.video_key]

            if self.audio_key in row and row[self.audio_key]:
                multi_modal_inputs["audios"] = row[self.audio_key]

            text = self.processor.apply_chat_template(
                convert_to_standard_chat(row[self.text_key], num_images=num_images),
                tokenize=False,
                add_generation_prompt=False,
            )

            # `text` already carries every special token written by the chat template
            # (Gemma's template emits `bos_token` while its tokenizer would add another one).
            model_inputs = self.processor(
                text=[text],
                return_tensors="pt",
                max_length=8192,
                truncation=True,
                add_special_tokens=False,
                **multi_modal_inputs,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.tokenizer(
                [text], return_tensors="pt", add_special_tokens=False
            )

        return model_inputs

    def __iter__(self):
        """
        Instead of getting an index, we iterate through the stream.
        """
        for row in self.dataset:
            item = self.process_item(row)
            if item is not None:
                yield item

    def get_collator(self):
        return DataCollator(self.tokenizer, self.processor)
