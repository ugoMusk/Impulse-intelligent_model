import os
import json
import tensorflow as tf
from sentencepiece import SentencePieceProcessor


class DataLoader:
    """
    Streaming DataLoader for causal language modeling (decoder-style training)
    """

    def __init__(self, data_dir: str, tokenizer_path: str, max_seq_len: int = 512):
        self.data_dir = data_dir
        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
        self.max_seq_len = max_seq_len

    def _format_example(self, example: dict) -> str:
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        output = example.get("output", "")

        return (
            f"Instruction: {instruction}\n"
            f"Context: {context}\n"
            f"Output: {output}"
        )

    def _tokenize(self, text: str):
        tokens = self.tokenizer.encode(text, out_type=int)
        tokens = tokens[: self.max_seq_len]

        pad_id = self.tokenizer.pad_id()
        if len(tokens) < self.max_seq_len:
            tokens += [pad_id] * (self.max_seq_len - len(tokens))

        return tokens

    def _generator(self, split: str):
        split_dir = os.path.join(self.data_dir, split)

        for file_name in os.listdir(split_dir):
            if not file_name.endswith(".json"):
                continue

            with open(os.path.join(split_dir, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)

                for example in data:
                    text = self._format_example(example)
                    tokens = self._tokenize(text)

                    # Causal LM: shift inputs/labels
                    input_ids = tokens[:-1]
                    labels = tokens[1:]

                    yield {
                        "input_ids": input_ids,
                        "labels": labels
                    }

    def build_dataset(self, split: str, batch_size: int, shuffle: bool = True):
        output_signature = {
            "input_ids": tf.TensorSpec(shape=(self.max_seq_len - 1,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(self.max_seq_len - 1,), dtype=tf.int32),
        }

        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(split),
            output_signature=output_signature
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset