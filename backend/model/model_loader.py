import os
from typing import Optional

import tensorflow as tf

from model.transformer import IIMoModel
from model.tokenizer import IIMoTokenizer
from model.config import Config


class ModelLoader:
    """
    Model loader for IIMo.
    Handles:
    - Model initialization
    - Checkpoint loading
    - Tokenization
    - Text generation
    """

    def __init__(
        self,
        config_path: str,
        model_dir: str,
        tokenizer_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.config = Config(config_path).model
        self.model_dir = model_dir
        self.tokenizer = IIMoTokenizer(tokenizer_path)

        self.model: Optional[IIMoModel] = None
        self.device = device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")

    # =========================
    # Model
    # =========================
    def build_model(self) -> IIMoModel:
        """
        Initialize model architecture
        """
        model = IIMoModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_length,
            ff_dim=self.config.ff_dim,
            dropout=self.config.dropout,
        )
        return model

    # =========================
    # Load Weights
    # =========================
    def load_model(self) -> IIMoModel:
        """
        Load trained model weights from latest checkpoint
        """
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        with tf.device(self.device):
            self.model = self.build_model()

            # Build model (required before loading weights)
            dummy_input = tf.zeros((1, self.config.max_seq_length), dtype=tf.int32)
            _ = self.model(dummy_input, training=False)

            checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {self.model_dir}")

            self.model.load_weights(checkpoint_path)

        return self.model

    # =========================
    # Generate Text
    # =========================
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        End-to-end inference:
        text → tokens → model → tokens → text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize
        input_ids = self.tokenizer.encode(prompt)

        # Truncate if too long
        if len(input_ids) > self.config.max_seq_length:
            input_ids = input_ids[-self.config.max_seq_length :]

        input_ids = tf.expand_dims(input_ids, axis=0)

        # Generate
        output_ids = self._generate_tokens(input_ids, max_length)

        # Convert to list
        output_ids = output_ids.numpy().tolist()[0]

        # Decode
        return self.tokenizer.decode(output_ids)

    # =========================
    # Internal Generation Logic
    # =========================
    def _generate_tokens(self, input_ids: tf.Tensor, max_length: int) -> tf.Tensor:
        """
        Token-level generation with EOS stopping
        """
        eos_id = self.tokenizer.eos_id()

        for _ in range(max_length):
            logits = self.model(input_ids, training=False)

            next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, axis=1)

            input_ids = tf.concat([input_ids, next_token], axis=1)

            # Stop if EOS token is generated
            if eos_id != -1 and tf.reduce_any(tf.equal(next_token, eos_id)):
                break

            # Prevent overflow beyond max_seq_length
            if tf.shape(input_ids)[1] >= self.config.max_seq_length:
                break

        return input_ids