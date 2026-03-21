import os
import sentencepiece as spm


class IIMoTokenizer:
    """
    Wrapper around SentencePiece tokenizer
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def pad_id(self):
        return self.sp.pad_id()

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

    def vocab_size(self):
        return self.sp.get_piece_size()

    @staticmethod
    def train(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe"
    ):
        """
        Train a SentencePiece tokenizer
        """
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=0.9995,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            unk_id=3
        )