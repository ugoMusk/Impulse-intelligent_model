import sentencepiece as spm
from typing import List


class Tokenizer:
    """
    SentencePiece Tokenizer Wrapper
    """

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(model_path):
            raise ValueError(f"Failed to load tokenizer model: {model_path}")

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        return self.sp.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def pad(self, tokens: List[int], max_length: int) -> List[int]:
        return tokens[:max_length] + [0] * max(0, max_length - len(tokens))