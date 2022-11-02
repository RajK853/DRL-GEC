from typing import List

from .base import Tokenizer


class WSTokenizer(Tokenizer):
    """
    Whitespace-based tokenizer
    """
    @staticmethod
    def text2tokens(text: str) -> List[str]:
        """
        Convert a text into tokens
        """
        return text.split()

    @staticmethod
    def tokens2text(tokens: List[str]) -> str:
        """
        Convert tokens into a text
        """
        return " ".join(tokens)
