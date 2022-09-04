from .base_tokenizer import Tokenizer


class WSTokenizer(Tokenizer):
    @staticmethod
    def text2tokens(text):
        return text.split()

    @staticmethod
    def tokens2text(tokens):
        return " ".join(tokens)
