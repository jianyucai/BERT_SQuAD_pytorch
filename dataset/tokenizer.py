
import unicodedata
import collections


class Vocab(object):
    r"""Vocabulary
    Read the vocabulary file, and store the
    [id, token] and [token, id] pairs.

    Args:
        vocab_file: Vocabulary file

    """

    def __init__(self, vocab_file):
        self.vocab = collections.OrderedDict()
        idx = 0

        with open(vocab_file, "r", encoding="utf-8") as f:
            while True:
                token = f.readline()
                if not token:
                    break
                # remove whitespaces
                token = token.strip()
                self.vocab[token] = idx
                idx += 1

        self.reverse_vocab = collections.OrderedDict(
            [(id, token) for (id, token) in self.vocab.items()]
        )

    def tokens_to_ids(self, tokens):
        """Convert tokens to ids"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def ids_to_tokens(self, ids):
        """Convert ids to tokens"""
        tokens = []
        for id in ids:
            tokens.append(self.reverse_vocab[id])
        return tokens


class TextCleaner(object):
    r"""Text Cleaner
    Cleans the text and tokens in following  steps:
        (1): Clean input text
        (2): Clean input token
    """

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def clean_token(self, token):
        """Clean token"""
        return self._run_split_on_punc(self._run_strip_accents(token))

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class Tokenizer(object):
    r"""Tokenizer
    Tasks:
        (1): Clean input text
        (2): Split text into tokens by whitespaces
        (3): Set all tokens lower case
        (4): Clean each token
        (5): Further tokenize each token with WordPiece rule
    """

    def __init__(self, vocab_file, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = Vocab(vocab_file)
        self.cleaner = TextCleaner()
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenization on a piece of text"""
        text = self.cleaner.clean_text(text)
        raw_tokens = self.whitespace_tokenize(text)
        tokens = [self.cleaner.clean_token(token.lower()) for token in raw_tokens]

        output_tokens = []
        for token in tokens:
            output_tokens.extend(self.wordpiece_tokenize(token))
        return output_tokens

    def tokens_to_ids(self, tokens):
        return self.vocab.tokens_to_ids(tokens)

    def ids_to_tokens(self, ids):
        return self.vocab.ids_to_tokens(ids)

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a peice of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def wordpiece_tokenize(self, token):
        """Use Greedy Rule to tokenize the input token based on vocabulary.
        The goal is to prevent too much [UNK] from appearing, since most
        tokens missing from vocabulary can be split into several simple
        sub-tokens which exists in vocabulary.

        Example:
            "flightless" -> "flight" + "##less"
        """
        sub_tokens = []
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            sub_tokens.append(self.unk_token)
            return sub_tokens

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end

        if is_bad:
            sub_tokens.append(self.unk_token)

        return sub_tokens