
import json
import logging
import collections
from . tokenizer import Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


"""
SQuAD 2.0 data format:

{'data':
        [
            {'paragraphs':
             [
                {'context': 'CONTEXT',
                    'qas':
                    [
                        {'answers':
                         [
                            {'answer_start': 144,
                             'text': 'ranch hand'}
                         ],
                         'id': '56cd92e562d2951400fa6728',
                         'is_impossible': False,
                         'question': 'QUESTION'},

                        {'answers': [],
                        'id': '5a8d9520df8bba001a0f9b15',
                        'is_impossible': True,
                        'plausible_answers': [{'answer_start': 144,
                                               'text': 'ranch '
                                                       'hand'}],
                        'question': 'QUESTION',
                         ...
                    ]

                },
                ...
             ],
            'title': 'TITLE'},
            ...
        ]
'version': 'v2.0'}

The SQuAD 2.0 dataset contains many `paragraph`, each `paragraph`
consists a `context` (content of paragraph) and a `qas`.

`qas` is a list of questions and corresponding answers based on
the `paragraph`.

Not every question has an answer, e.g., the unanswerable questions.
They have `plausible_answers`.

Notice that `answer_start` refers to the index of character.
"""


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class SquadSample(object):
    """SQuAD Data Sample
    A training / evaluate data sample
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        s = "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: %s" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SquadInputFeature(object):
    r"""SQuAD Input Feature
    Single input feature for model
    """

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):

        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SquadDataLoader(object):
    r"""SQuAD DataLoader
    Data Loader for SQuAD 2.0 Dataset
    (1): Read dataset json file into a list of `SquadSample`
    (2): Transform the SquadSample list into a list of `InputFeature`

    In step (1), we mainly extract the raw text from json file. And
    the most important thing we do is that we build a mapping from
    character index to token index. This allows us to formulate the
    answer start / end positions according to word index.

    In step (2), we convert a single `SquadSample` to `InputFeature`,
    mainly includes:
        1. Cut the question or context to max_length
        2. Tokenize question and context (tokens)
        3. (The most important step) Build a mapping
           from original context tokens to tokenized
           context tokens, as WordPiece could split
           a single token into many, which will dis-
           turb the index for start/end position.
    """

    def __init__(self, vocab_file):
        self.squad_samples = []
        self.input_features = []
        self.tokenizer = Tokenizer(vocab_file)

    def load(self, data_file, max_question_len, max_seq_len, doc_stride, is_training=True):
        self.read_file(data_file, is_training)
        return self.sample_to_feature(max_question_len, max_seq_len, doc_stride, is_training=True)

    def read_file(self, data_file, is_training=True):
        with open(data_file, "r", encoding='utf-8') as f:
            json_data = json.load(f)
            raw_data = json_data["data"]
            version = json_data["version"]

        for data_sample in raw_data:
            paragraphs = data_sample["paragraphs"]
            for paragraph in paragraphs:
                context = paragraph["context"]
                doc_tokens, char_to_word_offset = self.char_to_word_map(context)
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    is_impossible = qa["is_impossible"]
                    answer_text = None
                    start_position = None
                    end_position = None

                    if is_training:
                        if is_impossible:
                            answer_text = ""
                            start_position = -1
                            end_position = -1
                        else:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]

                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue

                    squad_sample = SquadSample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        answer_text=answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)

                    self.squad_samples.append(squad_sample)

        return self.squad_samples

    def sample_to_feature(self, max_question_len, max_seq_len, doc_stride, is_training=True):
        """Convert SQuAD sample to input features for model"""

        unique_id = 1000000000
        input_features = []

        for (sample_idx, sample) in enumerate(self.squad_samples):
            # tokenize question
            question_tokens = self.tokenizer.tokenize(sample.question_text)

            if len(question_tokens) > max_question_len:
                question_tokens = question_tokens[0: max_question_len]

            # tokenize context
            tok_to_orig_index = []
            orig_to_tok_index = []
            tokenized_doc_tokens = []

            # tokenize each token in doc_token
            for (token_idx, token) in enumerate(sample.doc_tokens):
                orig_to_tok_index.append(len(tokenized_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(token_idx)
                    tokenized_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None

            if is_training:
                if sample.is_impossible:
                    tok_start_position = -1
                    tok_end_position = -1
                else:
                    tok_start_position = orig_to_tok_index[sample.start_position]
                    if sample.end_position < len(sample.doc_tokens) - 1:
                        # the last subtoken
                        tok_end_position = orig_to_tok_index[sample.end_position + 1] - 1
                    else:
                        tok_end_position = len(tokenized_doc_tokens) - 1

                    (tok_start_position, tok_end_position) = self.improve_answer_span(
                        tokenized_doc_tokens, tok_start_position, tok_end_position,
                        sample.orig_answer_text)

            # The input sentence structure is
            # [CLS] Question [SEP] Context [SEP]
            max_tokens_for_doc = max_seq_len - len(question_tokens) - 3

            # when doc_length > max_tokens_for_doc, we use sliding window approach
            # to create
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0

            # build a list of doc spans
            while start_offset < len(tokenized_doc_tokens):
                length = len(tokenized_doc_tokens) - start_offset
                length = min(length, max_tokens_for_doc)
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(tokenized_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            # build a sample for each doc_span
            # a sample has 3 inputs:
            #   (1): tokens embeddings
            #   (2): segment embeddings
            #   (3): position embeddings
            #   (4): input masks

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []

                # The first token
                tokens.append("[CLS]")
                segment_ids.append(0)

                # Question
                for question_token in question_tokens:
                    tokens.append(question_token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                # Context
                for relative_idx in range(doc_span.length):
                    doc_token_idx = doc_span.start + relative_idx
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[doc_token_idx]
                    is_max_context = self.check_is_max_context(doc_spans, doc_span_index,
                                                           doc_token_idx)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(tokenized_doc_tokens[doc_token_idx])
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)

                input_embeddings = self.tokenizer.tokens_to_ids(tokens)

                input_mask = [1] * len(input_embeddings)

                input_len = len(input_embeddings)
                if input_len < max_seq_len:
                    input_embeddings.extend([0] * (max_seq_len - input_len))
                    input_mask.extend([0] * (max_seq_len - input_len))
                    segment_ids.extend([0] * (max_seq_len - input_len))

                assert len(input_embeddings) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                start_position = None
                end_position = None

                if is_training:
                    if sample.is_impossible:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1

                        out_of_span = False
                        if (sample.start_position < doc_start or
                                    sample.end_position < doc_start or
                                    sample.start_position > doc_end or sample.end_position > doc_end):
                            out_of_span = True

                        if out_of_span:
                            start_position = 0
                            end_position = 0
                        else:
                            doc_offset = len(question_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset

                if sample_idx < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (sample_idx))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_embeddings]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training and sample.is_impossible:
                        logger.info("impossible example")
                    if is_training and not sample.is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))

                input_features.append(
                    SquadInputFeature(
                        unique_id=unique_id,
                        example_index=sample_idx,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_embeddings,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=sample.is_impossible))
                unique_id += 1

        return input_features

    def char_to_word_map(self, text):
        """build a char_to_word mapping
           (and split the text into tokens by the way)
        """
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        return doc_tokens, char_to_word_offset

    def check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def improve_answer_span(self, doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)


def main():
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    doc_stride = 3
    tokenized_doc_tokens = 50
    max_tokens_for_doc = 31

    # build a list of doc spans
    while start_offset < len(tokenized_doc_tokens):
        length = len(tokenized_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(tokenized_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    loader = SquadDataLoader()
    loader.read_file("/Users/apple/Downloads/train-v2.0.json")
    print(1)



if __name__ == '__main__':
    main()