import spacy

spacy_nlp = spacy.load('en_core_web_sm') # Load the English Model
import re



def get_sub_sens(text):
    answer_tokens = []
    answer_token_is_sent_starts = []
    all_tokens = []
    sub_sentences = []
    doc = nlp(text)
    for s in doc.sents:
        s_text = s.text
        answer_tokens += s_text.split()
        sub_sent_starts = get_subsentence_starts(s_text.split())
        answer_token_is_sent_starts += sub_sent_starts
        all_tokens += s_text.split()
    sub = ""
    for i in range(len(answer_tokens)):
        if answer_token_is_sent_starts[i] and i != 0:
            sub_sentences.append(sub[:-1])
            sub = answer_tokens[i] + " "
        elif i == len(answer_tokens)-1:
            sub += answer_tokens[i] + " "
            sub_sentences.append(sub[:-1])
        else:
            sub += answer_tokens[i] + " "

    return sub_sentences

MIN_SUBSENT_WORDS = 5
def get_subsentence_starts(tokens):

    def _is_tok_end_of_subsent(tok):
        if re.match('[,;!?]', tok[-1]) is not None:
            return True
        return False

    assert len(tokens) > 0
    is_subsent_starts = [True]
    prev_tok = tokens[0]
    prev_subsent_start_idx = 0
    for i, tok in enumerate(tokens[1:]):
        tok_id = i + 1
        if _is_tok_end_of_subsent(prev_tok) and tok_id + MIN_SUBSENT_WORDS < len(tokens):
            if tok_id - prev_subsent_start_idx < MIN_SUBSENT_WORDS:
                if prev_subsent_start_idx > 0:
                    is_subsent_starts += [True]
                    is_subsent_starts[prev_subsent_start_idx] = False
                    prev_subsent_start_idx = tok_id
                else:
                    is_subsent_starts += [False]
            else:
                is_subsent_starts += [True]
                prev_subsent_start_idx = tok_id
        else:
            is_subsent_starts += [False]
        prev_tok = tok

    return is_subsent_starts


# print(get_sub_sens("A possible reason for the half-eaten sandwich could be that a person has taken a break in the middle of their meal, perhaps to attend to something urgent, have a conversation, or simply enjoy their meal at a slower pace. The sandwich, accompanied by a cup of potato salad, suggests that it's lunchtime, and the person might be engaged in multitasking or might have been interrupted while eating. It is also possible that the person could be full or not enjoying the taste of the sandwich and decided to stop eating. Without more context, it's difficult to determine the exact reason, but these are some plausible explanations for a half-eaten sandwich in the image."))


# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the helper functions that split long text into sentences and subsentences
#  All Rights Reserved.
#  *********************************************************


# split long text into sentences
def split_text_to_sentences(long_text, ):
    doc = spacy_nlp(long_text)
    return [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]


# split long text into subsentences
def split_text_to_subsentences(long_text, ):
    def get_sub_sentence_starts(tokens, min_subsent_words=5):

        def _is_tok_end_of_subsent(tok):
            if re.match('[,;!?]', tok[-1]) is not None:
                return True
            return False

        # assert len(tokens) > 0
        is_subsent_starts = [True]
        prev_tok = tokens[0]
        prev_subsent_start_idx = 0
        for i, tok in enumerate(tokens[1:]):
            tok_id = i + 1
            if _is_tok_end_of_subsent(prev_tok) and tok_id + min_subsent_words < len(tokens):
                if tok_id - prev_subsent_start_idx < min_subsent_words:
                    if prev_subsent_start_idx > 0:
                        is_subsent_starts += [True]
                        is_subsent_starts[prev_subsent_start_idx] = False
                        prev_subsent_start_idx = tok_id
                    else:
                        is_subsent_starts += [False]
                else:
                    is_subsent_starts += [True]
                    prev_subsent_start_idx = tok_id
            else:
                is_subsent_starts += [False]
            prev_tok = tok

        return is_subsent_starts

    def tokenize_with_indices(text):
        tokens = text.split()
        token_indices = []

        current_index = 0
        for token in tokens:
            start_index = text.find(token, current_index)
            token_indices.append((token, start_index))
            current_index = start_index + len(token)

        return token_indices

    doc = spacy_nlp(long_text)
    sentence_start_char_idxs = [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]

    char_starts = []

    for sentence_idx, sentence_start_char_idx in enumerate(sentence_start_char_idxs[:-1]):

        sentence = long_text[sentence_start_char_idx: sentence_start_char_idxs[sentence_idx + 1]]

        tokens_with_indices = tokenize_with_indices(sentence)

        tokens = [i[0] for i in tokens_with_indices]
        is_sub_starts = get_sub_sentence_starts(tokens, min_subsent_words=5)

        for token_with_idx, is_sub_start in zip(tokens_with_indices, is_sub_starts):
            if is_sub_start:
                char_starts.append(sentence_start_char_idx + token_with_idx[1])

    return char_starts + [len(long_text)]

# sen = ("A possible reason for the half-eaten sandwich could be that a person has taken a break in the middle of"
#        " their meal, perhaps to attend to something urgent, \n \n have a conversation, or simply enjoy their meal at"
#        " a slower pace. The sandwich, accompanied by a cup of potato salad, suggests that it's lunchtime, "
#        "and the person might be engaged in multitasking or might have been interrupted while eating. "
#        "It is also possible that the person could be full or not enjoying the taste of the sandwich "
#        "and decided to stop eating. Without more context, it's difficult to determine the exact reason, "
#        "but these are some plausible explanations for a half-eaten sandwich in the image.")
# indices = split_text_to_subsentences(sen)
# print(sen)
# print(indices)
# for i in range(len(indices)-1):
#     print(sen[indices[i]: indices[i+1]])