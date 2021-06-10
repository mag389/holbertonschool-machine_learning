#!/usr/bin/env python3
""" question answer based on pretrained bert model """
import tensorflow as tf
import tensorflow_hub as hub
import transformers
from transformers import BertTokenizer, AutoModelForQuestionAnswering


def question_answer(question, reference):
    """ answers question based on reference document """
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(reference_tokens) + 1)
    input_word_ids, input_mask, input_type_ids = map(
            lambda t: tf.expand_dims(
                tf.convert_to_tensor(t, dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
