#! -*- coding: utf-8 -*-
"""
@Author: Gump
@Create Time: 20230305
@Info:
"""
import tensorflow as tf

max_seq_length = 128


def serving_input_fn():
    """export input fn"""
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
    token_type_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='token_type_ids')
    labels = tf.placeholder(tf.int32, [], name='labels')

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids': input_ids,
        'input_mask': input_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    })()
    return input_fn
