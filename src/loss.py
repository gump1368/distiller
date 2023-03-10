#! -*- coding: utf-8 -*-
"""
@Author: Gump
@Create Time: 20230301
@Info: loss function
"""
import tensorflow as tf


def cos_loss(t_h, s_h, label):
    """cosine similarity loss"""
    t_norm = tf.sqrt(tf.reduce_sum(tf.square(t_h), axis=-1))
    s_norm = tf.sqrt(tf.reduce_sum(tf.square(s_h), axis=-1))

    inner = tf.reduce_sum(tf.multiply(t_h, s_h), axis=-1)
    cos = inner / (t_norm * s_norm)

    print(cos)

    return tf.cond(tf.equal(label, 1),
                   lambda: 1 - cos,
                   lambda: tf.cond(cos > 0, lambda: cos, lambda: tf.constant(0.0)))


def cos_loss_batch(batch_t_h, batch_s_h, labels):
    """batch cosine loss"""
    batch_size = batch_t_h.shape[0]
    batch_loss = []
    for i in range(batch_size):
        t_h = batch_t_h[i]  # (, h)
        s_h = batch_s_h[i]  # (, h)
        label = labels[i]

        loss = cos_loss(t_h, s_h, label)
        batch_loss.append(loss)

    return tf.reduce_mean(batch_loss)


def KL_loss(log_s, log_t):
    """KL divergence loss"""
    y_true = tf.exp(log_t)
    neg_t = tf.reduce_sum(y_true * log_t, axis=-1)
    neg_s = tf.reduce_sum(y_true * log_s, axis=-1)
    kl_loss = tf.reduce_mean(neg_t - neg_s, axis=-1)
    return kl_loss


def cross_entropy_loss(log_prob, labels, num_class):
    """cross entropy loss"""
    one_hot_labels = tf.one_hot(labels, depth=num_class, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_prob, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return per_example_loss, loss


