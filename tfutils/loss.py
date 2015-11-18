import tensorflow as tf


def logistic_loss(logits, labels):
    """labels in 0/1"""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))


def linear_regression_loss(predictions, targets):
    """inputs must be 1d vectors"""
    return tf.reduce_mean(tf.nn.l2_loss(tf.sub(predictions, targets)))


def l2_regularizer(*vars):
    reg = tf.nn.l2_loss(vars[0])
    for v in vars[1:]:
        reg += tf.nn.l2_loss(v)
    return reg

