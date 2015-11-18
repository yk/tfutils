import tensorflow as tf


def tf_logaddexp(t):
    tmax = tf.reduce_max(t, 1)
    tabsmax = tf.reduce_max(tf.abs(t), 1)
    tmin = tf.reduce_min(t, 1)
    bools = tf.where(tf.greater(tabsmax, tmax))
    c = tf.expand_dims(tf.concat(0, [tf.gather(tmax, tf.where(tf.greater(tabsmax, tmax))),
                                     tf.gather(tmin, tf.where(tf.less_equal(tabsmax, tmax)))]), -1)
    return tf.log(tf.reduce_sum(tf.exp(t - c))) + c


def tf_logoneplusexp(t):
    zero = tf.zeros_like(t)
    return tf_logaddexp(tf.pack([zero, t]))