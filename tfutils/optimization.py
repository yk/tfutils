import tensorflow as tf
from tensorflow.python.training import optimizer


class SGD(optimizer.Optimizer):
    def __init__(self, learning_rate):
        super(SGD, self).__init__(False, "SGD")
        self.learning_rate = learning_rate

    def _apply_dense(self, grad, var):
        return var.assign_sub(tf.mul(grad, self.learning_rate))


class SAGA(optimizer.Optimizer):
    def __init__(self, learning_rate, batch_ind, total_batches):
        super(SAGA, self).__init__(False, "SAGA")
        self.learning_rate = learning_rate
        self.batch_ind = batch_ind
        self.total_batches = total_batches

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "sum", "SAGA")
            self._get_or_make_slot(v, tf.pack([tf.zeros(shape=v.get_shape()) for _ in range(self.total_batches)]),
                                   "memory", "SAGA")

    def _apply_dense(self, grad, var):
        memory = self.get_slot(var, "memory")
        memsum = tf.reduce_mean(memory, [0])
        mem = tf.gather(memory, self.batch_ind)
        delta = grad - mem + memsum
        mem_op = tf.scatter_update(memory, self.batch_ind, grad)
        return tf.group(var.assign_sub(tf.mul(delta, self.learning_rate)), mem_op)


class AdaGrad(optimizer.Optimizer):
    def __init__(self, learning_rate):
        super(AdaGrad, self).__init__(False, "AdaGrad")
        self.learning_rate = learning_rate

    def _create_slots(self, var_list):
        for v in var_list:
            acc = tf.constant(0.1, shape=v.get_shape())
            self._get_or_make_slot(v, acc, "accumulator", "AdaGrad")

    def _apply_dense(self, grad, var):
        acc = self.get_slot(var, "accumulator")
        acc = acc.assign_add(tf.square(grad))
        return var.assign_sub(tf.mul(tf.mul(grad, tf.rsqrt(acc)), self.learning_rate))