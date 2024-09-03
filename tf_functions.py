import numpy as np
import tensorflow as tf


@tf.function
def calculate_gradient_penalty_fun(var, num_var, num_con, c, q, C):
    with tf.GradientTape() as tape:
        tape.watch(var)
        f = tf.math.add_n([c[-1][i] * var[:, i] for i in range(num_var)])
        for i in range(num_con):
            f = tf.math.add(f, C * (
                tf.math.maximum(float(0), (tf.math.add_n([c[i][j] * var[:, j] for j in range(num_var)]) - q[i]) ** 2)))
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_gradient_obj_fun(var, num_var, c):
    with tf.GradientTape() as tape:
        tape.watch(var)
        f = tf.math.add_n([c[-1][i] * var[:, i] for i in range(num_var)])
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_jacobian(var, num_var, num_con, c, q):
    with tf.GradientTape() as tape:
        tape.watch(var)
        for i in range(num_con):
            g = tf.stack(
                [tf.math.add_n([c[i][j] * (-1) * var[:, j] for j in range(num_var)]) + q[i] for i in range(num_con)])
    Jacobian = tape.jacobian(g, var)
    return Jacobian
