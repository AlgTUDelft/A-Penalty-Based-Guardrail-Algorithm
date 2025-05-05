import numpy as np
import tensorflow as tf


@tf.function
def calculate_gradient_lagrangian_fun(var, c_obj, c_con, q, rho, lambdas):
    with tf.GradientTape() as tape:
        tape.watch(var)
        f_obj = tf.tensordot(c_obj, var, axes=1)
        Ax = tf.linalg.matvec(c_con, var)
        violations = Ax - q
        penalty = (1 / rho) * tf.reduce_sum(tf.square(violations))
        linear_term = tf.tensordot(lambdas, violations, axes=1)
        f = f_obj + penalty + linear_term
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_gradient_penalty_fun(var, c_obj, c_con, q, C):
    with tf.GradientTape() as tape:
        tape.watch(var)
        # f = tf.math.add_n([c_obj[i] * var[i] for i in range(num_var)])
        f_obj = tf.tensordot(c_obj, var, axes=1)
        Ax = tf.linalg.matvec(c_con, var)  # Shape: (num_con,)

        # Constraint violations
        # If penalizing when Ax < q:
        violations = tf.nn.relu(q - Ax)  # Shape: (num_con,)

        # Penalty term: penalty = C * sum(violations^2)
        penalty = C * tf.reduce_sum(tf.square(violations))  # Scalar value
        f = f_obj + penalty
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_gradient_obj_fun(var, num_var, c):
    with tf.GradientTape() as tape:
        tape.watch(var)
        f = tf.math.add_n([c[-1][i] * var[i] for i in range(num_var)])
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_jacobian(var, num_var, num_con, c, q):
    with tf.GradientTape() as tape:
        tape.watch(var)
        for i in range(num_con):
            g = tf.stack(
                [tf.math.add_n([c[i][j] * (-1) * var[j] for j in range(num_var)]) + q[i] for i in range(num_con)])
    Jacobian = tape.jacobian(g, var)
    return Jacobian
