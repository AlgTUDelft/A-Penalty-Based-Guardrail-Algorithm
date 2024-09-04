import numpy as np
import tensorflow as tf
from scipy import optimize


def J(x):
    return x[0] ** 2 * x[1] + 10 * x[1] ** 2 * x[0] ** 2


def g(x):
    return [x[0], x[0] + 5 * x[1]]


@tf.function
def calculate_gradient(var):
    with tf.GradientTape() as tape:
        tape.watch(var)
        f = tf.math.add(var[:, 0] ** 2 * var[:, 1], 10 * var[:, 1] ** 2 * var[:, 0] ** 2)
    grads = tape.gradient(f, var)
    return grads


@tf.function
def calculate_jacobian(var):
    g = []
    with tf.GradientTape() as tape:
        tape.watch(var)
        g = tf.stack([var[:, 0], tf.math.add(var[:, 0], 5 * var[:, 1])])
    Jacobian = tape.jacobian(g, var)
    return Jacobian


def transform_Jacobian(Jacobian):
    Jacobian_ = []
    for i, Jac in enumerate(Jacobian):
        Jacobian_.append(Jac[0][0].tolist())
    return np.array(Jacobian_)


"""
x = [3, 5]
y = optimize.approx_fprime(x, J)
print(y)
Jacobian = optimize.approx_fprime(x, g)
print(Jacobian)
var = tf.Variable(np.array([3, 5]).reshape(1, -1), dtype=tf.float32)
grads = calculate_gradient(var)
print(grads)
"""
a = [-1, -2]
b=0.1*a
print(b)
var = tf.Variable(np.array([3, 5]).reshape(1, -1), dtype=tf.float32)
Jacobian = np.array(calculate_jacobian(var))
Jacobian = transform_Jacobian(Jacobian)
Jacobian_transposed = np.transpose(Jacobian)
