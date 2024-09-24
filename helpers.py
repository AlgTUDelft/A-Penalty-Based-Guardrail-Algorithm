import numpy as np
import pandas as pd


def save(dict_, J, f, runtime, path, name, vars=[]):
    dict_["J"] = J
    dict_["f"] = f
    dict_["runtime"] = runtime
    if vars:
        dict_["vars"] = vars
    df = pd.DataFrame(dict_)
    df.to_json(path.joinpath(name + ".json"))


def get_constraint_values(var, c_con, q):
    """
     Retrieve constraint values for value of variables.
    :param var: list, variables vector
    :param num_var: int
    :param num_con: int
    :param c: list
    :param q: list
    :param c_fac: int, transformation factor for constraint coefficients (depends on the specific algorithm, default value one)
    :param q_fac: int, transformation factor for the right side of constraints (depends on the specific algorithm, default value one)
    :return: list
    """
    constraint_values = np.matmul(c_con.numpy(), var) - q.numpy()
    return constraint_values.tolist()


def transform_Jacobian(Jacobian):
    """
    Transforms Jacobian matrix from tensor to numpy form, and transposes this matrix.
    :param Jacobian: tf.Tensor
    :return: np.ndarray
    """
    # Convert the tensor to a NumPy array
    Jacobian_np = Jacobian.numpy()
    # Transpose the NumPy array
    Jacobian_transposed = Jacobian_np.T
    return Jacobian_transposed


def P_x(var, ub, lb):
    """
    Projection operator P_x: sets all elements greater than ub to ub, and all elements smaller than lb to lb.
    :param var: np.array
    :param ub: int
    :param lb: int
    :return: np.array
    """
    var = np.maximum(np.minimum(var, ub), lb)
    return var
