import csv
import numpy as np
import time
import tensorflow as tf
from pyscipopt import Model, exp, quicksum
import pandas as pd
from collections import defaultdict
# from plot import plot
from pathlib import Path
from tf_functions import *


def save(dict_, J, f, runtime, name):
    dict_["J"] = J
    dict_["f"] = f
    dict_["runtime"] = runtime
    df = pd.DataFrame(dict_)
    df.to_json("data/fun_1/" + name + ".json")


def get_constraint_values(var, num_var, num_con, c, q, c_fac=1, q_fac=1):
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
    x = [
        sum(c[i][j] * c_fac * var[j] for j in range(num_var)) - q[i] * q_fac
        for i in range(num_con)
    ]
    return x


def get_constraint_values_subset(var, num_var, num_con, c, q, S_r, c_fac=1, q_fac=1):
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
    x = [
        sum(c[i][j] * c_fac * var[j] for j in range(num_var)) - q[i] * q_fac
        for i in range(num_con) if i in S_r
    ]
    return x


def transform_Jacobian(Jacobian):
    """
    Transforms Jacobian matrix from tensor to numpy form, and transposes this matrix.
    :param Jacobian: tf.Tensor
    :return: np.ndarray
    """
    Jacobian_ = []
    for i, Jac in enumerate(Jacobian):
        Jacobian_.append(list(Jac[0][0]))
    return np.transpose(np.array(Jacobian_))


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


def mps(num_var, num_con, c, q, ub, lb):
    """"
    Solution derived by mathematical programming solver (MPS).
    """
    start = time.time()
    m = Model("MPC")
    m.resetParams()
    x = {i: m.addVar(lb=lb[i], ub=ub[i], name=f"x({i})") for i in range(num_con)}
    for i in range(num_con):
        m.addCons(quicksum(c[i][j] * x[j] for j in range(num_var)) >= q[i], name=f"constraint({i})")
    objvar = m.addVar(name="J", lb=None, ub=None)
    m.setObjective(objvar, "minimize")
    m.addCons(objvar >= quicksum(c[-1][j] * x[j] for j in range(num_var)))
    m.optimize()
    end = time.time()
    var = [m.getVal(x[i]) for i in range(num_con)]
    J = m.getVal(objvar)
    constraint_values = get_constraint_values(var=var, num_var=problem_spec["num_var"], num_con=problem_spec["num_con"],
                                              c=problem_spec["c"],
                                              q=problem_spec["q"])
    return var, [J], [constraint_values], [end - start]


def standard_penalty_alg(num_var, num_con, c, q, ub, lb, C, initial_vector, delta, patience, grad_iter_max):
    # at each new iteration of outer loop, reset gradient iterations and patience counter.
    grad_iter, grad_patience_iter = 0, 0
    var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    grads_prev = tf.convert_to_tensor(
        np.array([100] * num_con, dtype=np.float32).reshape(1, num_con)
    )
    start_time = time.time()
    while grad_patience_iter < patience and grad_iter < grad_iter_max:
        grads = calculate_gradient_penalty_fun(var=var, num_var=num_var, num_con=num_con, c=c, q=q, C=C)
        delta_tf = tf.abs(grads_prev - grads)
        # if change in gradient is less than parameter delta, increase patience by one.
        # however, if it is greater than delta, reset patience.
        if tf.reduce_any(delta_tf < delta, 1):
            grad_patience_iter += 1
        else:
            grad_patience_iter = 0
        grads_prev = grads
        zipped = zip([grads], [var])
        opt.apply_gradients(zipped)
        # var = tf.Variable(tf.where(var< 0, tf.zeros_like(var), var))
        grad_iter += 1
    end_time = time.time()
    var = list(np.array(var[0]))
    var = [lb[i] if x < lb[i] else ub[i] if x > ub[i] else x for i, x in enumerate(var)]
    J = sum(c[-1][i] * var[i] for i in range(num_var))
    constraint_values = get_constraint_values(var=var, num_var=num_var, num_con=num_con, c=c, q=q)
    runtime = end_time - start_time
    return [J], [constraint_values], var, [runtime]


def gdpa(num_var, num_con, c, q, ub, lb, T, initial_vector, initial_lambdas, step_size, perturbation_term, beta, gamma):
    Js, constraint_values, solutions, runtimes = [], [], [], [0]
    solution_prev = initial_vector
    lambdas_prev = initial_lambdas
    outer_iter = 0
    while runtimes[-1] < T:
        outer_iter += 1
        start_time = time.time()
        solution_prev_tf = tf.Variable(solution_prev.reshape(1, -1), dtype=tf.float32)
        # first equation
        # np.ndarray([dJ/dx_1, dJ/dx_2,...,dJ/dx_n])_1xn
        delta_J = np.array(calculate_gradient_obj_fun(var=solution_prev_tf, num_var=num_var, c=c))[0]
        # tf.Tensor([df_1/dx_1,df_1/dx_2,...,df_1/dx_n],...,[df_m/dx_1,df_m/dx_2,...,df_m/dx_n])_mxn
        Jacobian = calculate_jacobian(var=solution_prev_tf, num_var=num_var, num_con=num_con, c=c, q=q)
        # np.array([df_1/dx_1,df_2/dx_1,...,df_m/dx_1],...,[df_1/dx_n,df_2/dx_n,...,df_m/dx_n])_nxm
        Jacobian = transform_Jacobian(Jacobian)
        tau_beta = beta * np.array(
            get_constraint_values(var=solution_prev, num_var=num_var, num_con=num_con, c=c, q=q, c_fac=-1,
                                  q_fac=-1)) + (1 - perturbation_term) * lambdas_prev
        tau_beta[tau_beta < 0] = 0
        solution = solution_prev - step_size * (delta_J + np.dot(Jacobian, np.transpose(tau_beta)))
        solution = P_x(var=solution, ub=ub, lb=lb)
        # second equation
        S_r = [i for i in range(num_con) if tau_beta[i] > 0]
        lambdas = np.zeros(num_con)
        for i in range(num_con):
            if i in S_r:
                lambdas[i] = max(0, (1 - perturbation_term) * lambdas_prev[i] + beta * (
                        sum(c[i][j] * (-1) * solution[j] for j in range(num_var)) - q[i] * (-1)))
        solution_prev = solution
        lambdas_prev = lambdas
        beta = beta * outer_iter ** (1 / 3)
        step_size = (1 / (outer_iter ** (1 / 3))) * step_size
        end_time = time.time()
        Js.append(sum(c[-1][i] * solution[i] for i in range(num_var)))
        constraint_values.append(get_constraint_values(var=solution, num_var=num_var, num_con=num_con, c=c, q=q))
        solutions.append(solution)
        runtimes.append(end_time - start_time + runtimes[-1])
    return Js, constraint_values, solutions, runtimes[1:]


def pga(num_var, num_con, c, q, ub, lb, T, C, initial_vector, delta, patience, grad_iter_max):
    Js, constraint_values, variables, runtimes = [], [], [], [0]
    epsilon = [0] * num_con
    outer_iteration = 0
    while runtimes[-1] < T:
        outer_iteration += 1
        # at each new iteration of outer loop, reset gradient iterations and patience counter.
        grad_iter, grad_patience_iter = 0, 0
        var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        grads_prev = tf.convert_to_tensor(
            np.array([100] * num_con, dtype=np.float32).reshape(1, num_con))
        start_time = time.time()
        q_pga = [np.float32(epsilon[i] + x) for i, x in enumerate(q)]
        while grad_patience_iter < patience and grad_iter < grad_iter_max:
            grads = calculate_gradient_penalty_fun(var=var, num_var=num_var, num_con=num_con, c=c, q=q_pga, C=C)
            delta_tf = tf.abs(grads_prev - grads)
            # if change in gradient is less than parameter delta, increase patience by one.
            # however, if it is greater than delta, reset patience.
            if tf.reduce_any(delta_tf < delta, 1):
                grad_patience_iter += 1
            else:
                grad_patience_iter = 0
            grads_prev = grads
            zipped = zip([grads], [var])
            opt.apply_gradients(zipped)
            # var = tf.Variable(tf.where(var< 0, tf.zeros_like(var), var))
            grad_iter += 1
        end_time = time.time()
        var = list(np.array(var[0]))
        var = [lb[i] if x < lb[i] else ub[i] if x > ub[i] else x for i, x in enumerate(var)]
        J = sum(c[-1][i] * var[i] for i in range(num_var))
        constraint_value = get_constraint_values(var=var, num_var=num_var, num_con=num_con, c=c, q=q)
        epsilon = [np.float32(eps - (1 / (outer_iteration ** (1 / 3))) * min(0, constraint_value[i])) for i, eps in
                   enumerate(epsilon)]
        variables.append(var)
        Js.append(J)
        constraint_values.append(constraint_value)
        runtimes.append(runtimes[-1] + end_time - start_time)
    return Js, constraint_values, variables, runtimes[1:]


if __name__ == "__main__":
    path: Path = Path("data/fun_1")
    mps_dict, pm_lb_dict, pm_ub_dict, gdpa_dict, pga_dict = {}, {}, {}, {}, {}
    problem_spec = {
        "num_con": 2,  # number of constraints
        "num_var": 2,  # number of variables
        "c": [[1, 0], [1, 2], [1, 10]],  # coefficients of constraints (the last constraint refers t objective function)
        "q": [1, 5],  # right-hand side of constraints
        "T": 200  # time limit
    }
    problem_spec["ub"] = [float("inf")] * problem_spec["num_var"]
    problem_spec["lb"] = [0] * problem_spec["num_var"]
    grad_spec = {"initial_vector": [20, 20], "patience": 50, "delta": 0.000001, "grad_iter_max": 25000}
    """
    var, J_mps, constraint_values_mps, runtime_mps = mps(num_var=problem_spec["num_var"],
                                                         num_con=problem_spec["num_con"], c=problem_spec["c"],
                                                         q=problem_spec["q"],
                                                         ub=problem_spec["ub"], lb=problem_spec["lb"])
    save(dict_=mps_dict, J=J_mps, f=constraint_values_mps, runtime=runtime_mps, name="mps")
    J_pm_lb, constraint_values_pm_lb, var, runtime_pm_lb = standard_penalty_alg(num_var=problem_spec["num_var"],
                                                                                num_con=problem_spec["num_con"],
                                                                                c=problem_spec["c"],
                                                                                q=problem_spec["q"],
                                                                                ub=problem_spec["ub"],
                                                                                lb=problem_spec["lb"], C=1,
                                                                                initial_vector=grad_spec[
                                                                                    "initial_vector"],
                                                                                delta=grad_spec["delta"],
                                                                                patience=grad_spec["patience"],
                                                                                grad_iter_max=grad_spec[
                                                                                    "grad_iter_max"])
    save(dict_=pm_lb_dict, J=J_pm_lb, f=constraint_values_pm_lb, runtime=runtime_pm_lb, name="pm_lb")
    J_pm_ub, constraint_values_pm_ub, var, runtime_pm_ub = standard_penalty_alg(num_var=problem_spec["num_var"],
                                                                                num_con=problem_spec["num_con"],
                                                                                c=problem_spec["c"],
                                                                                q=problem_spec["q"],
                                                                                ub=problem_spec["ub"],
                                                                                lb=problem_spec["lb"], C=100,
                                                                                initial_vector=grad_spec[
                                                                                    "initial_vector"],
                                                                                delta=grad_spec["delta"],
                                                                                patience=grad_spec["patience"],
                                                                                grad_iter_max=grad_spec[
                                                                                    "grad_iter_max"])
    save(dict_=pm_ub_dict, J=J_pm_ub, f=constraint_values_pm_ub, runtime=runtime_pm_ub, name="pm_ub")
    J_gdpa, constraint_values_gdpa, var, runtime_gdpa = gdpa(num_var=problem_spec["num_var"],
                                                             num_con=problem_spec["num_con"],
                                                             c=problem_spec["c"], q=problem_spec["q"],
                                                             ub=problem_spec["ub"],
                                                             lb=problem_spec["lb"],
                                                             T=problem_spec["T"],
                                                             initial_vector=np.array(grad_spec["initial_vector"]),
                                                             initial_lambdas=np.array([0, 0]),
                                                             step_size=1,
                                                             perturbation_term=0.9,
                                                             beta=0.9, gamma=1)
    save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, name="gdpa")
    """
    J_pga, constraint_values_pga, var, runtime_pga = pga(num_var=problem_spec["num_var"],
                                                         num_con=problem_spec["num_con"],
                                                         c=problem_spec["c"],
                                                         q=problem_spec["q"], ub=problem_spec["ub"],
                                                         lb=problem_spec["lb"], T=problem_spec["T"], C=1,
                                                         initial_vector=grad_spec["initial_vector"],
                                                         delta=grad_spec["delta"],
                                                         patience=grad_spec["patience"],
                                                         grad_iter_max=grad_spec["grad_iter_max"])
    save(dict_=pga_dict, J=J_pga, f=constraint_values_pga, runtime=runtime_pga, name="pga")
