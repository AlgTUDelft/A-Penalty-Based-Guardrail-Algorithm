import time
from pyscipopt import Model, exp, quicksum
import tensorflow as tf
from helpers import *
from optimization_specs import *
from tf_functions import *


def mps(num_var, num_con, c, q, ub, lb, T):
    """"
    Solution derived by mathematical programming solver (MPS).
    """
    print("MPS")
    start = time.time()
    m = Model("MPC")
    m.resetParams()
    m.setRealParam("limits/time", T)
    x = {i: m.addVar(lb=lb[i], ub=ub[i], name=f"x({i})") for i in range(num_var)}
    for i in range(num_con):
        m.addCons(quicksum(c[i][j] * x[j] for j in range(num_var)) >= q[i], name=f"constraint({i})")
    objvar = m.addVar(name="J", lb=None, ub=None)
    m.setObjective(objvar, "minimize")
    m.addCons(objvar >= quicksum(c[-1][j] * x[j] for j in range(num_var)))
    m.optimize()
    end = time.time()
    var = [m.getVal(x[i]) for i in range(num_var)]
    J = m.getVal(objvar)
    constraint_values = get_constraint_values(var=var, num_var=num_var, num_con=num_con,
                                              c=c,
                                              q=q)
    return [var], [J], [constraint_values], [end - start]


def standard_penalty_alg(num_var, num_con, c, q, ub, lb, C, initial_vector, delta, patience, grad_iter_max):
    print("PM")
    # at each new iteration of outer loop, reset gradient iterations and patience counter.
    grad_iter, grad_patience_iter = 0, 0
    var = tf.Variable(initial_vector, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    lb = tf.constant(lb, dtype=tf.float32)
    ub = tf.constant(ub, dtype=tf.float32)
    # Convert c and q to tensors
    c_obj = tf.constant(c[-1], dtype=tf.float32)  # Objective coefficients
    c_con = tf.constant(c[:-1], dtype=tf.float32)  # Constraint coefficients
    q = tf.constant(q, dtype=tf.float32)  # Constraint bounds
    start_time = time.time()
    while grad_iter < grad_iter_max:
        grads = calculate_gradient_penalty_fun(var=var, c_obj=c_obj, c_con=c_con,
                                               q=q, C=C)
        opt.apply_gradients([(grads, var)])
        # Apply variable bounds
        var.assign(tf.clip_by_value(var, lb, ub))
        grad_iter += 1
    end_time = time.time()
    var_np = var.numpy()
    J = float(np.sum(c_obj.numpy() * var_np))
    constraint_values = get_constraint_values(var=var_np, c_con=c_con, q=q)
    runtime = end_time - start_time
    return [J], [constraint_values], [var_np.tolist()], [runtime]


def ipdd(
        num_var, num_con, c, q, ub, lb, T, initial_vector, initial_lambdas, rho, grad_iter_max,
):
    print("IPDD")
    Js, constraint_values, solutions, runtimes = [], [], [], [0]
    solution = tf.Variable(initial_vector, dtype=tf.float32)
    lambdas = tf.Variable(initial_lambdas, dtype=tf.float32)
    lb = tf.constant(lb, dtype=tf.float32)
    ub = tf.constant(ub, dtype=tf.float32)
    outer_iter = 0
    # Convert c and q to tensors
    c_obj = tf.constant(c[-1], dtype=tf.float32)  # Objective coefficients
    c_con = tf.constant(c[:-1], dtype=tf.float32)  # Constraint coefficients
    q = tf.constant(q, dtype=tf.float32)  # Constraint bounds
    while runtimes[-1] < T:
        outer_iter += 1
        # at each new iteration of outer loop, reset gradient iterations and patience counter.
        grad_iter = 0
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        start_time = time.time()
        while grad_iter < grad_iter_max:
            grads = calculate_gradient_lagrangian_fun(var=solution, c_obj=c_obj, c_con=c_con, q=q, rho=rho,
                                                      lambdas=lambdas)
            opt.apply_gradients([(grads, solution)])
            # Apply variable bounds
            solution.assign(tf.clip_by_value(solution, lb, ub))
            grad_iter += 1
        solution_np = solution.numpy()
        J = np.sum(c_obj.numpy() * solution_np)
        # Compute constraint violations using TensorFlow
        constraint_value = tf.linalg.matvec(c_con, solution) - q
        # Update Lagrange multipliers
        lambdas.assign_add((1 / rho) * constraint_value)
        rho = rho * (1 / (outer_iter ** (1 / 3)))
        end_time = time.time()
        solutions.append(solution_np.tolist())
        Js.append(J)
        constraint_values.append(constraint_value.numpy())
        runtimes.append(runtimes[-1] + end_time - start_time)
    return Js, constraint_values, solutions, runtimes[1:]


def gdpa(num_var, num_con, c, q, ub, lb, T, initial_vector, initial_lambdas, step_size, perturbation_term, beta, gamma):
    print("GDPA")
    Js, constraint_values, solutions, runtimes = [], [], [], [0]
    solution_prev = initial_vector
    lambdas_prev = initial_lambdas
    outer_iter = 0
    while runtimes[-1] < T:
        outer_iter += 1
        start_time = time.time()
        solution_prev_tf = tf.Variable(solution_prev, dtype=tf.float32)
        if np.isnan(solution_prev).any():
            break
        # first equation
        # np.ndarray([dJ/dx_1, dJ/dx_2,...,dJ/dx_n])_1xn
        delta_J = np.array(calculate_gradient_obj_fun(var=solution_prev_tf, num_var=num_var, c=c))
        # tf.Tensor([df_1/dx_1,df_1/dx_2,...,df_1/dx_n],...,[df_m/dx_1,df_m/dx_2,...,df_m/dx_n])_mxn
        Jacobian = calculate_jacobian(var=solution_prev_tf, num_var=num_var, num_con=num_con, c=c, q=q)
        # np.array([df_1/dx_1,df_2/dx_1,...,df_m/dx_1],...,[df_1/dx_n,df_2/dx_n,...,df_m/dx_n])_nxm
        Jacobian = transform_Jacobian(Jacobian)
        tau_beta = beta * np.array((-1) * np.dot(c[:-1], solution_prev) + q) + (
                1 - perturbation_term) * lambdas_prev
        tau_beta_nonnegative = np.maximum(tau_beta, 0)
        solution = solution_prev - step_size * (delta_J + np.dot(Jacobian, np.transpose(tau_beta_nonnegative)))
        solution = P_x(var=solution, ub=ub, lb=lb)
        # second equation
        S_r = [i for i in range(num_con) if tau_beta[i] > 0]
        lambdas = np.zeros(num_con)
        for i in range(num_con):
            if i in S_r:
                lambdas[i] = max(0, (1 - perturbation_term) * lambdas_prev[i] + beta * (
                        sum((-1) * c[i][j] * solution[j] for j in range(num_var)) - q[i] * (-1)))
        solution_prev = solution
        lambdas_prev = lambdas
        beta = beta * outer_iter ** (1 / 3)
        step_size = (1 / (outer_iter ** (1 / 3))) * step_size
        end_time = time.time()
        Js.append(sum(c[-1][i] * solution[i] for i in range(num_var)))
        constraint_values.append(np.dot(c[:-1], solution) - q)
        solutions.append(solution)
        runtimes.append(end_time - start_time + runtimes[-1])
    return Js, constraint_values, solutions, runtimes[1:]


def pga(num_var, num_con, c, q, ub, lb, T, C, initial_vector, delta, patience, grad_iter_max):
    print("PGA")
    Js, constraint_values, variables, runtimes = [], [], [], [0]
    epsilon = [0] * num_con
    outer_iteration = 0
    lb = tf.constant(lb, dtype=tf.float32)
    ub = tf.constant(ub, dtype=tf.float32)
    # Convert c and q to tensors
    c_obj = tf.constant(c[-1], dtype=tf.float32)  # Objective coefficients
    c_con = tf.constant(c[:-1], dtype=tf.float32)  # Constraint coefficients
    q = tf.constant(q, dtype=tf.float32)  # Constraint bounds
    while runtimes[-1] < T:
        outer_iteration += 1
        # at each new iteration of outer loop, reset gradient iterations and patience counter.
        grad_iter, grad_patience_iter = 0, 0
        var = tf.Variable(initial_vector, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        start_time = time.time()
        q_pga = tf.constant([np.float32(epsilon[i] + x) for i, x in enumerate(q)], dtype=tf.float32)
        while grad_iter < grad_iter_max:
            grads = calculate_gradient_penalty_fun(var=var, c_obj=c_obj, c_con=c_con, q=q_pga, C=C)
            opt.apply_gradients([(grads, var)])
            # Apply variable bounds
            var.assign(tf.clip_by_value(var, lb, ub))
            grad_iter += 1
        end_time = time.time()
        var_np = var.numpy()
        J = np.sum(c_obj.numpy() * var_np)
        constraint_value = get_constraint_values(var=var_np, c_con=c_con, q=q)
        epsilon = [np.float32(eps - (1 / (outer_iteration ** (1 / 3))) * min(0, constraint_value[i])) for i, eps in
                   enumerate(epsilon)]
        variables.append(var_np.tolist())
        Js.append(J)
        constraint_values.append(constraint_value)
        runtimes.append(runtimes[-1] + end_time - start_time)
    return Js, constraint_values, variables, runtimes[1:]
