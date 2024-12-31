import json
import numpy as np
import math
import random
import pickle

from pathlib import Path


def create_weights(num_var, num_neuron_per_layer):
    """
    Create weights of compiled deep neural network using random initialization.
    :param num_var: int
    :param num_neuron_per_layer: list
    :return: list
    """
    weights = []
    for num_neurons in num_neuron_per_layer:
        num_neurons = [num_var] + num_neurons
        weights_constraint = []
        for i in range(len(num_neurons) - 1):
            weights_layer = []
            for j in range(num_neurons[i]):
                weights_neuron = [random.randint(1, 10) for _ in range(num_neurons[i + 1])]
                weights_layer.append(weights_neuron)
            weights_constraint.append(weights_layer)
        weights.append(weights_constraint)
    return weights


def create_nonlinear_program(num_var, num_con, num_layers, num_neuron_per_layer, activation_fun, T, path):
    """
    Create large nonlinear program with randomly specified weights, corresponding to the number of constraints, layers per constraint and neurons per layer.
    :param num_var: int
    :param num_con: int
    :param num_layers: list
    :param num_neuron_per_layer: list
    :param activation_fun: list
    :param T: int
    :param path: path
    :return: dict
    """
    if Path(path).exists():
        print("Function specification already exists")
        with open(path, 'rb') as f:
            dict_ = pickle.load(f)
            dict_["T"] = 500
            print(dict_)
            return dict_
    else:
        print("Function specification created")
        fun = {"num_var": num_var,
               "num_con": num_con,
               "num_layers": num_layers,
               "num_neuron_per_layer": num_neuron_per_layer,
               "activation_fun": activation_fun,
               "w": create_weights(num_var=num_var, num_neuron_per_layer=num_neuron_per_layer),
               "c_obj": [random.randint(1, 10) for _ in range(num_var)],
               "q": [random.randint(1, 10) for _ in range(num_con)],
               "T": T,
               "ub": [float(25)] * num_var,
               "lb": [-float(25)] * num_var
               }
        with open(path, 'wb') as f:
            pickle.dump(fun, f)
        # Save 'fun["q"]' into a JSON file for visual inspection
        q_json_path = path.with_name(path.stem + '_q.json')
        with open(q_json_path, 'w') as f_json:
            json.dump(fun["q"], f_json, indent=2)
        # Save 'fun["w"]' into a JSON file for visual inspection
        w_json_path = path.with_name(path.stem + '_w.json')
        with open(w_json_path, 'w') as f_json:
            json.dump(fun["w"], f_json, indent=2)
        return fun


def verify_initial_vector(initial_solution, program_spec):
    """
    Verify weather proposed initial solution is feasible and thus can be used as the initial solution.
    :param initial_solution: np.array
    :param program_spec: dict
    :return: bool
    """
    w = program_spec["w"]
    num_var = program_spec["num_var"]
    num_con = program_spec["num_con"]
    num_neuron_per_layer = program_spec["num_neuron_per_layer"]
    activation_fun = program_spec["activation_fun"]
    for i in range(num_con):
        w_con = w[i]
        num_neuron_per_layer_con = [num_var] + num_neuron_per_layer[i]
        activation_fun_con = activation_fun[i]
        layer_output = initial_solution
        for j in range(1, len(num_neuron_per_layer_con)):
            w_layer = w_con[j - 1]
            activation_fun_layer = activation_fun_con[j - 1]
            neuron_per_layer_prev = num_neuron_per_layer_con[j - 1]
            neuron_per_layer_curr = num_neuron_per_layer_con[j]
            temp = []
            for k in range(neuron_per_layer_curr):
                neuron = 0
                for n in range(neuron_per_layer_prev):
                    neuron += layer_output[n] * w_layer[n][k]
                if activation_fun_layer == "exponential":
                    neuron = math.exp(neuron)
                elif activation_fun_layer == "sigmoid":
                    neuron = 1 / (1 + math.exp(-neuron))
                elif activation_fun_layer == "tanh":
                    neuron = (math.exp(neuron) - math.exp(-neuron)) / (math.exp(neuron) + math.exp(-neuron))
                else:
                    neuron = neuron
                temp.append(neuron)
            layer_output = temp
        left_side = layer_output[0]
        if left_side - program_spec["q"][i] < 0:
            print("Left side is {}".format(left_side))
            raise ValueError("Infeasible initial solution!")
    return True


def get_initial_vector(initial_solution, problem_spec):
    """
    Return initial solution if it is feasible, otherwise throw ValueError
    :param initial_solution: np.array
    :param problem_spec: dict
    :return: np.array or Value Error
    """
    try:
        # First, try the all-ones vector
        if verify_initial_vector(initial_solution=initial_solution, program_spec=problem_spec):
            print("Feasible initial solution!")
            return np.array(initial_solution)
    except ValueError:
        raise ValueError("Infeasible initial solution!")


PROBLEM_SPECS = {
    "fun_4": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [2, 2],
        "num_neuron_per_layer": [[3, 1], [2, 1]],
        "activation_fun": [["relu", "relu"], ["elu", "relu"]],
        "w": [[[[2, 1, 3], [0, 5, 2]], [[1], [5], [2]]], [[[1, 2], [3, 4]], [[5], [4]]]],
        "c_obj": [1, 10],
        "q": [4, 16],
        "T": 500,
        "ub": [float(25)] * 2,
        "lb": [-float(25)] * 2
    },
    "fun_5": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [1, 1],
        "num_neuron_per_layer": [[1], [1]],
        "activation_fun": [["relu"], ["elu"]],
        # first brackets are per constraint, second brackets are per layer of constraint
        "w": [[[[2], [3]]], [[[3], [4]]]],
        "c_obj": [1, 1],
        "q": [5.75, -0.5],
        "T": 500,
        "ub": [float(25)] * 2,
        "lb": [-float(25)] * 2
    },
    "fun_6": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [1, 1],
        "num_neuron_per_layer": [[1], [1]],
        "activation_fun": [["relu"], ["tanh"]],
        # first brackets are per constraint, second brackets are per layer of constraint
        "w": [[[[2], [1]]], [[[1], [4]]]],
        "c_obj": [1, 1],
        "q": [1, 0.5],
        "T": 500,
        "ub": [float(25)] * 2,
        "lb": [float(0)] * 2
    },
    "fun_7": create_nonlinear_program(num_var=5, num_con=5, num_layers=[3, 1, 2, 3, 1],
                                      num_neuron_per_layer=[[3, 2, 1], [1], [2, 1], [2, 2, 1], [1]],
                                      activation_fun=[["relu", "selu", "relu"], ["elu"], ["relu", "leaky_relu"],
                                                      ["elu", "relu", "selu"], ["relu"]], T=500,
                                      path=Path("data/nonlinear/fun_7").joinpath("fun_7.pickle")),
    "fun_8": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [2, 1],
        "num_neuron_per_layer": [[2, 1], [1]],
        "activation_fun": [["sigmoid", "linear"], ["tanh"]],
        # first brackets are per constraint, second brackets are per layer of constraint
        "w": [[[[0.2, 0.5], [0.5, 0.7]], [[2], [3]]], [[[1], [2]]]],
        "c_obj": [1, 1],
        "q": [4, 1],
        "T": 500,
        "ub": [float(25)] * 2,
        "lb": [-float(25)] * 2
    },
}

GRADIENT_SPECS = {
    "fun_4": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_4"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_4"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
    "fun_5": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_5"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_5"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
    "fun_6": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_6"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_6"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
    "fun_7": {
        "initial_vector": get_initial_vector(initial_solution=np.array([25] * PROBLEM_SPECS["fun_7"]["num_var"]),
                                             problem_spec=PROBLEM_SPECS["fun_7"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_7"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
    "fun_8": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_8"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_8"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
}

EVAL_SPECS = {
    "fun_4": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_4"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_4"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_4"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1]},
    "fun_5": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_5"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_5"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_5"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1]},
    "fun_6": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_6"]["num_var"],
                                  [10] * PROBLEM_SPECS["fun_6"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_6"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1]},
    "fun_7": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_7"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_7"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_7"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1]},
    "fun_8": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_8"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_8"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_8"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1]},

}

PLOT_SPECS = {"fun_4": {"lb_obj": -100, "ub_obj": 100}, "fun_5": {"lb_obj": -10, "ub_obj": 30},
              "fun_7": {"lb_obj": -800, "ub_obj": 200}, "fun_6": {"lb_obj": -2, "ub_obj": 10},
              "fun_8": {"lb_obj": -60, "ub_obj": 40}}
