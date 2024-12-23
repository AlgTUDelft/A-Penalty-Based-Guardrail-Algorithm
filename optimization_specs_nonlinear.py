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
            return pickle.load(f)
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
        c_json_path = path.with_name(path.stem + '_w.json')
        with open(c_json_path, 'w') as f_json:
            json.dump(fun["w"], f_json, indent=2)
        return fun


def verify_initial_vector(initial_solution, program_spec):
    for i in range(program_spec["num_con"]):
        left_side = 0
        for j in range(program_spec["num_var"]):
            left_side += program_spec["c"][i][j] * initial_solution[j]
        if left_side - program_spec["q"][i] < 0:
            print("Left side is {}".format(left_side))
            raise ValueError("Infeasible initial solution!")
    return True


def get_initial_vector(initial_solution, problem_spec):
    try:
        # First, try the all-ones vector
        if verify_initial_vector(initial_solution=initial_solution, program_spec=problem_spec):
            print("Feasible initial solution!")
            return np.array(initial_solution)
    except ValueError:
        raise ValueError("Infeasible initial solution!")


PROBLEM_SPECS = {"fun_1": {
    "num_var": 2,
    "num_con": 1,
    "num_layers": [2],
    "num_neuron_per_layer": [[3, 1]],
    "activation_fun": [["elu", "elu"]],
    "w": [[[[0, 1, 3], [3, 2, 1]], [[2], [1], [1]]]],
    "c_obj": [1, 2],
    "q": [4],
    "T": 100,
    "ub": [float(50)] * 2,
    "lb": [-float(50)] * 2
},
    "fun_2": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [2, 2],
        "num_neuron_per_layer": [[3, 1], [2, 1]],
        "activation_fun": [["relu", "relu"], ["elu", "relu"]],
        "w": [[[[2, 1, 3], [0, 5, 2]], [[1], [5], [2]]], [[[1, 2], [3, 4]], [[5], [4]]]],
        "c_obj": [1, 10],
        "q": [4, 16],
        "T": 100,
        "ub": [float(25)] * 2,
        "lb": [-float(25)] * 2
    },
    "fun_3": {
        "num_var": 2,
        "num_con": 2,
        "num_layers": [2, 1],
        "num_neuron_per_layer": [[2, 1], [1]],
        "activation_fun": [["sigmoid", "linear"], ["tanh"]],
        # first brackets are per constraint, second brackets are per layer of constraint
        "w": [[[[0.2, 0.5], [0.5, 0.7]], [[2], [3]]], [[[1], [2]]]],
        "c_obj": [1, 1],
        "q": [4, 1],
        "T": 100,
        "ub": [float(25)] * 2,
        "lb": [-float(25)] * 2
    },
    "fun_4": {
        "num_var": 5,
        "num_con": 4,
        "num_layers": [1, 1, 1, 1],
        "num_neuron_per_layer": [[1], [1], [1], [1]],
        "activation_fun": [["linear"], ["linear"], ["linear"], ["linear"]],
        # first brackets are per constraint, second brackets are per layer of constraint
        "w": [[[[3], [2], [1], [1], [0]]], [[[1], [4], [2], [0], [1]]], [[[0], [1], [1], [3], [2]]],
              [[[2], [0], [1], [4], [1]]]],
        "c_obj": [2, 3, 4, 5, 6],
        "q": [50, 35, 40, 45],
        "T": 200,
        "ub": [float("inf")] * 5,
        "lb": [0] * 5
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
        "T": 100,
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
        "T": 100,
        "ub": [float(25)] * 2,
        "lb": [float(0)] * 2
    },
    "fun_7": create_nonlinear_program(num_var=5, num_con=5, num_layers=[3, 1, 2, 3, 1],
                                      num_neuron_per_layer=[[3, 2, 1], [1], [2, 1], [2, 2, 1], [1]],
                                      activation_fun=[["relu", "selu", "relu"], ["elu"], ["relu", "leaky_relu"],
                                                      ["elu", "relu", "selu"], ["relu"]], T=500,
                                      path=Path("data/nonlinear/fun_7").joinpath("fun_7.pickle"))
}

GRADIENT_SPECS = {
    "fun_1": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_1"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_1"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.01,
        "rho": 1
    },
    "fun_2": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_2"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_2"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.1,
        "rho": 1
    },
    "fun_3": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_3"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_3"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.1,
        "rho": 1
    },
    "fun_4": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_4"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_4"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "rho": 1
    },
    "fun_5": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_5"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_5"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.1,
        "rho": 1
    },
    "fun_6": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_5"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_5"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.1,
        "rho": 1
    },
    "fun_7": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_7"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_7"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.5,
        "rho": 1
    }
}

EVAL_SPECS = {
    "fun_2": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [10] * PROBLEM_SPECS["fun_2"]["num_var"], [0] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [-10] * PROBLEM_SPECS["fun_2"]["num_var"], [-25] * PROBLEM_SPECS["fun_2"]["num_var"]],
              "Cs": [5, 1, 0.5, 0.25, 0.1, 0.01]},
    "fun_3": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [10] * PROBLEM_SPECS["fun_2"]["num_var"], [0] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [-10] * PROBLEM_SPECS["fun_2"]["num_var"], [-25] * PROBLEM_SPECS["fun_2"]["num_var"]],
              "Cs": [5, 1, 0.5, 0.25, 0.1, 0.01]},
    "fun_5": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_2"]["num_var"]],
              "Cs": [5, 1, 0.5, 0.25, 0.1, 0.01]},
    "fun_6": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [0] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [-25] * PROBLEM_SPECS["fun_2"]["num_var"]],
              "Cs": [5, 1, 0.5, 0.25, 0.1, 0.01]}

}
