import numpy as np

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
    "activation_fun": [["relu", "relu"], ["relu", "relu"]],
    "w": [[[[2, 1, 3], [0, 5, 2]], [[1], [0.5], [2]]], [[[1, 2], [3, 4]], [[1], [2]]]],
    "c_obj": [1, 10],
    "q": [4, 6],
    "T": 100,
    "ub": [float(25)] * 2,
    "lb": [-float(25)] * 2
},
"fun_3": {
    "num_var": 2,
    "num_con": 2,
    "num_layers": [1, 1],
    "num_neuron_per_layer": [[1], [1]],
    "activation_fun": [["linear"], ["linear"]],
    # first brackets are per constraint, second brackets are per layer of constraint
    "w": [[[[1], [0]]], [[[1], [2]]]],
    "c_obj": [1, 10],
    "q": [1, 5],
    "T": 100,
    "ub": [float("inf")] * 2,
    "lb": [0] * 2
},
"fun_4": {
    "num_var": 5,
    "num_con": 4,
    "num_layers": [1, 1, 1, 1],
    "num_neuron_per_layer": [[1], [1], [1], [1]],
    "activation_fun": [["linear"], ["linear"], ["linear"], ["linear"]],
    # first brackets are per constraint, second brackets are per layer of constraint
    "w": [[[[3], [2], [1], [1], [0]]], [[[1], [4], [2], [0], [1]]], [[[0], [1], [1], [3], [2]]], [[[2], [0], [1], [4], [1]]]],
    "c_obj": [2, 3, 4, 5, 6],
    "q": [50, 35, 40, 45],
    "T": 200,
    "ub": [float("inf")] * 5,
    "lb": [0] * 5
}
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
        "C": 20,
        "rho": 1
    },
    "fun_3": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_3"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_3"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 5000,
        "C": 0.5,
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
    }
    }
