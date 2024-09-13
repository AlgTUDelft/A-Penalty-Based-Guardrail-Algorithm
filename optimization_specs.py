from typing import Dict, List, Union

PROBLEM_SPECS = {
    "fun_1": {
        "num_con": 2,
        "num_var": 2,
        "c": [[1, 0], [1, 2], [1, 10]],
        "q": [1, 5],
        "T": 200,
        "ub": [float("inf")] * 2,
        "lb": [0] * 2
    },
    "fun_2": {
        "num_con": 2,
        "num_var": 2,
        "c": [[1, 2], [1, 0], [1, 10]],
        "q": [5, 1],
        "T": 200,
        "ub": [float("inf")] * 2,
        "lb": [0] * 2
    },
    "fun_3": {
        "num_con": 4,
        "num_var": 5,
        "c": [[3, 2, 1, 1, 0], [1, 4, 2, 0, 1], [0, 1, 1, 3, 2], [2, 0, 1, 4, 1], [2, 3, 4, 5, 6]],
        "q": [50, 35, 40, 45],
        "T": 500,
        "ub": [float("inf")] * 5,
        "lb": [0] * 5
    },
    "fun_4": {
        "num_con": 2,
        "num_var": 2,
        "c": [[1, 2], [3, 1], [1, 1]],
        "q": [4, 5],
        "T": 50,
        "ub": [float("inf")] * 2,
        "lb": [0] * 2
    }
}

GRADIENT_SPECS = {
    "fun_1": {
        "initial_vector": [25, 25],
        "initial_lambdas": [0, 0],
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 25000
    },
    "fun_2": {
        "initial_vector": [25, 25],
        "initial_lambdas": [0, 0],
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 25000
    },
    "fun_3": {
        "initial_vector": [25] * PROBLEM_SPECS["fun_3"]["num_var"],
        "initial_lambdas": [0] * PROBLEM_SPECS["fun_3"]["num_con"],
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 25000
    },
    "fun_4": {
        "initial_vector": [5] * PROBLEM_SPECS["fun_4"]["num_var"],
        "initial_lambdas": [0] * PROBLEM_SPECS["fun_4"]["num_con"],
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 25000
    },
}

EVAL_SPECS = {
    "fun_1": {"initial_vectors": [[50, 50], [25, 25], [20, 20], [10, 10], [5, 5], [0, 0]],
              "Cs": [10, 5, 1, 0.75, 0.5, 0.25, 0.1, 0.01]},
    "fun_2": {"initial_vectors": [[50, 50], [25, 25], [20, 20], [10, 10], [5, 5], [0, 0]],
              "Cs": [10, 5, 1, 0.75, 0.5, 0.25, 0.1, 0.01]},
    "fun_3": {"initial_vectors": [[50] * PROBLEM_SPECS["fun_3"]["num_var"], [25] * PROBLEM_SPECS["fun_3"]["num_var"],
                                  [20] * PROBLEM_SPECS["fun_3"]["num_var"], [10] * PROBLEM_SPECS["fun_3"]["num_var"],
                                  [5] * PROBLEM_SPECS["fun_3"]["num_var"], [0] * PROBLEM_SPECS["fun_3"]["num_var"]],
              "Cs": [10, 5, 1, 0.75, 0.5, 0.25, 0.1, 0.01]},
    "fun_4": {"initial_vectors": [[50] * PROBLEM_SPECS["fun_4"]["num_var"], [25] * PROBLEM_SPECS["fun_4"]["num_var"],
                                  [20] * PROBLEM_SPECS["fun_4"]["num_var"], [10] * PROBLEM_SPECS["fun_4"]["num_var"],
                                  [5] * PROBLEM_SPECS["fun_4"]["num_var"], [0] * PROBLEM_SPECS["fun_4"]["num_var"]],
              "Cs": [10, 5, 1, 0.75, 0.5, 0.25, 0.1, 0.01]}
}


def get_problem_spec(problem_name: str) -> Dict[str, Union[int, List[List[int]], List[int], float]]:
    if problem_name not in PROBLEM_SPECS:
        raise ValueError(f"Unknown problem: {problem_name}")
    return PROBLEM_SPECS[problem_name]


def get_grad_spec(problem_name: str) -> Dict[str, Union[List[int], int, float]]:
    if problem_name not in GRADIENT_SPECS:
        raise ValueError(f"Unknown problem: {problem_name}")
    return GRADIENT_SPECS[problem_name]


def get_eval_spec(problem_name: str) -> Dict[str, Union[List[int], int, float]]:
    if problem_name not in EVAL_SPECS:
        raise ValueError(f"Unknown problem: {problem_name}")
    return EVAL_SPECS[problem_name]
