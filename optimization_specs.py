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

}

def get_problem_spec(problem_name: str) -> Dict[str, Union[int, List[List[int]], List[int], float]]:
    if problem_name not in PROBLEM_SPECS:
        raise ValueError(f"Unknown problem: {problem_name}")
    return PROBLEM_SPECS[problem_name]

def get_grad_spec(problem_name: str) -> Dict[str, Union[List[int], int, float]]:
    if problem_name not in GRADIENT_SPECS:
        raise ValueError(f"Unknown problem: {problem_name}")
    return GRADIENT_SPECS[problem_name]