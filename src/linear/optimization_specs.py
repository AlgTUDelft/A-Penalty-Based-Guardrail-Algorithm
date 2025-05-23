import json
import random
import pickle
from typing import Dict, List, Union
import numpy as np
from pathlib import Path


def weighted_random():
    if random.random() < 0.2:  # 20% chance
        return 0
    else:
        return random.randint(1, 9)


def large_linear_program(num_var, num_con, path):
    if Path(path).exists():
        print("Function specification already exists")
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Function specification created")
        fun = {"num_con": num_con, "num_var": num_var, "c": [], "q": []}

        fun["ub"] = [float("inf")] * fun["num_var"]
        fun["lb"] = [0] * fun["num_var"]

        # coefficients constraints
        for i in range(fun["num_con"]):
            fun["c"].append([weighted_random() for _ in range(fun["num_var"])])
            q_mul = random.uniform(0.01, 1.0)
            fun["q"].append(int(sum(fun["c"][-1]) * q_mul))
        # coefficients objective function
        fun["c"].append([random.randint(1, 9) for _ in range(fun["num_var"])])

        random.shuffle(fun["q"])

        with open(path, 'wb') as f:
            pickle.dump(fun, f)
        # Save 'fun["q"]' into a JSON file for visual inspection
        q_json_path = path.with_name(path.stem + '_q.json')
        with open(q_json_path, 'w') as f_json:
            json.dump(fun["q"], f_json, indent=2)
        # Save 'fun["c"]' into a JSON file for visual inspection
        c_json_path = path.with_name(path.stem + '_c.json')
        with open(c_json_path, 'w') as f_json:
            json.dump(fun["c"][-1], f_json, indent=2)
        return fun


def verify_initial_vector(initial_solution, program_spec):
    for i in range(program_spec["num_con"]):
        left_side = 0
        for j in range(program_spec["num_var"]):
            left_side += program_spec["c"][i][j] * initial_solution[j]
        if left_side - program_spec["q"][i] < 0:
            print("Left side is {}".format(left_side))
            raise ValueError("Initial solution is infeasible!")
    return True


def get_initial_vector(initial_solution, problem_spec):
    try:
        # First, try the all-ones vector
        if verify_initial_vector(initial_solution=initial_solution, program_spec=problem_spec):
            print("Feasible initial solution!")
            return np.array(initial_solution)
    except ValueError:
        raise ValueError("Unsuitable initial solution!")


PROBLEM_SPECS = {"fun_1": {
    "num_con": 2,
    "num_var": 2,
    "c": [[1, 0], [1, 2], [1, 10]],
    "q": [1, 5],
    "T": 500,
    "ub": [25] * 2,
    "lb": [0] * 2
}, "fun_2": {
    "num_con": 2,
    "num_var": 2,
    "c": [[1, 2], [3, 1], [1, 1]],
    "q": [4, 5],
    "T": 500,
    "ub": [float("inf")] * 2,
    "lb": [0] * 2
}, "fun_3": {
    "num_con": 4,
    "num_var": 5,
    "c": [[3, 2, 1, 1, 0], [1, 4, 2, 0, 1], [0, 1, 1, 3, 2], [2, 0, 1, 4, 1], [2, 3, 4, 5, 6]],
    "q": [50, 35, 40, 45],
    "T": 500,
    "ub": [float("inf")] * 5,
    "lb": [0] * 5
}}

GRADIENT_SPECS = {
    "fun_1": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_1"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_1"]["num_con"]),
        "patience": 100,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475

    },
    "fun_2": {
        "initial_vector": np.array([5] * PROBLEM_SPECS["fun_2"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_2"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    },
    "fun_3": {
        "initial_vector": np.array([25] * PROBLEM_SPECS["fun_3"]["num_var"]),
        "initial_lambdas": np.array([0] * PROBLEM_SPECS["fun_3"]["num_con"]),
        "patience": 50,
        "delta": 0.000001,
        "grad_iter_max": 10000,
        "C": 0.1,
        "C_large": 1000,
        "beta": 0.25,
        "gamma": 0.99,
        "perturbation_term": 0.2475,
        "step_size": 0.2475
    }}

EVAL_SPECS = {
    "fun_1": {"initial_vectors": [[25, 25], [10, 10], [0, 0]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1,0.05]},
    "fun_2": {"initial_vectors": [[5] * PROBLEM_SPECS["fun_2"]["num_var"],
                                  [2.5] * PROBLEM_SPECS["fun_2"]["num_var"], [0] * PROBLEM_SPECS["fun_2"]["num_var"]],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1, 0.05]},
    "fun_3": {"initial_vectors": [[25] * PROBLEM_SPECS["fun_3"]["num_var"],
                                  [10] * PROBLEM_SPECS["fun_3"]["num_var"], [0] * PROBLEM_SPECS["fun_3"]["num_var"],
                                  ],
              "Cs": [1, 0.75, 0.5, 0.25, 0.1, 0.05]},
}
PLOT_SPECS = {"fun_1": {"lb_obj": -5, "ub_obj": 50}, "fun_2": {"lb_obj": 1, "ub_obj": 10},
              "fun_3": {"lb_obj": 75, "ub_obj": 150}}


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
