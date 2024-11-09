import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from collections import OrderedDict


def calculate_max_percent_f_and_q(f, q):
    """
    Calculate the largest constraint violation in percentage per algorithm's iteration
    :param f: list[lists]]
    :param q: list
    :return: list
    """
    # Check for zeros in 'q'
    if 0 in q:
        raise ValueError("Division by zero encountered in 'q' array.")

    # Check that all sublists in 'f' have the same length as 'q'
    for sublist in f:
        if len(sublist) != len(q):
            raise ValueError("Sublists in 'f' must be the same length as 'q'.")

    # Perform the operation
    result = [
        [(abs(elem) / q[j]) * 100 for j, elem in enumerate(sublist)]
        for sublist in f
    ]

    result_max = [max(elem) for elem in result]

    return result_max


linestyles = OrderedDict(
    [('solid', (0, ())),
     ('loosely dotted', (0, (1, 10))),
     ('dotted', (0, (1, 5))),
     ('densely dotted', (0, (1, 1))),

     ('loosely dashed', (0, (5, 10))),
     ('dashed', (0, (5, 5))),
     ('densely dashed', (0, (5, 1))),

     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('dashdotted', (0, (3, 5, 1, 5))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

visualization_spec = {
    "mps": {"color": "#36FF33", "marker": "s", "linestyle": "dotted", "label": "MPS", "markersize": 12},
    "pm_lb": {"color": "#B6C800", "marker": "^", "linestyle": "dashed", "label": "$PM_{C\searrow}$", "markersize": 12},
    "pm_ub": {"color": "#f119c3", "marker": "v", "linestyle": "dotted", "label": r"$PM_{C\nearrow}$", "markersize": 12},
    "ipdd": {"color": "#0d5915", "marker": "2", "linestyle": linestyles['densely dashed'],
             "label": "IPDD",
             "markersize": 15},
    "gdpa": {"color": "#E31D1D", "marker": "o", "linestyle": "dashdot", "label": "GDPA", "markersize": 7},
    "pga": {"color": "#1c24dc", "marker": "*", "linestyle": "solid", "label": "PGA", "markersize": 10}}

visualization_spec_init = {
    "mps": {"color": "#36FF33", "marker": "s", "linestyle": "solid", "label": r"$MPS(\mathbf{x}^0_{max})$",
            "markersize": 12},
    "pm_lb_init_25_25_25_25_25": {"color": "#B6C800", "marker": "^", "linestyle": "solid",
                         "label": r"$PM_{C\searrow}(\mathbf{x}^0_{max})$", "markersize": 12},
    "pm_ub_init_25_25_25_25_25": {"color": "#f119c3", "marker": "v", "linestyle": "solid",
                         "label": r"$PM_{C\nearrow}(\mathbf{x}^0_{max})$", "markersize": 12},
    "ipdd_init_25_25_25_25_25": {"color": "#0d5915", "marker": "2", "linestyle": "solid",
                        "label": r"$IPDD(\mathbf{x}^0_{max})$",
                        "markersize": 15},
    "gdpa_init_25_25_25_25_25": {"color": "#E31D1D", "marker": "o", "linestyle": "solid",
                        "label": r"$GDPA(\mathbf{x}^0_{max})$",
                        "markersize": 10},
    "gdpa_init_20_20_20_20_20": {"color": "#df6f67", "marker": "o", "linestyle": linestyles["loosely dotted"],
                        "label": r"$GDPA(\mathbf{x}^0_1)$", "markersize": 6},
    "gdpa_init_10_10_10_10_10": {"color": "#dea39f", "marker": "o", "linestyle": linestyles["loosely dashed"],
                        "label": r"$GDPA(\mathbf{x}^0_2)$", "markersize": 6},
    "pga_init_25_25_25_25_25": {"color": "#1c24dc", "marker": "*", "linestyle": "solid",
                       "label": r"$PGA(\mathbf{x}^0_{max})$",
                       "markersize": 10},
    "pga_init_20_20_20_20_20": {"color": "#595fdc", "marker": "*", "linestyle": linestyles["loosely dotted"],
                       "label": r"$PGA(\mathbf{x}^0_1)$", "markersize": 6},
    "pga_init_10_10_10_10_10": {"color": "#9598dc", "marker": "*", "linestyle": linestyles["loosely dashed"],
                       "label": r"$PGA(\mathbf{x}^0_2)$", "markersize": 6}}

def objective_value_dnn(num_con, T, opt_name, path, path_w, function_name, freq_s):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(opt + ".json"))
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        opts[opt] = data
    # objective function
    # plt.figure(figsize=(10, 5))
    for opt in opt_name:
        plt.plot(
            opts[opt]["runtime"],
            opts[opt]["J"],
            color=visualization_spec[opt]["color"],
            marker=visualization_spec[opt]["marker"],
            linestyle=visualization_spec[opt]["linestyle"],
            label=visualization_spec[opt]["label"],
            markersize=visualization_spec[opt]["markersize"],
        )
        if opt in ["pm_lb", "pm_ub"]:
            plt.axhline(
                y=opts[opt]["J"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec[opt]["color"],
                linestyle=visualization_spec[opt]["linestyle"], )
    plt.xlim([0, T])
    plt.ylim([9000, 10100])
    # plt.ylim([lower_bound, upper_bound])
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel(r"Objective value [€]", fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_" + function_name + ".svg"), format='svg')
    plt.show()


def constraint_violation_dnn(num_con, T, q, opt_name, path, path_w, function_name, freq_s):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(f"{opt}.json"))[["f", "runtime"]]
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        opts[opt] = {"runtime": data["runtime"].tolist(), "f": []}
        f = data["f"].tolist()
        for i in range(len(f)):
            opts[opt]["f"].append(abs(min(0, min(f[i]))))
        #opts[opt]["f"] = calculate_max_percent_f_and_q(f, q)
    # objective function
    # plt.figure(figsize=(10, 5))
    for opt in opt_name:
        plt.plot(
            opts[opt]["runtime"],
            opts[opt]["f"],
            color=visualization_spec[opt]["color"],
            marker=visualization_spec[opt]["marker"],
            linestyle=visualization_spec[opt]["linestyle"],
            label=visualization_spec[opt]["label"],
            markersize=visualization_spec[opt]["markersize"],
        )
        if opt in ["pm_lb", "pm_ub"]:
            plt.axhline(
                y=opts[opt]["f"],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec[opt]["color"],
                linestyle=visualization_spec[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.yscale("log")
    # plt.ylim([lower_bound, upper_bound])
    plt.legend()
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Constraint violation [MW]", fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violations_" + function_name + ".svg"), format='svg')
    plt.show()

if __name__ == "__main__":
        objective_value_dnn(num_con=1, T=500, opt_name=["mps", "pm_lb", "pm_ub"], path=Path("data").joinpath("dnn"),
                    path_w=Path("plots").joinpath("dnn"), function_name="dnn",
                    freq_s=10)
        
        constraint_violation_dnn(num_con=1, T=500, q= [0], opt_name=["mps", "pm_lb", "pm_ub"], path=Path("data").joinpath("dnn"),
                    path_w=Path("plots").joinpath("dnn"), function_name="dnn",
                    freq_s=10)
