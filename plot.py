import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

visualization_spec = {
    "mps": {"color": "#36FF33", "marker": "s", "linestyle": "dotted", "label": "MPS", "markersize": 12},
    "pm_lb": {"color": "#B6C800", "marker": "^", "linestyle": "dashed", "label": r"$PM_{C\searrow}$", "markersize": 12},
    "pm_ub": {"color": "#C133FF", "marker": "v", "linestyle": ":", "label": r"$PM_{C\nearrow}$", "markersize": 12},
    "gdpa": {"color": "#E31D1D", "marker": "o", "linestyle": "dashdot", "label": "GDPA", "markersize": 10},
    "pga": {"color": "#3374FF", "marker": "*", "linestyle": "solid", "label": "PGA", "markersize": 10}}


def objective_value(num_con, T, opt_name, path, freq_s):
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(opt + ".json"))
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        opts[opt] = data
    # objective function
    #plt.figure(figsize=(10, 5))
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
        if opt in ["mps", "pm_lb", "pm_ub"]:
            plt.axhline(
                y=opts[opt]["J"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec[opt]["color"],
                linestyle=visualization_spec[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.ylim([lower_bound, upper_bound])
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Objective value", fontsize=14)
    plt.grid()
    plt.savefig("objective_value.svg", format='svg')
    plt.show()


def constraint_violation(num_con, T, opt_name, path, freq_s):
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
        if opt in ["mps", "pm_lb", "pm_ub"]:
            plt.axhline(
                y=opts[opt]["f"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec[opt]["color"],
                linestyle=visualization_spec[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.yscale("log")
    # plt.ylim([lower_bound, upper_bound])
    plt.legend()
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Constraint violation", fontsize=14)
    plt.grid()
    plt.savefig("constraint_violations.svg", format='svg')
    plt.show()


if __name__ == "__main__":
    objective_value(2, 200, opt_name=["mps", "pm_lb", "pm_ub", "gdpa", "pga"], path=Path("data/fun_1"), freq_s=10)
    constraint_violation(2, 200, opt_name=["mps", "pm_lb", "pm_ub", "gdpa", "pga"], path=Path("data/fun_1"), freq_s=10)
