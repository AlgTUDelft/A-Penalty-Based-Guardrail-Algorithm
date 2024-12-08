import wandb
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from optimization_specs_nonlinear import *

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
    "pga": {"color": "#1c24dc", "marker": "*", "linestyle": "solid", "label": "PGA", "markersize": 10},
    "pm": {"color": "#B6C800", "marker": "^", "linestyle": "dashed", "label": "PM", "markersize": 12}}

visualization_spec_init = {
    "mps": {"color": "#36FF33", "marker": "s", "linestyle": "solid", "label": r"$MPS(\mathbf{x}^0_{max})$",
            "markersize": 12},
    "pm_lb_init_25_25": {"color": "#B6C800", "marker": "^", "linestyle": "solid",
                         "label": r"$PM_{C\searrow}(\mathbf{x}^0_{max})$", "markersize": 12},
    "pm_ub_init_25_25": {"color": "#f119c3", "marker": "v", "linestyle": "solid",
                         "label": r"$PM_{C\nearrow}(\mathbf{x}^0_{max})$", "markersize": 12},
    "ipdd_init_25_25": {"color": "#0d5915", "marker": "2", "linestyle": "solid",
                        "label": r"$IPDD(\mathbf{x}^0_{max})$",
                        "markersize": 15},
    "gdpa_init_25_25": {"color": "#E31D1D", "marker": "o", "linestyle": "solid",
                        "label": r"$GDPA(\mathbf{x}^0_{max})$",
                        "markersize": 10},
    "gdpa_init_0_0": {"color": "#df6f67", "marker": "o", "linestyle": linestyles["loosely dotted"],
                      "label": r"$GDPA(\mathbf{x}^0_1)$", "markersize": 6},
    "gdpa_init_-25_-25": {"color": "#dea39f", "marker": "o", "linestyle": linestyles["loosely dashed"],
                          "label": r"$GDPA(\mathbf{x}^0_2)$", "markersize": 6},
    "pga_init_25_25": {"color": "#1c24dc", "marker": "*", "linestyle": "solid",
                       "label": r"$PGA(\mathbf{x}^0_{max})$",
                       "markersize": 10},
    "pga_init_0_0": {"color": "#595fdc", "marker": "*", "linestyle": linestyles["loosely dotted"],
                     "label": r"$PGA(\mathbf{x}^0_1)$", "markersize": 6},
    "pga_init_-25_-25": {"color": "#9598dc", "marker": "*", "linestyle": linestyles["loosely dashed"],
                         "label": r"$PGA(\mathbf{x}^0_2)$", "markersize": 6}}


def form_optimizers_init_names(opt_names, initial_vectors):
    r = []
    for opt_name in opt_names:
        if opt_name == "mps":
            r.append(opt_name)
        elif opt_name in ["pm_ub", "pm_lb", "ipdd"]:
            suffix = '_'.join(map(str, initial_vectors[0]))
            r.append(opt_name + "_init_" + suffix)
        else:
            for initial_vector in initial_vectors:
                suffix = '_'.join(map(str, initial_vector))
                r.append(opt_name + "_init_" + suffix)
    return r


def objective_value(path_r, path_w, T, opt_name, function_name, freq_s):
    # setting KMP_DUPLICATE_LIB_OK=TRUE is used to prevent the error: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    # this environment variable is set only after program execution in order to avoid interfering with this execution.
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Plot using Matplotlib
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path_r.joinpath(opt + ".json"))
        data = data.dropna(subset=['J'])
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        opts[opt] = data
    plt.figure(figsize=(8, 6))
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
    # plt.yscale("log")
    plt.legend()
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Objective value", fontsize=14)
    plt.grid()

    path_w_png = path_w.joinpath("objective_value_" + function_name + ".png")
    path_w_svg = path_w.joinpath("objective_value_" + function_name + ".svg")
    # Log the plot to Weights & Biases
    plt.savefig(path_w_png)
    plt.savefig(path_w_svg)
    plt.close()

    j_values = {method: list(df["J"]) for method, df in opts.items()}
    runtime_values = {method: list(df["runtime"]) for method, df in opts.items()}

    table = wandb.Table(
        data=[
            [method, obj, runtime]
            for method, (obj, runtime) in zip(
                opts.keys(),
                zip(j_values.values(), runtime_values.values())
            )
        ],
        columns=["method", "objective", "runtime"]
    )

    wandb.log({
        "objective_value": wandb.Image(str(path_w_png)),
        "results_table_objective_value": table,
    })


def constraint_violation(path_r, path_w, T, opt_name, function_name, freq_s):
    plt.figure(figsize=(8, 6))
    constraints_der = {opt: [] for opt in opt_name}
    runtimes = {opt: [] for opt in opt_name}
    for opt in opt_name:
        data = pd.read_json(path_r.joinpath(opt + ".json"))
        data = data.dropna(subset=['J'])
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        data_constraint = list(data["f"])
        runtimes[opt] = list(data["runtime"])
        for i in range(len(data_constraint)):
            constraints_der[opt].append(abs(min(0, min(data_constraint[i]))))
    for key in opt_name:
        plt.plot(runtimes[key], constraints_der[key], label=visualization_spec[key]["label"],
                 color=visualization_spec[key]["color"], linestyle=visualization_spec[key]["linestyle"],
                 marker=visualization_spec[key]["marker"], markersize=visualization_spec[key]["markersize"])
        if key in ["mps", "pm_lb", "pm_ub"]:
            plt.axhline(
                y=constraints_der[key][0],
                xmin=runtimes[key][0] / (T),
                xmax=1,
                color=visualization_spec[key]["color"],
                linestyle=visualization_spec[key]["linestyle"], )
    plt.xlim([0, T])
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Constraint violation", fontsize=14)
    plt.legend()
    plt.grid()
    plt.xlim([0, T])
    # plt.yscale("log")
    # plt.ylim([lower_bound, upper_bound])
    path_w_png = path_w.joinpath("constraint_violations_" + function_name + ".png")
    path_w_svg = path_w.joinpath("constraint_violations_" + function_name + ".svg")
    # Log the plot to Weights & Biases
    plt.savefig(path_w_png)
    plt.savefig(path_w_svg)
    plt.close()

    table = wandb.Table(
        data=[
            [method, cons, runtime]
            for method, (cons, runtime) in zip(
                constraints_der.keys(),
                zip(constraints_der.values(), runtimes.values())
            )
        ],
        columns=["method", "constraint_violations", "runtime"]
    )

    wandb.log({
        "constraint_violations": wandb.Image(str(path_w_png)),
        "results_table_constraint_violations": table,
    })


def parameter_C(opt_names, T, Cs, path_r, path_w):
    # Define colors and styles
    opts = ["#36FF33", "#B6C800", "#f119c3", "#0d5915", "#E31D1D", "#1c24dc"]
    if len(opts) < len(Cs):
        raise ValueError("The number of parameters C must be at least equal to the number of C values.")
    colors = {}
    for i, C in enumerate(Cs):
        colors[str(C)] = opts[i]
    styles = {
        'pm': {'linestyle': '--', 'marker': 'o', "markersize": 7},
        'pga': {'linestyle': '-', 'marker': '*', "markersize": 10}
    }

    plt.figure(figsize=(12, 6))

    # First set of lines for C values
    for C in Cs:
        for opt_name in opt_names:
            filename = f"{opt_name}_{C}.json"
            filepath = path_r.joinpath(filename)

            data = pd.read_json(filepath)

            plt.plot(data["runtime"], data["J"],
                     color=colors[str(C)],
                     **styles[opt_name])
            if opt_name == "pm":
                plt.axhline(
                    y=data["J"][0],
                    xmin=data["runtime"][0] / (T),
                    xmax=1,
                    color=colors[str(C)],
                    linestyle=styles[opt_name]["linestyle"])

    # Create legend for C values
    C_legend = [plt.Line2D([0], [0], color=colors[str(C)], lw=2, label=f'C={C}') for C in Cs]

    # Create legend for optimizers
    opt_legend = [plt.Line2D([0], [0], **styles[opt], color='black', label=visualization_spec[opt]["label"]) for opt in
                  opt_names]

    # Add C values legend
    first_legend = plt.legend(handles=C_legend, title='C values', loc='upper left')
    plt.gca().add_artist(first_legend)

    # Add Optimizers legend
    plt.legend(handles=opt_legend, title='Optimizers', loc='upper right')

    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Objective value", fontsize=14)
    plt.xlim([0, T])
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_parameter_C.svg"), format='svg')
    plt.show()

    plt.figure(figsize=(12, 6))

    # First set of lines for C values
    for C in Cs:
        for opt_name in opt_names:
            filename = f"{opt_name}_{C}.json"
            filepath = path_r.joinpath(filename)

            data = pd.read_json(filepath)
            constraint_violations = []
            f = data["f"].tolist()
            print("param C f ", f)
            for i in range(len(f)):
                constraint_violations.append(abs(min(0, min(f[i]))))

            plt.plot(data["runtime"], constraint_violations,
                     color=colors[str(C)],
                     **styles[opt_name])
            if opt_name == "pm":
                plt.axhline(
                    y=constraint_violations[0],
                    xmin=data["runtime"][0] / (T),
                    xmax=1,
                    color=colors[str(C)],
                    linestyle=styles[opt_name]["linestyle"])

    # Create legend for C values
    C_legend = [plt.Line2D([0], [0], color=colors[str(C)], lw=2, label=f'C={C}') for C in Cs]

    # Create legend for optimizers
    opt_legend = [plt.Line2D([0], [0], **styles[opt], color='black', label=visualization_spec[opt]["label"]) for opt in
                  opt_names]

    # Add C values legend
    first_legend = plt.legend(handles=C_legend, title='C values', loc='upper left')
    plt.gca().add_artist(first_legend)

    # Add Optimizers legend
    plt.legend(handles=opt_legend, title='Optimizers', loc='upper right')

    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Constraint violation", fontsize=14)
    plt.xlim([0, T])
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violations_parameter_C.svg"), format='svg')
    plt.show()


def initialization(opt_names, initial_vectors, T, path_r, path_w, freq_s):
    opt_names_init = form_optimizers_init_names(opt_names=opt_names, initial_vectors=initial_vectors)
    plt.figure(figsize=(8, 6))
    opts = {}
    opts_constraints = {}
    for opt in opt_names_init:
        data = pd.read_json(path_r.joinpath(opt + ".json"))
        data = data.dropna(subset=['J'])
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            data = data.iloc[::freq_elem, :]
        opts[opt] = data
        constraint_violations = []
        f = data["f"].tolist()
        for i in range(len(f)):
            constraint_violations.append(abs(min(0, min(f[i]))))
        opts_constraints[opt] = constraint_violations
    # objective function
    # plt.figure(figsize=(10, 5))
    for opt in opt_names_init:
        plt.plot(
            opts[opt]["runtime"],
            opts[opt]["J"],
            color=visualization_spec_init[opt]["color"],
            marker=visualization_spec_init[opt]["marker"],
            linestyle=visualization_spec_init[opt]["linestyle"],
            label=visualization_spec_init[opt]["label"],
            markersize=visualization_spec_init[opt]["markersize"],
        )
        if any(optimizer in opt for optimizer in ["mps", "pm_lb", "pm_ub"]):
            plt.axhline(
                y=opts[opt]["J"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec_init[opt]["color"],
                linestyle=visualization_spec_init[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.ylim([lower_bound, upper_bound])
    plt.legend(loc="upper right")
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Objective value", fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_init.svg"), format='svg')
    plt.show()

    for opt in opt_names_init:
        plt.plot(
            opts[opt]["runtime"],
            opts_constraints[opt],
            color=visualization_spec_init[opt]["color"],
            marker=visualization_spec_init[opt]["marker"],
            linestyle=visualization_spec_init[opt]["linestyle"],
            label=visualization_spec_init[opt]["label"],
            markersize=visualization_spec_init[opt]["markersize"],
        )
        if any(optimizer in opt for optimizer in ["mps", "pm_lb", "pm_ub"]):
            plt.axhline(
                y=opts_constraints[opt][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec_init[opt]["color"],
                linestyle=visualization_spec_init[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.ylim([lower_bound, upper_bound])
    plt.legend(loc="upper right")
    plt.xlabel("Computational time [s]", fontsize=14)
    plt.ylabel("Constraint violation", fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violation_init.svg"), format='svg')
    plt.show()


if __name__ == "__main__":
    function = "fun_2"
    problem_spec = PROBLEM_SPECS[function]
    grad_spec = GRADIENT_SPECS[function]
    eval_spec = EVAL_SPECS[function]
    """
    wandb.init(
        project="nonlinear_optimization",
        config={**problem_spec, **grad_spec},
        # Optional: add a name for this run
        name=f"optimization_run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    """
    parameter_C(opt_names=["pm", "pga"], T=problem_spec["T"], Cs=eval_spec["Cs"],
                path_r=Path("data/nonlinear").joinpath(function).joinpath("parameter_C"),
                path_w=Path("plots/nonlinear").joinpath(function))
    initialization(opt_names=["mps", "pm_lb", "pm_ub", "ipdd", "gdpa", "pga"],
                   initial_vectors=[[25, 25], [0, 0], [-25, -25]], T=problem_spec["T"],
                   path_r=Path("data/nonlinear").joinpath(function).joinpath("initialization"),
                   path_w=Path("plots/nonlinear").joinpath(function), freq_s=10)
    """
    objective_value(path_r=Path("data/nonlinear").joinpath(function),
                    path_w=Path("plots/nonlinear").joinpath(function), T=problem_spec["T"],
                    opt_name=["mps", "pm_lb", "pm_ub", "ipdd", "gdpa", "pga"], function_name=function, freq_s=10)
    constraint_violation(path_r=Path("data/nonlinear").joinpath(function),
                    path_w=Path("plots/nonlinear").joinpath(function), T=problem_spec["T"],
                    opt_name=["mps", "pm_lb", "pm_ub", "ipdd", "gdpa", "pga"], function_name=function, freq_s=10)
    wandb.finish()
    """
