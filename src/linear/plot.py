import matplotlib.pyplot as plt
import pandas as pd

from src.linear.optimization_specs import *
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
    "pm_lb": {"color": "#B6C800", "marker": "^", "linestyle": "dashed", "label": r"$PA_{C\searrow}$", "markersize": 12},
    "pm_ub": {"color": "#f119c3", "marker": "v", "linestyle": "dotted", "label": r"$PA_{C\nearrow}$", "markersize": 12},
    "ipdd": {"color": "#0d5915", "marker": "2", "linestyle": linestyles['densely dashed'],
             "label": "IPDD",
             "markersize": 15},
    "gdpa": {"color": "#E31D1D", "marker": "o", "linestyle": "dashdot", "label": "GDPA", "markersize": 7},
    "pga": {"color": "#1c24dc", "marker": "*", "linestyle": "solid", "label": "PGA", "markersize": 10},
}

visualization_spec_init = {
    "mps": {"color": "#36FF33", "marker": "s", "linestyle": "solid", "label": r"$MPS(\mathbf{x}^0_{max})$",
            "markersize": 12},
    "pm_lb_init_25_25_25_25_25": {"color": "#B6C800", "marker": "^", "linestyle": "solid",
                       "label": r"$PA_{C\searrow}(\mathbf{x}^0_{max})$", "markersize": 10},
    "pm_lb_init_10_10_10_10_10": {"color": "#b8c348", "marker": "^", "linestyle": linestyles["loosely dotted"],
                           "label": r"$PA_{C\searrow}(\mathbf{x}^0_{median})$", "markersize": 6},
    "pm_lb_init_0_0_0_0_0": {"color": "#bec291", "marker": "^", "linestyle": linestyles["loosely dashed"],
                       "label": r"$PA_{C\searrow}(\mathbf{x}^0_{min})$", "markersize": 6},
    "pm_ub_init_25_25_25_25_25": {"color": "#f119c3", "marker": "v", "linestyle": "solid",
                       "label": r"$PA_{C\nearrow}(\mathbf{x}^0_{max})$", "markersize": 10},
    "pm_ub_init_10_10_10_10_10": {"color": "#f190dd", "marker": "v", "linestyle": linestyles["loosely dotted"],
                           "label": r"$PA_{C\nearrow}(\mathbf{x}^0_{median})$", "markersize": 6},
    "pm_ub_init_0_0_0_0_0": {"color": "#ecc0e3", "marker": "v", "linestyle": linestyles["loosely dashed"],
                       "label": r"$PA_{C\nearrow}(\mathbf{x}^0_{min})$", "markersize": 6},
    "ipdd_init_25_25_25_25_25": {"color": "#0d5915", "marker": "2", "linestyle": "solid",
                      "label": r"$IPDD(\mathbf{x}^0_{max})$",
                      "markersize": 10},
    "ipdd_init_10_10_10_10_10": {"color": "#6ec977", "marker": "2", "linestyle": linestyles["loosely dotted"],
                          "label": r"$IPDD(\mathbf{x}^0_{median})$",
                          "markersize": 6},
    "ipdd_init_0_0_0_0_0": {"color": "#96f6a0", "marker": "2", "linestyle": linestyles["loosely dashed"],
                      "label": r"$IPDD(\mathbf{x}^0_{min})$",
                      "markersize": 6},
    "gdpa_init_25_25_25_25_25": {"color": "#E31D1D", "marker": "o", "linestyle": "solid",
                      "label": r"$GDPA(\mathbf{x}^0_{max})$",
                      "markersize": 10},
    "gdpa_init_10_10_10_10_10": {"color": "#df6f67", "marker": "o", "linestyle": linestyles["loosely dotted"],
                          "label": r"$GDPA(\mathbf{x}^0_{median})$", "markersize": 6},
    "gdpa_init_0_0_0_0_0": {"color": "#dea39f", "marker": "o", "linestyle": linestyles["loosely dashed"],
                      "label": r"$GDPA(\mathbf{x}^0_{min})$", "markersize": 6},
    "pga_init_25_25_25_25_25": {"color": "#1c24dc", "marker": "*", "linestyle": "solid",
                     "label": r"$PGA(\mathbf{x}^0_{max})$",
                     "markersize": 10},
    "pga_init_10_10_10_10_10": {"color": "#595fdc", "marker": "*", "linestyle": linestyles["loosely dotted"],
                         "label": r"$PGA(\mathbf{x}^0_{median})$", "markersize": 6},
    "pga_init_0_0_0_0_0": {"color": "#9598dc", "marker": "*", "linestyle": linestyles["loosely dashed"],
                     "label": r"$PGA(\mathbf{x}^0_{min})$", "markersize": 6}}


def objective_value(num_con, T, opt_name, path, path_w, function_name, freq_s, lb_obj, ub_obj):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(opt + ".json"))
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem)) + [-1]
            data = data.iloc[indices]
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
        if opt in ["mps", "pm_lb", "pm_ub"]:
            plt.axhline(
                y=opts[opt]["J"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec[opt]["color"],
                linestyle=visualization_spec[opt]["linestyle"], )
    plt.xlim([0, T])
    plt.ylim([lb_obj, ub_obj])
    # plt.yscale("log")
    plt.legend(loc='upper right')
    plt.xlabel("Computational time [s]", fontsize=15)
    plt.ylabel("Objective value", fontsize=15)
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_" + function_name + ".svg"), format='svg')
    plt.show()


def constraint_violation(num_con, T, q, opt_name, path, path_w, function_name, freq_s):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(f"{opt}.json"))[["f", "runtime"]]
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem)) + [-1]
            data = data.iloc[indices]
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
    plt.legend(loc='upper right')
    plt.xlabel("Computational time [s]", fontsize=15)
    plt.ylabel("Constraint violation", fontsize=15)
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violations_" + function_name + ".svg"), format='svg')
    plt.show()


def form_optimizers_init_names(opt_names, opt_names_init, initializations):
    for opt_name in opt_names[1:]:
        for initial_vector in initializations:
            suffix = '_'.join(map(str, initial_vector))
            opt_names_init.append(opt_name + "_init_" + suffix)
    print(opt_names_init)
    return opt_names_init


def get_element_end_iteration(opt_name, path, initial_solutions):
    """
    Print the objective function value and maximum constraint violation in the last iteration.
    :param opt_name: str
    :param path: Path
    :param initial_solutions: list
    :return: None
    """
    data = pd.read_json(path.joinpath(opt_name + ".json"))
    print("J ", data["J"].iloc[-1])
    print("f ", data["f"].iloc[-1])


def objective_value_initialization(num_con, T, opt_name, path, path_w, function_name, freq_s, lb_obj, ub_obj):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(opt + ".json"))
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem)) + [-1]
            data = data.iloc[indices]
        opts[opt] = data
    # objective function
    # plt.figure(figsize=(10, 5))
    for opt in opt_name:
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
    plt.ylim([lb_obj, ub_obj])
    # plt.yscale("log")
    plt.legend(loc="upper right")
    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Objective value", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_init_" + function_name + ".svg"), format='svg')
    plt.show()


def constraint_violation_initialization(num_con, T, opt_name, path, path_w, function_name, freq_s):
    plt.figure(figsize=(8, 6))
    opts = {}
    for opt in opt_name:
        data = pd.read_json(path.joinpath(f"{opt}.json"))[["f", "runtime"]]
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem)) + [-1]
            data = data.iloc[indices]
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
            color=visualization_spec_init[opt]["color"],
            marker=visualization_spec_init[opt]["marker"],
            linestyle=visualization_spec_init[opt]["linestyle"],
            label=visualization_spec_init[opt]["label"],
            markersize=visualization_spec_init[opt]["markersize"],
        )
        if any(optimizer in opt for optimizer in ["mps", "pm_lb", "pm_ub"]):
            plt.axhline(
                y=opts[opt]["f"][0],
                xmin=opts[opt]["runtime"][0] / (T),
                xmax=1,
                color=visualization_spec_init[opt]["color"],
                linestyle=visualization_spec_init[opt]["linestyle"], )
    plt.xlim([0, T])
    # plt.yscale("log")
    # plt.ylim([lower_bound, upper_bound])
    plt.legend(loc='upper right')
    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Constraint violation", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violations_init_" + function_name + ".svg"), format='svg')
    plt.show()


def parameter_C(opt_names, T, Cs, path_r, path_w, function_name, lb_obj, ub_obj):
    # Define colors and styles
    opts = ["#36FF33", "#B6C800", "#f119c3", "#0d5915", "#E31D1D", "#1c24dc", "#36FF33"]
    if len(opts) < len(Cs):
        raise ValueError("The number of parameters C must be at least equal to the number of C values.")
    colors = {}
    for i, C in enumerate(Cs):
        colors[str(C)] = opts[i]
    styles = {
        'pm_lb': {'linestyle': '--', 'marker': 'o', "markersize": 7},
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
            if opt_name == "pm_lb":
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

    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Constraint violation", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    # plt.ylim([lb_obj, ub_obj])
    plt.xlim([0, T])
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_parameter_C_" + function_name + ".svg"), format='svg')
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
            for i in range(len(f)):
                constraint_violations.append(abs(min(0, min(f[i]))))

            plt.plot(data["runtime"], constraint_violations,
                     color=colors[str(C)],
                     **styles[opt_name])
            if opt_name == "pm_lb":
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

    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Constraint violation", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    plt.xlim([0, T])
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violations_parameter_C_" + function_name + ".svg"), format='svg')
    plt.show()


def evaluate_gdpa(T, function, betas, freq_s, path_r, path_w, lb_obj, ub_obj):
    styles = {
        0.9: {"color": "#E31D1D", "marker": "o", "linestyle": "solid", "label": r"$\beta=0.9$", "markersize": 7},
        0.75: {"color": "#36FF33", "marker": "^", "linestyle": linestyles["densely dashdotdotted"],
               "label": r"$\beta=0.75$", "markersize": 7},
        0.5: {"color": "#1c24dc", "marker": "s", "linestyle": "dashdot", "label": r"$\beta=0.5$", "markersize": 7},
        0.25: {"color": "#0d5915", "marker": "*", "linestyle": linestyles['densely dashed'],
               "label": r"$\beta=0.25$",
               "markersize": 10},
        0.1: {"color": "#f119c3", "marker": "2", "linestyle": "dotted", "label": r"$\beta=0.1$", "markersize": 15}}
    plt.figure(figsize=(8, 6))
    for beta in betas:
        filename = f"gdpa_beta_{beta}.json"
        filepath = path_r.joinpath(filename)
        data = pd.read_json(filepath)
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem))
            if indices[-1] < len(data) * 0.97:
                indices += [-1]
            data = data.iloc[indices]
        plt.plot(data["runtime"], data["J"],
                 color=styles[beta]["color"],
                 marker=styles[beta]["marker"],
                 linestyle=styles[beta]["linestyle"],
                 label=styles[beta]["label"],
                 markersize=styles[beta]["markersize"],
                 )
    plt.xlim([0, T])
    plt.ylim([lb_obj, ub_obj])
    # plt.yscale("log")
    plt.legend(loc='upper right')
    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Objective value", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("objective_value_" + function + "_gdpa.svg"), format='svg')
    plt.show()

    plt.figure(figsize=(8, 6))
    for beta in betas:
        constraints = []
        filename = f"gdpa_beta_{beta}.json"
        filepath = path_r.joinpath(filename)
        data = pd.read_json(filepath)
        freq_elem = int(len(data) / (T / freq_s))
        if freq_elem > 0:
            indices = list(range(0, len(data), freq_elem))
            if indices[-1] < len(data) * 0.97:
                indices += [-1]
            data = data.iloc[indices]
        data_constraint = list(data["f"])
        for i in range(len(data_constraint)):
            constraints.append(abs(min(0, min(data_constraint[i]))))
        plt.plot(data["runtime"], constraints,
                 color=styles[beta]["color"],
                 marker=styles[beta]["marker"],
                 linestyle=styles[beta]["linestyle"],
                 label=styles[beta]["label"],
                 markersize=styles[beta]["markersize"],
                 )
    plt.xlim([0, T])
    # plt.yscale("log")
    plt.legend(loc='upper right')
    plt.xlabel("Computational time [s]", fontsize=18)
    plt.ylabel("Constraint violation", fontsize=18)
    plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(path_w.joinpath("constraint_violation_" + function + "_gdpa.svg"), format='svg')
    plt.show()


if __name__ == "__main__":
    function = "fun_3"
    lb_obj = PLOT_SPECS[function]["lb_obj"]
    ub_obj = PLOT_SPECS[function]["ub_obj"]
    freq_s = 20
    path_read: Path = Path("../../data").joinpath(function)
    path_write: Path = Path("../../plots").joinpath(function)
    problem_spec = get_problem_spec(function)
    grad_spec = get_grad_spec(function)
    eval_spec = get_eval_spec(function)
    opt_name = ["mps", "pm_lb", "pm_ub", "ipdd", "gdpa", "pga"]
    # get_element_end_iteration(opt_name="pga", path=path_read, initial_solutions="")
    """
    objective_value(num_con=problem_spec["num_con"], T=problem_spec["T"], opt_name=opt_name, path=path_read,
                    path_w=path_write, function_name=function,
                    freq_s=freq_s, lb_obj=lb_obj, ub_obj=ub_obj)
    constraint_violation(num_con=problem_spec["num_con"], T=problem_spec["T"], q=problem_spec["q"], opt_name=opt_name,
                         path=path_read,
                         path_w=path_write, function_name=function,
                         freq_s=freq_s)
    """
    opt_names_init = form_optimizers_init_names(opt_names=opt_name,
                                                opt_names_init=[],
                                                initializations=[[25, 25, 25, 25, 25],
                                                                 [10, 10, 10, 10, 10], [0, 0, 0, 0, 0]])
    objective_value_initialization(num_con=problem_spec["num_con"], T=problem_spec["T"], opt_name=opt_names_init,
                                   path=path_read.joinpath("initialization"), path_w=path_write, function_name=function,
                                   freq_s=freq_s, lb_obj=lb_obj, ub_obj=ub_obj)
    constraint_violation_initialization(num_con=problem_spec["num_con"], T=problem_spec["T"], opt_name=opt_names_init,
                                        path=path_read.joinpath("initialization"), path_w=path_write,
                                        function_name=function,
                                        freq_s=freq_s)

    """
    #get_element_end_iteration(opt_name="pm_ub", path=path_read.joinpath("initialization"),
    #                          initial_solutions=eval_spec["initial_vectors"])
    parameter_C(opt_names=["pm_lb", "pga"], T=problem_spec["T"], Cs=[1, 0.75, 0.5, 0.25, 0.1, 0.05],
                path_r=path_read.joinpath("parameter_C"), path_w=path_write, function_name=function, lb_obj=lb_obj,
                ub_obj=ub_obj)
    evaluate_gdpa(T=problem_spec["T"], function=function, betas=[0.9, 0.75, 0.5, 0.25, 0.1], freq_s=freq_s,
                  path_r=Path("data").joinpath(function).joinpath("evaluate_gdpa"),
                  path_w=Path("plots").joinpath(function), lb_obj=lb_obj, ub_obj=ub_obj)
    """
