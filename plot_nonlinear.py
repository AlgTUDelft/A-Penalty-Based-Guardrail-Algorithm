import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from collections import OrderedDict

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


def wandb_login(save_path, runtimes, objectives, T):
    # Plot using Matplotlib
    plt.figure(figsize=(8, 6))
    for key in runtimes.keys():
        plt.plot(runtimes[key], objectives[key], label=visualization_spec[key]["label"],
                 color=visualization_spec[key]["color"], linestyle=visualization_spec[key]["linestyle"],
                 marker=visualization_spec[key]["marker"], markersize=visualization_spec[key]["markersize"])
        if key in ["mps", "pm_lb", "pm_ub"]:
            plt.axhline(
                y=objectives[key][0],
                xmin=runtimes[key][0] / (T),
                xmax=1,
                color=visualization_spec[key]["color"],
                linestyle=visualization_spec[key]["linestyle"], )
    plt.xlim([0, T])
    plt.xlabel("Runtime")
    plt.ylabel("Objective")
    plt.title("Objective Values vs Runtime")
    plt.legend()
    plt.grid()

    objective_save_path = save_path.joinpath("objectives_scatter.png")
    # Log the plot to Weights & Biases
    plt.savefig(objective_save_path)
    plt.close()

    table = wandb.Table(
        data=[
            [method, obj, runtime]
            for method, (obj, runtime) in zip(
                objectives.keys(),
                zip(objectives.values(), runtimes.values())
            )
        ],
        columns=["method", "objective", "runtime"]
    )

    wandb.log({
        "objectives_scatter": wandb.Image(str(objective_save_path)),
        "results_table": table,
    })
