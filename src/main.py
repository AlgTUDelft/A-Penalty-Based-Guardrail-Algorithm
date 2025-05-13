from linear.algorithms_run import run as linear
from nonlinear.algorithms_run import run as nonlinear
from dnn.algorithms_run import run as dnn

from linear.plot import plot as plot_linear
from nonlinear.plot_nonlinear import plot as plot_nonlinear
from dnn.plot_dnn import plot as plot_dnn

if __name__ == "__main__":
    # running algorithms
    linear(functions=["fun_1", "fun_2", "fun_3"])
    nonlinear(functions=["fun_4", "fun_5", "fun_6"])
    dnn()

    # plotting results
    plot_linear(function="fun_1")
    plot_nonlinear(function="fun_8")
    plot_dnn()
