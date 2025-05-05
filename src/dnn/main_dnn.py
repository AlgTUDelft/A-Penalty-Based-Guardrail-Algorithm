from pathlib import Path
from datetime import datetime
import statistics
import pandas as pd

from grid_config import PipePreset
from param import StatePar, OutputPar
from prediction import (
    multi_step_prediction,
    save_loss,
)
from state import StateDNN
from output import OutputDNN
from nn import NN

if __name__ == "__main__":
    now = datetime.now()
    date_time_str = now.strftime("%m-%d")
    predictions_p: Path = Path(__file__).parents[1] / "data/dnn/predictions"
    models_p: Path = Path(__file__).parents[1] / "data/dnn/models"
    num_run = 1 # number of runs
    layer_sizes = [[1, 1]]
    time_delay: int = PipePreset["time_delay"]  # number of previous actions
    time_delay_q: int = PipePreset["time_delay_q"]  # number of previous heat demands
    # whether to normalize the data
    normalize: bool = True
    # whether we early stop
    early_stop: bool = True
    # whether we are pre-training (warming up) the model or training the model
    warm_ups: bool = [True, False]
    # whether to write the data during predictions in .csv file for further analysis
    write_prediction_data: bool = False
    # whether to save the model and predictions
    save_up: bool = True
    for layer_size_ in layer_sizes:
        # dataframes for errors
        err_one_step = {
            "Supply_inlet_temp": [],
            "Supply_outlet_temp": [],
            "Mass_flow": [],
            "Delivered_heat": [],
        }
        err_multi_step = {
            "Supply_inlet_temp": [],
            "Supply_outlet_temp": [],
            "Mass_flow": [],
            "Delivered_heat": [],
        }
        # save training and validation loss both with and without early stopping
        train_loss = {"state": [], "output": []}
        val_loss = {"state": [], "output": []}
        # state parameters
        layer_size = layer_size_.copy()
        layer_size.append(3)
        s_par = StatePar(
            time_delay=time_delay,
            time_delay_q=time_delay_q,
            columns=[
                "Supply in temp ",
                "Supply out temp ",
                "Supply mass flow ",
                "Heat demand ",
                "Supply plugs ",
                "Ret plugs ",
            ],
            output_num=3,
            layer_size=layer_size,
            outputs=[
                "Supply inlet temperature ",
                "Supply outlet temperature ",
                "Mass flow ",
            ],
            x_p="x_s.csv",
            y_p="y_s.csv",
            electricity_price_p="electricity_price.csv",
            supply_pipe_plugs_p="supply_pipe_plugs.pickle",
            return_pipe_plugs_p="return_pipe_plugs.pickle",
        )
        # output parameters
        layer_size = layer_size_.copy()
        layer_size.append(1)
        y_par = OutputPar(
            time_delay=time_delay,
            columns=[
                "Supply in temp ",
                "Supply out temp ",
                "Supply mass flow ",
                "Delivered heat ",
            ],
            output_num=1,
            layer_size=layer_size,
            outputs=["Delivered heat "],
            x_p="x_y.csv",
            y_p="y_y.csv",
            electricity_price_p="",
            supply_pipe_plugs_p="",
            return_pipe_plugs_p="",
        )
        # state model
        for i in range(num_run):
            for warm_up in warm_ups:
                # extension to the saved model depending on whether we are warming up
                if warm_up:
                    warm_up_ext = "_warm_up"
                else:
                    warm_up_ext = ""
                state = StateDNN(
                    result_p=s_par.result_p, #global result path
                    model_p=s_par.model_p, #model path
                    time_delay=s_par.time_delay,
                    time_delay_q=s_par.time_delay_q,
                    columns=s_par.columns,
                    normalize=normalize,
                    warm_up=warm_up,
                    warm_up_ext=warm_up_ext,
                )
                state.get_min_max()
                scaler_x_s, scaler_y_s, scaler_s_s = state.dataset(
                    x_p=s_par.x_p,
                    y_p=s_par.y_p,
                    electricity_price_p=s_par.electricity_price_p,
                    plugs_supply_p=s_par.supply_pipe_plugs_p,
                    plugs_return_p=s_par.return_pipe_plugs_p,
                )
                x_train_s, x_test_s, y_train_s, y_test_s = state.train_test_split(
                    x_p=s_par.x_p, y_p=s_par.y_p
                )
                model_s, train_loss_state_, val_loss_state_ = state.train_nn(
                    x_train=x_train_s,
                    feature_num=s_par.feature_num,
                    y_train=y_train_s,
                    layer_size=s_par.layer_size,
                    batch_size=32,
                    num_run=i,
                    save_up=save_up,
                    early_stop=early_stop,
                )
                err_s = state.prediction(
                    model=model_s,
                    x_test=x_test_s,
                    y_test=y_test_s,
                    outputs=s_par.outputs,
                    scaler_y=scaler_y_s,
                    time_delay=s_par.time_delay,
                )
                output = OutputDNN(
                    result_p=y_par.result_p,
                    model_p=y_par.model_p,
                    time_delay=y_par.time_delay,
                    columns=y_par.columns,
                    normalize=normalize,
                    warm_up=warm_up,
                    warm_up_ext=warm_up_ext,
                )
                output.get_min_max()
                scaler_x_y, scaler_y_y = output.dataset(
                    x_p=y_par.x_p,
                    y_p=y_par.y_p,
                    electricity_price_p="",
                    plugs_supply_p=y_par.supply_pipe_plugs_p,
                    plugs_return_p=y_par.return_pipe_plugs_p,
                )
                x_train_y, x_test_y, y_train_y, y_test_y = output.train_test_split(
                    x_p=y_par.x_p, y_p=y_par.y_p
                )
                model_y, train_loss_out_, val_loss_out_ = output.train_nn(
                    x_train=x_train_y,
                    feature_num=y_par.feature_num,
                    y_train=y_train_y,
                    layer_size=y_par.layer_size,
                    batch_size=32,
                    num_run=i,
                    save_up=save_up,
                    early_stop=early_stop,
                )
                err_y = output.prediction(
                    model=model_y,
                    x_test=x_test_y,
                    y_test=y_test_y,
                    outputs=y_par.outputs,
                    scaler_y=scaler_y_y,
                    time_delay=y_par.time_delay,
                )
                if not warm_up:
                    for output in s_par.outputs:
                        s_par.err[output].append(err_s[output])
                    for output in y_par.outputs:
                        y_par.err[output].append(err_y[output])
                    train_loss["state"].append(train_loss_state_)
                    val_loss["state"].append(val_loss_state_)
                    train_loss["output"].append(train_loss_out_)
                    val_loss["output"].append(val_loss_out_)
                    # NN.save_datasets(
                    #    x_s=x_test_s,
                    #    y_s=y_test_s,
                    #    x_y=x_test_y,
                    #    y_y=y_test_y,
                    #    time_delay=s_par.time_delay,
                    #    s_ext=s_ext["delta_s"],
                    #    result_p = result_p
                    # )
                    err_one_step_ = multi_step_prediction(
                        result_p=Path(__file__).parents[1] / "data/dnn",
                        model_s=model_s,
                        model_y=model_y,
                        time_delay=s_par.time_delay,
                        time_delay_q=s_par.time_delay_q,
                        T=1,
                        scaler_s_s=scaler_s_s,
                        scaler_x_s=scaler_x_s,
                        scaler_y_s=scaler_y_s,
                        scaler_y_y=scaler_y_y,
                        write_prediction_data=write_prediction_data,
                    )
                    err_multi_step_ = multi_step_prediction(
                        result_p=Path(__file__).parents[1] / "data/dnn",
                        model_s=model_s,
                        model_y=model_y,
                        time_delay=s_par.time_delay,
                        time_delay_q=s_par.time_delay_q,
                        T=5,
                        scaler_s_s=scaler_s_s,
                        scaler_x_s=scaler_x_s,
                        scaler_y_s=scaler_y_s,
                        scaler_y_y=scaler_y_y,
                        write_prediction_data=write_prediction_data,
                    )
                    for output in list(err_one_step.keys()):
                        err_one_step[output].append(err_one_step_[output])
                        err_multi_step[output].append(err_multi_step_[output])
        err_one_step = pd.DataFrame(err_one_step)
        err_multi_step = pd.DataFrame(err_multi_step)
        if not warm_up and save_up:
            save_loss(
                result_p=predictions_p,
                loss=train_loss,
                type_loss="train_loss_",
                layer_size=layer_size,
                early_stop=early_stop,
            )
            save_loss(
                result_p=predictions_p,
                loss=val_loss,
                type_loss="val_loss_",
                layer_size=layer_size,
                early_stop=early_stop,
            )
            err_one_step.to_csv(
                predictions_p.joinpath(
                    "prediction_"
                    + "one_step_"
                    + NN.neurons_ext(s_par.layer_size)
                    + "_"+date_time_str
                    + ".csv"
                )
            )
            err_multi_step.to_csv(
               predictions_p.joinpath(
                    "prediction_"
                    + "multi_step_"
                    + NN.neurons_ext(s_par.layer_size)
                    + "_"+date_time_str
                    + ".csv"
                )
            )
        """
        for i, output in enumerate(s_par.outputs):
            print(output + " error {:.6f}".format(statistics.mean(s_par.err[output])))
        for i, output in enumerate(y_par.outputs):
            delivered_heat_real_world_feature = pd.DataFrame(y_par.err[output])
            if save_up:
                delivered_heat_real_world_feature.to_csv(
                    result_p.joinpath(result_p_sub).joinpath(
                        type_s
                        + "_prediction_real_world_feature"
                        + experiments_type["model_ext"]
                        + "_L_"
                        + str(PipePreset1["Length"])
                        + s_ext["delta_s"]
                        + "err_one_step"
                        + "_time_delay_"
                        + str(s_par.time_delay)
                        + "_"
                        + NN.neurons_ext(s_par.layer_size)
                        + s_ext["early_stop"]
                        + ".csv"
                    )
                )
            print(output + " error {:.6f}".format(statistics.mean(y_par.err[output])))
    """
