from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import pickle
import warnings
from src.dnn.tensor_constraint import ParNonNeg

from src.dnn.grid_config import GridProperties, ProducerPreset, PipePreset

class NN(ABC):
    def __init__(self, model_p, normalize, warm_up, warm_up_ext):
        self.model_p: Path = model_p
        # defines the path for storing results
        self.result_p: Path = Path(__file__).parents[1] / "data/dnn/predictions"
        # defines whether model will be trained with normalized data or not
        self.nor: bool = normalize
        # defines whether the data used for training neural network are warming up data or regular data
        self.warm_up: bool = warm_up
        # defines extension for warming up training
        self.warm_up_ext: str = warm_up_ext

    @abstractmethod
    def name_columns(self):
        """
        Names of columns of dataset for training neural network.
        """
        pass

    @abstractmethod
    def dataset(self, x_p, y_p, electricity_price_p, plugs_p):
        """
        Create dataset for training neural network depending
        on the model we are trying to approximate.
        """
        pass

    @abstractmethod
    def neural_net(self, layer_size, feature_num):
        """
        Creating neural network.
        param: number of neurons per layer, number of features
        return: created model
        """
        pass

    @abstractmethod
    def train_nn(self, x_train, feature_num, y_train, layer_size, batch_size, epochs):
        """
        Training of the neural network.
        """
        pass

    def get_columns(self, columns_):
        """
        Reading columns from the real-world dataset depending on the number of consumers.
        """
        columns = []
        for i in range(int(GridProperties["PipeNum"] / 2)):
            for column in columns_:
                columns.append(column + str(i + 1))
        columns.extend(["Produced heat", "Electricity price"])
        return columns

    def read_data(self, result_p):
        """
        Reading data from the main file, corresponding to previously formed columns.
        """
        with open(
            result_p.joinpath(
                "data"
                + self.warm_up_ext
                + "_num_{}_heat_demand_real_world_for_L={}_time_interval_3600_max_Q_70MW_deltaT_{}C.csv".format(
                    GridProperties["ConsumerNum"],
                    PipePreset["Length"],
                    ProducerPreset["MaxRampRateTemp"],
                )
            ),
            "rb",
        ) as f:
            data = pd.read_csv(f)[self.columns]
        return data

    def data_normalization(self, x, y, columns_x, columns_y):
        """
        Normalize input and output data between 0 and 1
        """
        scaler_x = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)
        x_scaled = pd.DataFrame(x_scaled, columns=columns_x)
        y_scaled = pd.DataFrame(y_scaled, columns=columns_y)
        return scaler_x, scaler_y, x_scaled, y_scaled

    def train_test_split(self, x_p, y_p):
        """
        Split the dataset on training and testing dataset.
        """
        with open(x_p) as file:
            x = np.array(pd.read_csv(file))
        with open(y_p) as file:
            y = np.array(pd.read_csv(file))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True
        )
        return x_train, x_test, y_train, y_test

    def prediction(self, model, x_test, y_test, outputs, scaler_y, time_delay):
        """
        One-step predictions.
        """
        y_pred = model.predict(x_test)
        if self.nor:
            y_test = scaler_y.inverse_transform(y_test)
            y_pred = scaler_y.inverse_transform(y_pred)
        err = {}
        for i, output in enumerate(outputs):
            err[output] = mean_squared_error(y_pred[:, i], y_test[:, i], squared=False)
        return err

    def get_min_max(self, dict_p):
        """
        Create dictionary with maximum and minimum values of each variable, in order to perform normalization in the optimization.
        """
        dict = {}
        for column in self.columns:
            if column != "Supply plugs 1" and column != "Electricity price":
                dict[column + " max"] = max(self.data[column])
                dict[column + " min"] = min(self.data[column])
        with open(dict_p, "wb") as f:
            pickle.dump(dict, f)

    def load_dnn_model(self, name, layer_size):
        """
        Load pretrained deep neural network model. Add customized objects depending on whether it is PLNN or ICNN model.
        """
        model = load_model(
                self.model_p.joinpath(
                    name
                    + "_"
                    + NN.neurons_ext(layer_size)
                    + ".h5"
                ),
                custom_objects={"ParNonNeg": ParNonNeg},)
        return model

    @staticmethod
    def save_datasets(x_s, y_s, x_y, y_y, time_delay, s_ext, result_p):
        """
        Save test datasets used for evaluating DNN models.
        """
        ext: str = s_ext + "time_delay_" + str(time_delay) + ".csv"
        pd.DataFrame(x_s).to_csv(result_p.joinpath("x_test_s" + ext), index=False)
        pd.DataFrame(y_s).to_csv(result_p.joinpath("y_test_s" + ext), index=False)
        pd.DataFrame(x_y).to_csv(result_p.joinpath("x_test_y" + ext), index=False)
        pd.DataFrame(y_y).to_csv(result_p.joinpath("y_test_y" + ext), index=False)

    @staticmethod
    def state_normalization(x):
        """
        Normalize the state.
        """
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        return x_scaled, scaler

    @staticmethod
    def neurons_ext(layer_size) -> str:
        """
        Form a string indicating number of neurons in each hidden layer.
        """
        neurons = "neurons"
        for i in range(len(layer_size) - 1):
            neurons += "_" + str(layer_size[i])
        return neurons

    @staticmethod
    def min_max_dict(y):
        """
        Get maximum and minimum value for each value of dictionary.
        """
        dict = {}
        for column in y.keys():
            dict[column + " max"] = max(y[column])
            dict[column + " min"] = min(y[column])
            temp = np.array(y[column])
            dict[column + " index max"] = np.argmax(temp)
            dict[column + " index min"] = np.argmin(temp)
