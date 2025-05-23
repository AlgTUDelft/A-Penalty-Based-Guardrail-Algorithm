import pandas as pd
import pickle
import tensorflow as tf
from abc import abstractmethod 
from keras.layers import Dense, Input, Add, Activation
from keras.losses import MeanSquaredError
from keras.models import Model as ModelNN
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.constraints import NonNeg

from src.dnn.tensor_constraint import ParNonNeg
from src.dnn.param import OutputDNNParam
from src.dnn.nn import NN


class Output(NN):
    """
    Learning output function y_t = f(s_{t-1},...,s_{t-tau}, h_t,...,h_{t-tau})
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        columns,
        normalize,
        warm_up,
        warm_up_ext,
        param,
    ):
        super().__init__(model_p, normalize, warm_up, warm_up_ext)
        self.time_delay = time_delay
        self.dict_p = result_p.joinpath("output_dict.pkl")
        self.columns = self.get_columns(columns_=columns)
        self.data = self.read_data(result_p=result_p)
        self.N: int = len(self.data)
        self.param = param

    @abstractmethod
    def custom_loss(self, alpha, y_min):
        """
        Customized loss function.
        """
        pass

    def name_columns(self):
        """
        Naming columns of the neural network features dataset depending on the time delay parameter.
        """
        base_x = ["in_t_", "out_t_", "m_", "h_"]
        columns_x = []
        time_delay_plus = self.time_delay + 1
        for col in base_x:
            for i in range(1, time_delay_plus + 1):
                if (col == "in_t_" or col == "out_t_" or col == "m_") and (
                    i == time_delay_plus
                ):
                    columns_x.append(col + str(i))
        for i in range(1, time_delay_plus + 1):
            columns_x.append("h_" + str(i))
        columns_y = [
            "tilde_q" + str(time_delay_plus),
        ]
        return columns_x, columns_y

    def dataset(self, x_p, y_p, electricity_price_p, plugs_supply_p, plugs_return_p):
        """
        Create two datasets, x--features and y--output. As y is dependent on
        past features, we specify time delay parameter.
        """
        x, y = [], []
        columns_x, columns_y = self.name_columns()
        for i in range(self.time_delay + 1, self.N):
            temp = []
            temp.append(self.data["Supply in temp 1"][i])
            temp.append(self.data["Supply out temp 1"][i])
            temp.append(self.data["Supply mass flow 1"][i])
            temp.extend(self.data["Produced heat"][i - self.time_delay : i + 1])
            x.append(temp)
            y.append(
                [
                    self.data["Delivered heat 1"][i],
                ]
            )
        x = pd.DataFrame(x, columns=columns_x)
        y = pd.DataFrame(y, columns=columns_y)
        scaler_x = 0
        scaler_y = 0
        if self.nor:
            scaler_x, scaler_y, x, y = self.data_normalization(
                x, y, columns_x=columns_x, columns_y=columns_y
            )
        x.to_csv(x_p, index=False)
        y.to_csv(y_p, index=False)
        return scaler_x, scaler_y

    def train_nn(
        self,
        x_train,
        feature_num,
        y_train,
        layer_size,
        batch_size,
        num_run,
        save_up,
        early_stop,
    ):
        """
        Training of the neural network.
        return: model
        """
        if self.warm_up:
            model = self.neural_net(layer_size=layer_size, feature_num=feature_num)
        else:
            model = self.load_dnn_model(
                name=str(num_run) + "_model_output_warm_up",
                layer_size=layer_size,
            )
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        if early_stop:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=self.param[str(layer_size[:-1])]["delta"],
                patience=self.param[str(layer_size[:-1])]["patience"],
            )
            callbacks = [early_stopping]
        else:
            callbacks = []
        model.compile(optimizer=opt, loss=MeanSquaredError())
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=self.param[str(layer_size[:-1])]["epochs"],
            validation_split=0.2,
            shuffle=False,
            verbose=1,
            callbacks=callbacks,
        )
        if save_up or self.warm_up:
            model.save(
                self.model_p.joinpath(
                    str(num_run)
                    + "_model_output"
                    + self.warm_up_ext
                    + "_"
                    + NN.neurons_ext(layer_size)
                    + ".h5"
                )
            )
        return model, history.history["loss"], history.history["val_loss"]

    def get_min_max(self):
        """
        Get min and max value for every key of the dictionary.
        """
        super(Output, self).get_min_max(dict_p=self.dict_p)

    def get_nn_type(self) -> str:
        """
        Returns type of state neural network: plnn, icnn or monotonic_icnn
        """
        return self.ad


class OutputDNN(Output):
    """
    Learning output function as input convex (non-decreasing) neural network.
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        columns,
        normalize,
        warm_up,
        warm_up_ext,
    ):
        super().__init__(
            result_p,
            model_p,
            time_delay,
            columns,
            normalize,
            warm_up,
            warm_up_ext,
            OutputDNNParam,
        )

    def custom_loss(self, alpha, y_min):
        """
        Customized MSE that tries to encourage lowest error in optimum.
        """

        def loss(y_true, y_pred):
            mean_sqr_error = K.mean(K.square(y_true - y_pred))
            p = K.mean(alpha * K.square(y_pred - y_min))
            error = mean_sqr_error + p
            return error

        return loss

    def neural_net(self, layer_size, feature_num):
        """
        Create neural network convex in its inputs.
        """
        input_layer = Input(shape=(feature_num,))
        layer = None
        for n, nn in enumerate(layer_size):
            # first hidden layer
            if layer is None:
                layer = Dense(
                    nn,
                    activation="relu",
                    kernel_constraint=ParNonNeg(
                        time_delay=0,
                        param_start=0,
                        param_end=feature_num,
                        feature_num=feature_num,
                    ),
                    use_bias=True,
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            else:
                layer_forward = Dense(
                    nn,
                    use_bias=True,
                    kernel_constraint=NonNeg(),
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(layer)
                layer_pass = Dense(
                    nn,
                    kernel_constraint=ParNonNeg(
                        time_delay=0,
                        param_start=0,
                        param_end=feature_num,
                        feature_num=feature_num,
                    ),
                    use_bias=False,
                )(input_layer)
                layer_merge = Add()([layer_forward, layer_pass])
                if n == len(layer_size) - 1:
                    layer = Activation("linear")(layer_merge)
                else:
                    layer = Activation("relu")(layer_merge)
        model = ModelNN(inputs=input_layer, outputs=layer)
        return model
