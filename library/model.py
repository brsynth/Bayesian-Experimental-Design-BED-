##############################################################################
# models architechture and modelling metric
###############################################################################
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout


def my_r2(y_true, y_pred):
    # Custom metric function to evaluate model when training
    SS = K.sum(K.square(y_true - y_pred))
    ST = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS / (ST + K.epsilon())


class Models:
    # Local class
    def __init__(self, parameter):
        self.input_dim = parameter.X.shape[1]
#        self.output_dim = parameter.y_cobra.shape[0]
        self.n_hidden = parameter.n_hidden
        self.hidden_dim = parameter.hidden_dim
        self.n_models = parameter.n_models
        self.models = []
        self.dropout = parameter.dropout
        self.epochs = parameter.epochs
        self.batch_size = parameter.batch_size

    # Create one NN
    def create(self):
        inputs = Input(shape=(self.input_dim,))
        hidden = inputs
        for i in range(self.n_hidden):
            hidden = Dense(
                self.hidden_dim,
                kernel_initializer="random_normal",
                bias_initializer="zeros",
                activation="relu",
            )(hidden)
            hidden = Dropout(self.dropout)(hidden)
        outputs = Dense(
#            self.output_dim,   # unit = 1
            units=1,
            kernel_initializer="random_normal",
            bias_initializer="zeros",
            activation="linear",
        )(hidden)
        model = keras.models.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss="mse", optimizer="adam", metrics=[my_r2])
        return model

    def train(self, X, y, verbose=False):
        # Create n_models models
        if len(self.models) == 0:
            for i in range(self.n_models):
                self.models.append(self.create())
        # Train all models
        for i in range(len(self.models)):
            try:  # yes sometimes keras/tensorflow crashes
                self.models[i].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    verbose=verbose,
                )
            except:  # recreate model and fit again
                self.models[i] = self.create()
                self.models[i].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    verbose=verbose,
                )

    # predict n_models times and calculate average and stdv
    def predict(self, X):
        y_pred = []
        for i in range(self.n_models):
            y_p = self.models[i].predict(X, verbose=None)
            y_pred.append(y_p)
        y_pred = np.array(y_pred)
        y_mean = np.average(y_pred, axis=0).ravel()
        y_stdv = np.std(y_pred, axis=0).ravel()
        return y_mean, y_stdv
