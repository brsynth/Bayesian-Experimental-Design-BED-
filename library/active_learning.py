###############################################################################
# This library perform Active Learning to find Cobra best intake medium fluxes
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Created: June 26, 2023. Updated: Jul 13, 2023
###############################################################################

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from library.utils import get_r2
from library.cobra import run_cobra
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # Tensorflow WARNINGS because of GPU unused, silence it
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

###############################################################################
# Keras & Tensorflow local functions
###############################################################################


def my_r2(y_true, y_pred):
    # Custom metric function
    SS = K.sum(K.square(y_true-y_pred))
    ST = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS/(ST + K.epsilon())

def my_binary_crossentropy(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.binary_crossentropy(y_true[:, :end], y_pred[:, :end])

class Models:
    # Local class
    def __init__(self, parameter):
        self.input_dim = parameter.X.shape[1]
        self.output_dim = parameter.y_cobra.shape[1]
        self.n_hidden = parameter.n_hidden
        self.hidden_dim = parameter.hidden_dim
        self.n_models = parameter.n_models
        self.models = []
        self.dropout = parameter.dropout
        self.epochs = parameter.epochs
        self.batch_size = parameter.batch_size

    def create(self, verbose=False, regression=True):
        inputs = Input(shape=(self.input_dim,))
        hidden = inputs
        for i in range(self.n_hidden):
            hidden = Dense(self.hidden_dim,
                           kernel_initializer='random_normal',
                           bias_initializer='zeros',
                           activation='relu')(hidden)
            hidden = Dropout(self.dropout)(hidden)
        outputs = Dense(self.output_dim,
                        kernel_initializer='random_normal',
                        bias_initializer='zeros',
                        activation='relu')(hidden)
        model = keras.models.Model(inputs=[inputs], outputs=[outputs])
        if regression:
            model.compile(loss='mse', optimizer='adam', metrics=[my_r2])
        return model

    def predict(self, X):
        y_pred = []
        for i in range(self.n_models):
            y_p = self.models[i].predict(X, verbose=None)
            y_pred.append(y_p)
        y_pred = np.array(y_pred)
        y_mean = np.average(y_pred, axis=0)
        y_stdv = np.std(y_pred, axis=0)
        return y_mean, y_stdv

    def train(self, X, y, regression=True, verbose=False):
        # Create models
        if len(self.models) == 0:
            for i in range(self.n_models):
                self.models.append(self.create(regression=regression, verbose=verbose))
        # Train all models
        for i in range(len(self.models)):
            try:  # yes sometimes keras/tensorflow crashes
                self.models[i].fit(X, y,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   verbose=verbose)
            except:  # recreate model and fit again
                self.models[i] = self.create(regression=regression, verbose=verbose)
                self.models[i].fit(X, y,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   verbose=verbose)
            
###############################################################################
# Media generation local functions
###############################################################################


def ucb_value(true, pred, pred_stdev,
              constant_ucb=0.5, verbose=False):
    # Local function that applies the ucb formula
    exploitation = -((true - pred) ** 2)  # this is sme
    exploitation = exploitation/true if true else exploitation
    # highest stdev first
    exploration = -pred_stdev
    ucb = (1-constant_ucb) * exploitation + constant_ucb * exploration
    return ucb


def build_media(M, parameter, verbose=False):
    # local function to build media using ucb

    def build_random_medium(n_medium, mask, parameter):
        X = np.zeros((n_medium, mask.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if j < parameter.fixedmediumsize:
                    X[i, j] = parameter.fixedmediumvalue
                else:
                    X[i, j] = parameter.min_medium
                    X[i, j] += parameter.max_medium * np.random.random()
                    X[i, j] = X[i,j] * mask[j]
        return X

    # mask is a 0,1 mask
    mask = parameter.Xmask
    mask[mask != 0] = 1

    # One generate 10 times the nbr of medium to be added
    X, y = parameter.X, parameter.y_true
    n_medium_ucb = parameter.n_medium_ucb
    n_medium = 10 * n_medium_ucb

    # create media
    # initialize with previous training set
    # add n_medium * mask.shape[0] elements
    XX, Xucb = np.array([[]]), np.array([[]])
    yy, yucb = np.array([[]]), np.array([[]])
    for i in range(mask.shape[0]):
        Xi = build_random_medium(n_medium, mask[i], parameter)
        yi = np.array([y[i]]*n_medium)
        Xi[0] = X[i]  # keep the initial value
        XX = np.concatenate((XX, Xi), axis=0) if XX.shape[1] else Xi
        yy = np.concatenate((yy, yi), axis=0) if yy.shape[1] else yi

    # make prediction
    y_pred_mean, y_pred_stdv = M.predict(XX)  
    if verbose == 2:
        print(f'ucb set size {yy.shape}')

    # select prediction according to ucb for each medium
    for i in range(mask.shape[0]):
        ym = y_pred_mean[i*n_medium:(i+1)*n_medium, :]
        ys = y_pred_stdv[i*n_medium:(i+1)*n_medium, :]
        yt = yy[i*n_medium:(i+1)*n_medium, :]

        # Get sorted according to ucb
        yucb_dict = {}
        for j in range(ym.shape[0]):
            yucb_dict[j] = ucb_value(yt[j], ym[j], ys[j])
        yucb_dict = {k: v for k, v in sorted(yucb_dict.items(),
                                             key=lambda item: item[1])}

        # Get top n_medium_ucb indices
        I = np.array(list(yucb_dict.keys()))[0:n_medium_ucb]
        if verbose == 2:
            print(f'y index {i}, best ucb indices {I}')
        Xucbi = XX[i*n_medium:(i+1)*n_medium, :]
        yucbi = yy[i*n_medium:(i+1)*n_medium, :]
        Xucbi, yucbi = Xucbi[I], yucbi[I]
        Xucb = np.concatenate((Xucb, Xucbi), axis=0) \
        if Xucb.shape[1] else Xucbi
        yucb = np.concatenate((yucb, yucbi), axis=0) \
        if yucb.shape[1] else yucbi

    return Xucb, yucb


def select_cobra_media(X, y_true, N, n, cobramodel,
                       fixedmedium, fixedmediumvalue,
                       variablemedium,
                       objective,
                       verbose=False):
    # local function
    # the cell is equivalent of performing measurements on the
    # provided ucb generated media X (n media for N different
    # y_true experimental values)
    # Run Cora with media X
    y_cobra = run_cobra(X,
                        cobramodel,
                        fixedmedium,
                        fixedmediumvalue,
                        variablemedium,
                        objective)

    # Get best media i.e., the X vectors minimizing
    # sme(y_true, y_cobra) for each different y_true values
    K = []
    for i in range(N):
        yc = y_cobra[i*n:(i+1)*n, :]
        yt = y_true[i*n:(i+1)*n, :]
        # Get sorted according to ucb
        yucb_dict = {}
        for j in range(yc.shape[0]):
            yucb_dict[j] = ucb_value(yt[j], yc[j], 0)
        yucb_dict = {k: v for k, v in sorted(yucb_dict.items(),
                                             key=lambda item: item[1])}
        ki = list(yucb_dict.keys())[0]  # first is best
        K.append(i*n+ki)
    return X[K], y_cobra[K]

###############################################################################
# Active Learning local functions
###############################################################################

def active_learning_train(M, parameter, regression=True, verbose=False):
    # Local function
    # For parameters cf. active_learning below

    X, y_true, y_cobra = parameter.X, parameter.y_true, parameter.y_cobra
    fixedmedium = parameter.medium[:parameter.fixedmediumsize]
    variablemedium = parameter.medium[parameter.fixedmediumsize:]

    # Train and evaluate models on y_cobra data
    M.train(X, y_cobra, regression=regression, verbose=False)
    y_pred_mean, y_pred_stdv = M.predict(X)
    if regression:
        r2_cobra_pred = get_r2(y_cobra, y_pred_mean)
        r2_true_pred = get_r2(y_true, y_pred_mean)

    # Create new media (XX) and predict yy values
    XX, yy_true = build_media(M, parameter, verbose=verbose)

    # Perform measurments (i.e. run Cobra) and
    # get best media i.e., the X vectors minimizing
    # sme(y_true, y_cobra) for each different y_true values
    X, y_cobra = select_cobra_media(XX, yy_true,
                                    parameter.X.shape[0],
                                    parameter.n_medium_ucb,
                                    parameter.cobramodel,
                                    fixedmedium,
                                    parameter.fixedmediumvalue,
                                    variablemedium,
                                    parameter.objective,
                                    verbose=verbose)
    if regression:
        r2_true_cobra = get_r2(y_true, y_cobra)

    return X, y_cobra, r2_cobra_pred, r2_true_pred, r2_true_cobra


def active_learning_test(M, parameter, verbose=False):
    # Local function
    # For parameters cf. active_learning below
    X, y_true, y_cobra = parameter.X, parameter.y_true, parameter.y_cobra
    fixedmedium = parameter.medium[:parameter.fixedmediumsize]
    variablemedium = parameter.medium[parameter.fixedmediumsize:]

    # Predict and get y_cobra
    y_pred_mean, y_pred_stdv = M.predict(X)
    y_cobra = run_cobra(X,
                        parameter.cobramodel,
                        fixedmedium,
                        parameter.fixedmediumvalue,
                        variablemedium,
                        parameter.objective)
    # Stats
    r2_cobra_pred = get_r2(y_cobra, y_pred_mean)
    r2_true_pred = get_r2(y_true, y_pred_mean)
    r2_true_cobra = get_r2(y_true, y_cobra)

    return X, y_cobra, r2_cobra_pred, r2_true_pred, r2_true_cobra


###############################################################################
# Active Learning callable functions
###############################################################################

class active_learning_stats:

    def __init__(self, v1, v2, v3, v4, v5, v6):
        self.cobra_pred_mean_train = np.mean(v1)
        self.cobra_pred_stdv_train = np.std(v1)
        self.true_pred_mean_train = np.mean(v2)
        self.true_pred_stdv_train = np.std(v2)
        self.true_cobra_mean_train = np.mean(v3)
        self.true_cobra_stdv_train = np.std(v3)
        self.cobra_pred_mean_test = np.mean(v4)
        self.cobra_pred_stdv_test = np.std(v4)
        self.true_pred_mean_test = np.mean(v5)
        self.true_pred_stdv_test = np.std(v5)
        self.true_cobra_mean_test = np.mean(v6)
        self.true_cobra_stdv_test = np.std(v6)

    def printout(self, text, train=True, test=False):
        scp, stp, stc = '', '', ''
        if train:
            scp += f'cobra-pred: R2 {self.cobra_pred_mean_train:.3f}'
            scp += f'+/-{self.cobra_pred_stdv_train:.3f}'
            stp += f'true-pred:  R2 {self.true_pred_mean_train:.3f}'
            stp += f'+/-{self.true_pred_stdv_train:.3f}'
            stc += f'true-cobra: R2 {self.true_cobra_mean_train:.3f}'
            stc += f'+/-{self.true_cobra_stdv_train:.3f}'
        if test:
            scp += f' cobra-pred: Q2 {self.cobra_pred_mean_test:.3f}'
            scp += f'+/-{self.cobra_pred_stdv_test:.3f}'
            stp += f' true-pred:  Q2 {self.true_pred_mean_test:.3f}'
            stp += f'+/-{self.true_pred_stdv_test:.3f}'
            stc += f' true-cobra: Q2 {self.true_cobra_mean_test:.3f}'
            stc += f'+/-{self.true_cobra_stdv_test:.3f}'
        print(f'{text} {scp}')
        print(f'{text} {stp}')
        print(f'{text} {stc}')


def active_learning_fold_loop(parameter, regression=True, verbose=False):
    # Active learning loop is run for each train-test fold
    # ARGUMENTS and RETURNS are the same than active_learning below 
    fixedmedium = parameter.medium[:parameter.fixedmediumsize]
    variablemedium = parameter.medium[parameter.fixedmediumsize:]
    X = np.copy(parameter.X)
    y_true = np.copy(parameter.y_true)
    y_cobra = np.copy(parameter.y_cobra)

    # no test set
    if parameter.xfold < 2:

        # train with active learning loop
        M = Models(parameter)
        parameter.Xmask = np.copy(parameter.X)
        for iloop in range(parameter.Nloop):
            parameter.X, parameter.y_cobra, cp, tp, tc = \
            active_learning_train(M, parameter, verbose=verbose)
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f'TRAIN no-fold iteration: {iloop+1}')

        X_test = parameter.X
        y_true, y_cobra_test = parameter.y_true, parameter.y_cobra
        stats = active_learning_stats(cp, tp, tc, cp, tp, tc)

        # restore initial parameter
        parameter.X = X
        parameter.y_true = y_true
        parameter.y_cobra = y_cobra

        return X_test, y_true, y_cobra_test, stats

    # xfold initialization
    kfold = KFold(n_splits=parameter.xfold, shuffle=True)
    X_test = np.zeros(parameter.X.shape)
    y_cobra_test = np.zeros(parameter.y_cobra.shape)
    CP_TRAIN, TP_TRAIN, TC_TRAIN = [], [], []
    CP_TEST,  TP_TEST,  TC_TEST = [], [], []

    # xfold test set loop
    fold = 0
    for train, test in kfold.split(X, y_true):
        fold += 1

        # train with active learning loop
        parameter.X, parameter.Xmask = np.copy(X[train]), np.copy(X[train])
        parameter.y_true = np.copy(y_true[train])
        parameter.y_cobra = np.copy(y_cobra[train])
        M = Models(parameter)
        for iloop in range(parameter.Nloop):
            parameter.X, parameter.y_cobra, cp, tp, tc = \
            active_learning_train(M, parameter, regression=regression, verbose=verbose)
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f'TRAIN fold: {fold} iteration: {iloop+1}')

        CP_TRAIN.append(cp)
        TP_TRAIN.append(tp)
        TC_TRAIN.append(tc)

        # predict on test set
        parameter.X, parameter.Xmask = np.copy(X[test]), np.copy(X[test])
        parameter.y_true = np.copy(y_true[test])
        parameter.y_cobra = np.copy(y_cobra[test])
        parameter.X, parameter.y_cobra, cp, tp, tc = \
        active_learning_test(M, parameter, verbose=verbose)
        if verbose:
            s = active_learning_stats(cp, tp, tc, cp, tp, tc)
            s.printout(f'TEST  fold: {fold}', train=False, test=True)
        CP_TEST.append(cp)
        TP_TEST.append(tp)
        TC_TEST.append(tc)
        for i in range(len(test)):
            X_test[test[i]] = parameter.X[i]
            y_cobra_test[test[i]] = parameter.y_cobra[i]

    # stats for all folds
    stats = active_learning_stats(CP_TRAIN, TP_TRAIN, TC_TRAIN,
                                  CP_TEST, TP_TEST, TC_TEST)

    # restore initial parameter
    parameter.X = X
    parameter.y_true = y_true
    parameter.y_cobra = y_cobra

    return X_test, y_true, y_cobra_test, stats


def active_learning_loop_fold(parameter, regression=True, verbose=False):
    # Train-test fold is carried out for each active learning fold
    # ARGUMENTS and RETURNS are the same than active_learning below
    X, XX = parameter.X, np.copy(parameter.X)
    y_true, yy_true = parameter.y_true, np.copy(parameter.y_true)
    y_cobra, yy_cobra = parameter.y_cobra, np.copy(parameter.y_cobra)
    for iloop in range(parameter.Nloop):
        parameter.X = X
        parameter.y_true = y_true
        parameter.y_cobra = y_cobra

        # train only
        if parameter.xfold < 2:
            M = Models(parameter)
            parameter.Xmask = np.copy(parameter.X)
            parameter.X, parameter.y_cobra, cp, tp, tc = \
            active_learning_train(M, parameter, regression=regression, verbose=verbose)
            X_test = parameter.X
            y_true, y_cobra_test = parameter.y_true, parameter.y_cobra
            stats = active_learning_stats(cp, tp, tc, cp, tp, tc)
            if verbose:
                stats.printout(f'TRAIN no-fold iteration: {iloop+1}')
            continue

        # xfold initialization
        kfold = KFold(n_splits=parameter.xfold, shuffle=True)
        X_test = np.zeros(parameter.X.shape)
        y_cobra_test = np.zeros(parameter.y_cobra.shape)
        CP_TRAIN, TP_TRAIN, TC_TRAIN = [], [], []
        CP_TEST,  TP_TEST,  TC_TEST = [], [], []
        # xfold test set loop
        fold = 0
        for train, test in kfold.split(X, y_true):
            fold += 1

            # train with active learning loop
            parameter.X, parameter.Xmask = X[train], X[train]
            parameter.y_true = y_true[train]
            parameter.y_cobra = y_cobra[train]
            M = Models(parameter)
            parameter.X, parameter.y_cobra, cp, tp, tc = \
            active_learning_train(M, parameter, regression=regression, verbose=verbose)
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f'TRAIN iteration: {iloop+1} fold: {fold}')
            CP_TRAIN.append(cp)
            TP_TRAIN.append(tp)
            TC_TRAIN.append(tc)
            X[train] = parameter.X
            y_cobra[train] = parameter.y_cobra

            # predict on test set
            parameter.X, parameter.Xmask = X[test], X[test]
            parameter.y_true = y_true[test]
            parameter.y_cobra = y_cobra[test]
            parameter.X, parameter.y_cobra, cp, tp, tc = \
            active_learning_test(M, parameter, verbose=verbose)
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f'TEST  iteration: {iloop+1} fold: {fold}',
                           train=False, test=True)
            CP_TEST.append(cp)
            TP_TEST.append(tp)
            TC_TEST.append(tc)
            X_test[test] = parameter.X
            y_cobra_test[test] = parameter.y_cobra

        # stats for all folds
        stats = active_learning_stats(CP_TRAIN, TP_TRAIN, TC_TRAIN,
                                      CP_TEST, TP_TEST, TC_TEST)
        if verbose:
            stats.printout(f'TRAIN-TEST iteration: {iloop+1}',
                           train=True, test=True)

    # restore initial parameter
    parameter.X = XX
    parameter.y_true = yy_true
    parameter.y_cobra = yy_cobra

    return X_test, y_true, y_cobra_test, stats


def active_learning(parameter, regression=True, verbose=False):
    # Given some measurements (y_true) for input medium intake fluxes (X),
    # this cell run an active learning loop to find the medium intake
    # fluxes (X) minimizing a upper-confidence bound (ucb) function.
    # The ucb function is composed of exploitation and exploration terms.
    # The exploitation term is the sme between y_true and y_cobra values
    # where y_cobra is computed via Cobra providing X as input.
    # The exploration term is the opposite of the stdv for y_pred values
    # predicted by an ensemble of neural network models (M) trained on
    # Cobra's output.  The neural models provide a vector X minimizing
    # the ucb function that vector is fed to Cobra.
    # ARGUMENTS:
    # parameter.Nloop: the number of loops
    # parameter.xfold = 1 # validation set fold 1 = no validation set
    # parameter.n_models = 5 # the number of neural network models
    # parameter.constant_ucb = 0.1 # ratio exploration vs. exploitation
    # parameter.n_medium_ucb = 5 # nbr of medium created for each y_true
    # parameter.min_medium = 0 # minimum value medium can take
    # parameter.max_medium = 5 # maximum value medium can take
    # parameter.X
    # parameter.y_true
    # parameter.y_cobra
    # RETURNS:
    # Updated X, (input y_true) and y_cobra vectors and active learning stats

    if parameter.mode == 'fold_loop':
        return active_learning_fold_loop(parameter,
                                         regression=regression,
                                         verbose=verbose)
    else:
        return active_learning_loop_fold(parameter,
                                         regression=regression,
                                         verbose=verbose)
