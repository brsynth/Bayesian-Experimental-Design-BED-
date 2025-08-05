from library.utils import measure_performance
from library.query import query_by_comittee, query_by_cobra
from library.model import Models
from library.parallel import run_cobra
import numpy as np
from sklearn.model_selection import KFold


###############################################################################
# Active Learning performance class
###############################################################################
class active_learning_stats:
    def __init__(self, v1, v2, v3, v4 = [], v5 = [], v6 = []):
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

    def printout(self, text, regression=True, train=True, test=False):
        """
        Print performance metrics for training and/or testing.

        Parameters:
        - text (str): A prefix text to include in the printed output.
        - regression (bool, optional): If True, treat the problem as a regression task; 
            if False, treat it as a classification task.
        - train (bool, optional): If True, print training metrics.
        - test (bool, optional): If True, print testing metrics.

        Prints:
        - Performance metrics for COBRA predictions and true values, including R2 (regression) or Accuracy (classification).
        """
        scp, stp, stc = "", "", ""
        if regression: 
            metric = "R2"
        else: metric = "Accuracy"
        if train:
            scp += f" R2 cobra-pred: {self.cobra_pred_mean_train:.3f}"
            scp += f"+/-{self.cobra_pred_stdv_train:.3f}"
            stp += f"{metric} true-pred: {self.true_pred_mean_train:.3f}"
            stp += f"+/-{self.true_pred_stdv_train:.3f}"
            stc += f"{metric} true-cobra: {self.true_cobra_mean_train:.3f}"
            stc += f"+/-{self.true_cobra_stdv_train:.3f}"
        if test:
            scp += f"Q2 cobra-pred: {self.cobra_pred_mean_test:.3f}"
            scp += f"+/-{self.cobra_pred_stdv_test:.3f}"
            stp += f"{metric} true-pred:  Q2 {self.true_pred_mean_test:.3f}"
            stp += f"+/-{self.true_pred_stdv_test:.3f}"
            stc += f"{metric} true-cobra: Q2 {self.true_cobra_mean_test:.3f}"
            stc += f"+/-{self.true_cobra_stdv_test:.3f}"
        print(f"{text} {scp}")
        print(f"{text} {stp}")
        print(f"{text} {stc}")


###############################################################################
# Local functions
###############################################################################
def active_learning_train(M, parameter, regression, threshold):
    """
    Perform comittee training using COBRA predictions and pick best new XX by active learning

    Parameters:
    - M (object): The machine learning models TO BE TRAINED as comittee.
    - parameter (object): An object containing various parameters including input data (X), 
                          true labels (y_true), COBRA predictions (y_cobra), and others.
    - regression (bool): by default True, treat the problem as a regression task, 
                        if False, treat it as a classification task.
    - threshold (float): Threshold for classification, by default 0.1

    Returns:
    - XX (numpy.ndarray): Best generated XX by active learning
    - yy_cobra (numpy.ndarray): COBRA predictions for new best XX
    - cobra_pred (float): Performance measure for comittee predictions on COBRA prediction.
    - true_pred (float): Performance measure for comittee predictions on initial data.
    - true_cobra (float): Performance measure for COBRA predictions on initial data.
    """
    X, y_true, y_cobra = parameter.X, parameter.y_true, parameter.y_cobra

    # Train and evaluate models on y_cobra data
    M.train(X, y_cobra)
    y_pred_mean, _ = M.predict(X)
    cobra_pred = measure_performance(y_cobra, y_pred_mean,regression, threshold)
    true_pred = measure_performance(y_true, y_pred_mean,regression, threshold)

    # Generate and select new media (XX) and predict yy values
    XX, yy_true = query_by_comittee(parameter, M)
#   XX, yy_cobra, yy_true = query_by_cobra(XX, yy_true, parameter) #select best by cobra
#   true_cobra = measure_performance(yy_true, yy_cobra,regression, threshold)
    
    idx, yy_cobra = query_by_cobra(XX, yy_true, parameter)
    yy_cobra_best, yy_true_best = yy_cobra[idx], yy_true[idx]
    true_cobra = measure_performance(yy_true_best, yy_cobra_best,regression, threshold)

    # update new parameter
    parameter.X, parameter.y_cobra, parameter.y_true = XX, yy_cobra, yy_true

    return XX, yy_cobra, yy_true, cobra_pred, true_pred, true_cobra


def active_learning_test(M, parameter, regression, threshold):
    """
    Perform comittee training using COBRA predictions and pick best new XX by active learning

    Parameters:
    - M (object): The TRAINED models as comittee.
    - parameter (object): An object containing various parameters including input data (X), 
                          true labels (y_true), COBRA predictions (y_cobra), and others.
    - regression (bool): by default True, treat the problem as a regression task, 
                        if False, treat it as a classification task.
    - threshold (float): Threshold for classification, by default 0.1

    Returns:
    - XX (numpy.ndarray): Best generated XX by active learning
    - yy_cobra (numpy.ndarray): COBRA predictions for new best XX
    - cobra_pred (float): Performance measure for comittee predictions on COBRA prediction.
    - true_pred (float): Performance measure for comittee predictions on initial data.
    - true_cobra (float): Performance measure for COBRA predictions on initial data.
    """
    X, y_true = parameter.X, parameter.y_true

    # Predict and get y_cobra
    y_pred, _ = M.predict(X)

    y_cobra, _ = run_cobra(X, parameter)

    cobra_pred = measure_performance(y_cobra, y_pred, regression, threshold)
    true_pred = measure_performance(y_true, y_pred, regression, threshold)
    true_cobra = measure_performance(y_true, y_cobra, regression, threshold)

    return X, y_cobra, y_true, cobra_pred, true_pred, true_cobra


def active_learning(parameter, verbose=False):
    # Active learning loop is run for each train-test fold
    # ARGUMENTS and RETURNS are the same than active_learning below
    regression, threshold = parameter.regression, parameter.threshold
    X = np.copy(parameter.X)
    y_true = np.copy(parameter.y_true)
    y_cobra = np.copy(parameter.y_cobra)
    stat = []

    # no test set
    if parameter.xfold < 2:
        # train with active learning loop
        M = Models(parameter)
        for iloop in range(parameter.Nloop):
            parameter.X, parameter.y_cobra, parameter.y_true, cp, tp, tc = active_learning_train(
                M,
                parameter,
                regression=regression,
                threshold=threshold
            )
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f"TRAIN no-fold iteration: {iloop+1}")
            
            stat.append([iloop+1, cp, tp, tc])

        X_test = parameter.X
        y_true, y_cobra_test = parameter.y_true, parameter.y_cobra
        stats = active_learning_stats(cp, tp, tc, cp, tp, tc)

        # restore initial parameter
        parameter.X = X
        parameter.y_true = y_true
        parameter.y_cobra = y_cobra

        return X_test, y_true, y_cobra_test, stats, stat

    # xfold initialization
    kfold = KFold(n_splits=parameter.xfold, shuffle=True)
    X_test = np.zeros(parameter.X.shape)
    y_cobra_test = np.zeros(parameter.y_cobra.shape)
    CP_TRAIN, TP_TRAIN, TC_TRAIN = [], [], []
    CP_TEST, TP_TEST, TC_TEST = [], [], []

    # xfold test set loop
    fold = 0
    for train, test in kfold.split(X, y_true):
        fold += 1

        # train with active learning loop
        parameter.X = np.copy(X[train])
        parameter.y_true = np.copy(y_true[train])
        parameter.y_cobra = np.copy(y_cobra[train])
        M = Models(parameter)
        for iloop in range(parameter.Nloop):
            parameter.X, parameter.y_cobra, parameter.y_true, cp, tp, tc = active_learning_train(
                M,
                parameter,
                regression=regression,
                threshold=threshold
            )
            if verbose:
                s = active_learning_stats(cp, tp, tc, cp, tp, tc)
                s.printout(f"TRAIN fold: {fold} iteration: {iloop+1}")
            
            stat.append([iloop+1, cp, tp, tc])

            CP_TRAIN.append(cp)
            TP_TRAIN.append(tp)
            TC_TRAIN.append(tc)

        # predict on test set
        parameter.X, parameter.Xmask = np.copy(X[test]), np.copy(X[test])
        parameter.y_true = np.copy(y_true[test])
        parameter.y_cobra = np.copy(y_cobra[test])
        parameter.X, parameter.y_cobra, parameter.y_true, cp, tp, tc = active_learning_test(
            M, parameter, regression=regression, threshold=threshold
        )
        if verbose:
            s = active_learning_stats(cp, tp, tc, cp, tp, tc)
            s.printout(f"TEST  fold: {fold}", train=False, test=True)
            CP_TEST.append(cp)
            TP_TEST.append(tp)
            TC_TEST.append(tc)
        for i in range(len(test)):
            X_test[test[i]] = parameter.X[i]
            y_cobra_test[test[i]] = parameter.y_cobra[i]

    # stats for all folds
    stats = active_learning_stats(
        CP_TRAIN, TP_TRAIN, TC_TRAIN, CP_TEST, TP_TEST, TC_TEST
    )

    # restore initial parameter
    parameter.X = X
    parameter.y_true = y_true
    parameter.y_cobra = y_cobra

    return X_test, y_true, y_cobra_test, stats, stat
