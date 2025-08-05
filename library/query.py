import numpy as np
from library.parallel import run_cobra


###############################################################################
# Medium generation local functions
###############################################################################
def ucb_value(
    true,
    pred,
    pred_stdev,
    parameter
):
    """
    Calculate the Upper Confidence Bound (UCB) value.
    Args:
        true (np.array): The true value (ground truth).
        pred (np.array): The predicted value.
        pred_stdev (np.array): The standard deviation of the prediction.
        constant_ucb (float, optional): UCB constant (default is 0.5).
        threshold (float, optional): Threshold for non-regression mode (default is 0.1)
        regression (bool, optional): Whether to use regression mode (default is True).
    Returns:
        np.array: The UCB value.
    """
    # Local function that applies the ucb formula
    constant_ucb,regression,threshold = parameter.constant_ucb,parameter.regression,parameter.threshold
    if regression:
        exploitation = -((true - pred) ** 2)  # this is sme
        true[true == 0] = 1
        exploitation = exploitation / true 
        exploration =  pred_stdev
        # highest stdev first
    else:
        true[true == 0] = -1
    #    exploitation = ((threshold - pred) ** 2)
        exploitation = (pred-threshold)*true
    #    exploitation = exploitation / threshold if threshold else exploitation
        exploration =  pred_stdev
    ucb = (1 - constant_ucb) * exploitation + constant_ucb * exploration
    return ucb


def sampling_new_medium(parameter, n_generated):
    """
    Generate new data with new variable columns between min-max bounds.
    Args:
        parameter (object): An object containing various parameters including X, fixedmediumsize,
                          mediumsize, gene, min_medium, max_medium, and y_true.
        n_generated (int): The number of new data samples to generate.

    Returns:
        XX (numpy.ndarray): A 2D array containing the generated data 
        yy_true (numpy.ndarray): A 1D array containing the true labels correspondant to XX.
    """
    var_X = parameter.mask # avoid change every interation
    gene = parameter.gene

    min_medium = parameter.min_medium
    max_medium = var_X*parameter.scale #scaling from C concentration to V flux
#    max_medium = parameter.max_medium
    row, column = var_X.shape[0], var_X.shape[1]

    # generated new var_X between min_medium and max_medium bound
    XX = np.random.uniform(min_medium, max_medium,(n_generated,row,column))
#    XX[0] = var_X # save initial X
    XX = XX.reshape(-1).reshape(n_generated*row,column)
        
    # attached same fixed and gene
    #XX = np.concatenate((np.tile(fixed, (n_generated, 1)), XX), axis=1)
    if len(gene) > 0:
        XX = np.concatenate((XX, np.tile(gene, (n_generated, 1))), axis=1)

    # make y_true
    yy_true = np.tile(parameter.y, n_generated)

    # concat with previous data
    yy_true = np.concatenate((parameter.y_true, yy_true), axis = 0)
    XX = np.concatenate((parameter.X, XX), axis = 0)

    return XX, yy_true


def select_by_ucb(ucb, n_medium, n_generated, type ):
    """
    Find the top n_medium INDICES based on UCB values.

    Parameters:
        ucb (numpy.ndarray): 1D array of UCB values.
        n_medium (int): The number of medium indices to select.
        n_generated (int): The number of groups generated.

    Returns:
        idx (numpy.ndarray): 1D array containing the selected indices.
    """
    size_X = int(len(ucb)/n_generated)
    idx = np.array([])

    for i in range(size_X):
        group = []
        for j in range(n_generated):
            if type == 'model':
                group.append(ucb[i + j*size_X])
            if type == 'cobra':
                group.append(ucb[j + i*n_generated])
        
        group = np.array(group)
        idx_top_n = np.argpartition(-group, n_medium)[:n_medium]
        if type == 'model':
                idx_top_n = i + idx_top_n*size_X
        if type == 'cobra':
                idx_top_n = idx_top_n + i*n_generated

        idx = np.concatenate((idx,idx_top_n)).astype(np.int32)
    
    idx = idx.astype(int)
    return idx

###################################################################################
# Callable functions in other scripts
###################################################################################
def query_by_comittee(parameter, model = None):
    """
    Generate data, query based on a comittee of models and select a subset based on UCB values.
    Args:
        parameter (object): An object containing various parameters.
        model (object, optional): A machine learning model that has a predict method.
                               If None, a ValueError is raised.

    Returns:
        XX (numpy.ndarray): 2D array containing the selected data.
        yy_true (numpy.ndarray): 1D array containing the true labels corresponding to the selected data.
    """
    if model is None:
        raise ValueError("The 'model' argument cannot be None. Please provide a valid model.")
    
    n_medium = parameter.n_medium_ucb
    n_generated = 10 * n_medium
    print(n_generated)
    # Generate new data
    XX, yy_true = sampling_new_medium(parameter, n_generated)

    # Make predictions
    print("Query ensemble of ANN")
    y_pred_mean, y_pred_stdv = model.predict(XX)
    print("Calculating ucb")
    # Calculate UCB values
    ucb = ucb_value(yy_true, y_pred_mean, y_pred_stdv, parameter)

    # Select a subset based on UCB values
    idx = select_by_ucb(ucb, n_medium, n_generated, type = 'model')

    return XX[idx, :], yy_true[idx]


def query_by_cobra(XX, yy_true, parameter):
    """
    Select the best media conditions for COBRA predictions based on UCB values.

    Parameters:
    - XX (numpy.ndarray): Input dataset for COBRA predictions.
    - yy_true (numpy.ndarray): True labels corresponding to the input dataset.
    - parameter (object): An object containing various parameters including cobramodel,mediumsize,
                          medium, fixedmediumvalue, gene, genename, and objective.

    Returns:
    - selected_X (numpy.ndarray): The selected subset of input data based on UCB values.
    - selected_y_cobra (numpy.ndarray): The corresponding COBRA predictions for the selected data.
    - selected_y_cobra (numpy.ndarray): The corresponding ground truths for the selected data.
    """
    print("Query with Cobra")

    yy_cobra, _ = run_cobra(XX, parameter)
        
    print("Calculating ucb")
    ucb = ucb_value(yy_true, yy_cobra, 0, parameter)

    # Select the best X on UCB values
    idx = select_by_ucb(ucb, 1, parameter.n_medium_ucb, type = 'cobra')

#    return XX[idx, :], yy_cobra[idx], yy_true[idx]
    return idx, yy_cobra