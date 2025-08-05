import os
import numpy as np
import ray
import warnings
from library.cobra import scaling, knock_out, knock_out_rebound, set_medium, set_medium_strict


###############################################################################
# Local functions
###############################################################################
def unload(result, batch_size):
    """
    For parallelization, unload lists of results into separated 'y' and 'warn'.

    Args:
    - result (list): A list of results, where each result is a sublist of shape (batch_size, 2).
    - batch_size (int): The size of each batch in the results.

    Returns:
    - y (list): A list containing 'y' prediction values from the results.
    - warn (list): A list containing indexes where solver cannot solve input medium and throw warning.
    """
    y, warn = [], []
    for i in range(len(result)):
        for j in range(batch_size):
            y.append(result[i][j][0])
            warn.append(result[i][j][1])
    return y, warn


def fit_cobra(i, medium_X, mediumname, cobramodel, gene, genename , objective):
    """
    Fit a COBRA model with a single data point.

    Parameters:
    - i (int): position in data array.

    Returns:
    - y_cobra (float or None): The optimized value of the specified objective function
    - rows_with_warnings (list): A list containing indices of rows where warnings were generated during optimization.
    """
    if i > medium_X.shape[0]: 
        print(f"Index {i} exceeds the number of samples {medium_X.shape[0]}.")
        return np.nan, []
    medium_array = medium_X[i,:]
    medium = dict(zip(mediumname, medium_array))
    rows_with_warnings = []

    with warnings.catch_warnings(record=True) as w:
        cobramodel = set_medium(cobramodel,medium)
#        cobramodel = set_medium_strict(cobramodel,medium)
#         if len(gene) > 0:
# #            cobramodel = knock_out(cobramodel, i, gene, genename)
#             cobramodel = knock_out_rebound(cobramodel,i,gene,genename, esp = 0)
        solution = cobramodel.optimize()
        y_cobra = solution.fluxes[objective]
        if w:
            y_cobra = np.nan
            rows_with_warnings.append(i)
            print(f"Sample {i}: y_cobra = {y_cobra}")
    return y_cobra, medium_array


@ray.remote
def cobra_one_batch(start, end, medium_X, mediumname, cobramodel, gene, genename , objective):
    """
    Fit COBRA model with a BATCH of X

    Parameters:
    - start (int): The start index of the batch in X array.
    - end (int): The end index of the batch in X array.
    - medium_X (numpy.ndarray): A 2D array contains SCALED medium conditions 

    Returns:
    - results (list): A list of results, where each result is a tuple containing 
        both y_cobra and row indices where there was warning.
    """
    if start >= end:
        print(f"Start index {start} is greater than or equal to end index {end}. No data to process.")
        return []
    return [fit_cobra(i, medium_X, mediumname, cobramodel, gene, genename , objective) for i in range(start, end)]


###############################################################################
# Callable function in other scripts
###############################################################################
def run_cobra_parallel(X, cobramodel, mediumsize, mediumname, fixedmediumvalue,  genename, objective):
    """
    Run parallel cobra predictions with a different batch on each CPU core

    Parameters:
    - X (numpy.ndarray): Input dataset for predictions.
    - cobramodel (cobra.Model): A COBRApy model to perform optimizations on.
    - mediumsize (int): The size of the medium.
    - mediumname (list): reaction names.
    - fixedmediumvalue (float): The scaling value.
    - genename (list): A list of gene names corresponding to the genes to be knocked out.
    - objective (str): The objective function to optimize in the COBRA model.

    Returns:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
#    cobramodel.solver.configuration = optlang.glpk_interface.Configuration(timeout=5, presolve='auto', lp_method='simplex'
    import math
    ray.shutdown()
    ray.init(num_gpus=0)
    X = scaling(X, mediumsize, fixedmediumvalue)
    medium = X[:,:mediumsize]
    # gene = X[:,mediumsize:]
    gene = []
    size = medium.shape[0]
    batch_size = math.floor(size/os.cpu_count())
    result_id = []

    for x in range(os.cpu_count()):
        start_idx = x * batch_size
        if x == os.cpu_count() - 1:
            end_idx = size
        else:
            end_idx = (x + 1) * batch_size
        result_id.append(cobra_one_batch.remote(start_idx,
                                        end_idx,
                                        medium, 
                                        mediumname, 
                                        cobramodel, 
                                        gene, 
                                        genename , 
                                        objective))

    result = ray.get(result_id)
    ray.shutdown()

    y_cobra, warning = unload(result, batch_size)
    y_cobra = np.array(y_cobra).ravel()
    return y_cobra, warning


def run_cobra_slow(X, cobramodel, mediumsize, mediumname, fixedmediumvalue,  genename, objective):
    """
    Run parallel cobra predictions without parallel

    Parameters:
    - X (numpy.ndarray): Input dataset for predictions.
    - cobramodel (cobra.Model): A COBRApy model to perform optimizations on.
    - mediumsize (int): The size of the medium.
    - mediumname (list): reaction names.
    - fixedmediumvalue (float): The scaling value.
    - genename (list): A list of gene names corresponding to the genes to be knocked out.
    - objective (str): The objective function to optimize in the COBRA model.

    Returns:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
    X = scaling(X, mediumsize, fixedmediumvalue)
    medium_X = X[:,:mediumsize]
    gene = X[:,mediumsize:]
    size = medium_X.shape[0]
    y_cobra = []
    rows_with_warnings = []
    for i in range(size):
        medium = dict(zip(mediumname, medium_X[i,:]))

        with warnings.catch_warnings(record=True) as w:
            cobramodel = set_medium(cobramodel,medium)
#            cobramodel = set_medium_strict(cobramodel,medium)
            if len(gene) > 0:
#                cobramodel = knock_out(cobramodel, i, gene, genename)
                cobramodel = knock_out_rebound(cobramodel,i,gene,genename, esp = 0)
            solution = cobramodel.optimize()
            y = solution.fluxes[objective]
            y_cobra.append(y)
            if w:
                rows_with_warnings.append(i)
                
    y_cobra = np.array(y_cobra).ravel()
    
    return y_cobra, rows_with_warnings


def run_cobra(XX, parameter):
    """
    Choose running cobra parallel for big dataset (biolog/biolog_mini) or without
    Return:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
    if parameter.parallel == True:
        return run_cobra_parallel(XX, 
                                    parameter.cobramodel, 
                                    parameter.mediumsize, 
                                    parameter.medium, 
                                    parameter.fixedmediumvalue, 
                                    parameter.genename, 
                                    parameter.objective)
    else:
        return run_cobra_slow(XX, 
                                    parameter.cobramodel, 
                                    parameter.mediumsize, 
                                    parameter.medium, 
                                    parameter.fixedmediumvalue, 
                                    parameter.genename, 
                                    parameter.objective)
        