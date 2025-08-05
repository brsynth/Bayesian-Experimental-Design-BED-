# This file contain a dictionary of all specification 
# of each dataset and active learning experiments
specification = {
    "iML1515_EXP": {
        "mediumpath": "./Dataset_input/iML1515_EXP",
        "cobrapath": "./Dataset_input/iML1515_EXP", # full map
        "mediumbound": "UB",
        "fixedmediumsize": 28,
        "mediumsize": 38,
        "genesize": 0,
        "fixedmediumvalue": 2.5,
        "scale": 2, 
        "method": "EXP",# use for experimental data
        "regression": True,
        "threshold": None,
        "parallel": False
    },
    "iML1515_Paul_EXP": {
        "mediumpath": "./Dataset_input/iML1515_Paul_EXP", 
#        "mediumpath": "./Result/iML1515_Paul_EXP_AL_rep1_exact_bound_solution",
        "cobrapath": "./Dataset_input/iML1515_duplicated", # full map
        "mediumbound": "UB",
        "fixedmediumsize": 23,
        "mediumsize": 51,
        "genesize": 0,
        "fixedmediumvalue": 7.5,
        "scale": 7.5, 
        "method": "EXP",# use for experimental data
        "regression": True,
        "threshold": None,
        "parallel": True
    },
    "biolog_iML1515_EXP": { # E. coli grown in 120 unique media compositions and 145 different single metabolic gene KOs
        "mediumpath": "./Dataset_input/biolog_iML1515_EXP", # Ecoli gene KO mutants, medium [0,50] 
        "cobrapath": "./Dataset_input/iML1515_duplicated",  # simplify map
        "mediumbound": "UB",
        "fixedmediumsize": 23,
        "mediumsize": 151,
        "genesize": 430,
        "fixedmediumvalue": 3,
        "method": "EXP",
        "regression": True, # could also be classification 
        "threshold": 0.165, # 5% max y_true
        "parallel": True
        },
    "biolog_iML1515_mini": { # E. coli grown in 120 unique media compositions and 145 different single metabolic gene KOs
        "mediumpath": "./Dataset_input/biolog_iML1515_EXP_0.1", # Ecoli gene KO mutants, medium [0,50] 
        "cobrapath": "./Dataset_input/iML1515_duplicated",  # simplify map
        "mediumbound": "UB",
        "fixedmediumsize": 23,
        "mediumsize": 151,
        "genesize": 430,
        "fixedmediumvalue": 3,
        "method": "EXP",
        'seed': 42, # to regenerated this dataset from biolog_iMN1515
        "regression": True, # could also be classification 
        "threshold": 0.1585,
        "parallel": True
    },
    "IJN1463_EXP": {
        "mediumpath": "./Dataset_input/IJN1463_EXP_modify", # P.Putida medium [0,1], 
        # re-arranged column 23: EX_glc__D_e_i [0.   1.   0.63] before column 16: EX_nh4_e_i [1. 0.] 
        # therefore fixed medium and variable medium are seperable
        "cobrapath": "./Dataset_input/IJN1463_duplicated", # simplify map
        "mediumbound": "UB",
        "fixedmediumsize": 21,
        "mediumsize": 196,
        "genesize": 0,
        "fixedmediumvalue": 1,
        "method": "EXP",
        "regression": False, 
        "threshold": 0.02,
        "parallel": False
    },
}


def print_specification(input):
    """
    Print out specification in dictionary input
    """
    print("Input parameter")
    for k in input.keys():
        print(k,':', input[k]) 


###############################################################################################
# ACTIVE LEARNING specification
###############################################################################################
active_input = {
    "iML1515_EXP": {    
        'file' : "./Dataset_model/iML1515_EXP",
        'solver': 'glpk', # default in cobrapy, could pick between glpk, scipy and gurobi
        'Nloop' : 10, # number of active learning loops
        'constant_ucb' : 0, 
        'n_medium_ucb' : 5, 
        'min_medium' : 0,
        'scale': 2, # ratio between intake FLUXES for variable mediums / fixed mediums  
        'mode' : "fold_loop",
        'model' : "iML1515_EXP"
    },
    "iML1515_Paul_EXP": {
        'file' : "./Dataset_model/iML1515_Paul_EXP",
        'solver': 'gurobi', 
        'Nloop' : 10, # number of active learning loops
        'constant_ucb' : 0, 
        'n_medium_ucb' : 5, 
        'min_medium' : 0,
        'scale': 2, # ratio between intake FLUXES for variable mediums / fixed mediums  
        'mode' : "fold_loop",
        'model' : "iML1515_Paul_EXP"
    },
    "biolog_iML1515": {
        'file' : "./Dataset_model/biolog_iML1515_EXP",
        'solver': 'gurobi', # gurobi require license and gurobipy install
        'Nloop' : 5, # ~20m for each loops
        'constant_ucb' : 0.1, # ratio exploration vs. exploitation recommended 0.1
        'n_medium_ucb' : 5, # nbr of medium created for each y_true mesurements recommended 5
        'min_medium' : 0, # minimum value medium can take
        'scale': 3,
        'mode' : "fold_loop", # to perform fold in each loop or loop in each fold,
        'model' : "biolog_iML1515"
    },
    "biolog_iML1515_mini": {
        'file' : "./Dataset_model/biolog_iML1515_mini",
        'solver': 'gurobi',  # gurobi require license and gurobipy install
        'Nloop' : 10, # ~4m for each loops
        'constant_ucb' : 0.1, # ratio exploration vs. exploitation recommended 0.1
        'n_medium_ucb' : 5, # nbr of medium created for each y_true mesurements recommended 5
        'min_medium' : 0, # minimum value medium can take
        'scale': 3,
        'mode' : "fold_loop", # to perform fold in each loop or loop in each fold
        'model' : "biolog_iML1515"
    },
    "IJN1463_EXP": {
        'file' : "./Dataset_model/IJN1463_EXP",
        'solver': 'glpk', # default in cobrapy
        'Nloop' : 8, 
        'constant_ucb' : 0.1, 
        'n_medium_ucb' : 10, 
        'min_medium' : 0, 
        'scale': 2,
        'mode' : "fold_loop",
        'model' : "IJN1463_EXP"
    }
}


model_input = {
    "iML1515_EXP": {
        'n_models' : 1,
        'xfold': 1, # use everything for training, no testing/validation
        'n_hidden': 1, # nb of hidden layers
        'hidden_dim' : 38, 
        'dropout' : 0.1,
        'epochs' : 20,
        'batch_size' : 5
    },
    "iML1515_Paul_EXP": {
        'n_models' : 1,
        'xfold': 1,
        'n_hidden': 1, # nb of hidden layers
        'hidden_dim' : 23, 
        'dropout' : 0.1,
        'epochs' : 20,
        'batch_size' : 5
    },
    "biolog_iML1515": {
        'n_models' : 5,
        'xfold': 1, 
        'n_hidden': 3, 
        'hidden_dim' : 430, 
        'dropout' : 0.1,
        'epochs' : 20,
        'batch_size' : 16
    },
    "IJN1463_EXP": {
        'n_models' : 1,
        'xfold': 1, # no testing and models validation 
        'n_hidden': 1, 
        'hidden_dim' : 196, 
        'dropout' : 0.1,
        'epochs' : 20,
        'batch_size' : 10
    }
}

def update_parameter(parameter,input, verbose = False):
    """
    Updates the attributes of an object with values from a dictionary using keys as attribute names.

    Parameters:
    parameter (object): The target object to update.
    input (dict): A dictionary containing key-value pairs where the keys represent
                      attribute names of the object, and the values are the new attribute values."""
    
    update_list = ['Nloop','constant_ucb','n_medium_ucb','min_medium', 'scale','mode']

    for key in update_list:
        value = input[key]
        setattr(parameter, key, value)
    
    # choosing solver
    parameter.cobramodel.solver = input["solver"]

    # update model inside active learning loop
    model = model_input[input["model"]]

    for key, value in model.items():
        setattr(parameter, key, value)

    if verbose:
        print_specification(input)
        print_specification(model)
    return parameter

res = {'50': 
       {'iML1515':[],
        'iML1515_Paul':[],
        'iJN1463':[]
       },
       '500':
       {'iML1515':[0.898,0.863,0.879],
        'iML1515_Paul':[0.724,0.737, 0.650],
        'iJN1463':[]
       },
       '5000':
       {'iML1515':[0.903,0.849,0.932],
        'iML1515_Paul':[0.793, 0.766,0.786],
        'iJN1463':[]
       },
}