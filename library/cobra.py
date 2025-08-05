from library.imports import *
from library.utils import read_csv
import cobra
import numpy as np


###############################################################################
# Functions for Class
###############################################################################

def get_objective(model):
    """
    Get the reaction carring the objective, is there a cleaner way ???
    """
    r = str(model.objective.expression)
    r = r.split()
    r = r[0].split("*")
    obj_id = r[1]
    # line below crashes if obj_id does not exist
    r = model.reactions.get_by_id(obj_id)
    return obj_id


def check_warning(self, trainingfile, cobraname, mediumname, mediumsize):
    """
    Use inside class TrainingSet to check if all the files and links are valid
    """
    if trainingfile != "":
        self.load(trainingfile)
        return 0
    if cobraname == "":
        return 0 # create an empty object
    if not os.path.isfile(cobraname + ".xml"):
        print(cobraname)
        sys.exit("xml cobra file not found")
    if not os.path.isfile(mediumname + ".csv"):
        print(mediumname)
        sys.exit("medium or experimental file not found")
    if mediumsize < 1:
        sys.exit("must indicate medium size with experimental dataset")
        

def find_medium(self, mediumname, mediumsize, fixedmediumsize):
    """
    Use inside class TrainingSet to import experiemental data and split into medium (X), gene, y, mask
    """
    # set medium
    H, M = read_csv(mediumname)
    medium = []
    genename = []
    for i in range(mediumsize):
        medium.append(H[i])
    if M.shape[1] > (mediumsize + 1):
        self.gene = M[:, mediumsize : -1]
        for i in range(mediumsize,(M.shape[1]-1)):
            genename.append(H[i])
            self.genename = genename
    else: self.genename, self.gene = [],[]
    self.medium = medium
    self.X = M[:, 0 : -1]
    self.y_true = M[:, -1]
    self.y = M[:, -1]
    self.size = self.y_true.shape[0]
#    self.mask = M[:,fixedmediumsize:mediumsize]
    self.mask = M[:,0:mediumsize]

    return self

def get_input(self, input):
    self.cobraname = input['cobrapath']  # model cobra file
    self.mediumname = input['mediumpath']  # medium file
    self.mediumbound = input['mediumbound']  # EB or UB
    self.mediumsize = input['mediumsize']
    self.fixedmediumsize = input['fixedmediumsize']
    self.fixedmediumvalue = input['fixedmediumvalue']
    self.method = input['method']
    self.regression = input['regression']
    self.threshold = input['threshold']
    self.parallel = input['parallel']
    self.cobramodel = cobra.io.read_sbml_model(self.cobraname + ".xml")
    find_medium(self, self.mediumname, self.mediumsize, self.fixedmediumsize)
    # set objectve
    self.objective = (get_objective(self.cobramodel))


###############################################################################
# Cobra class
###############################################################################
class TrainingSet:
    """
    Callable class with all element necessary to run Cobra
    """
    def __init__(
        self,
        trainingfile = "",
        cobraname = "",
        mediumname = "",
        mediumbound="EB",
        mediumsize=-1,
        fixedmediumsize=-1,
        fixedmediumvalue=0,
        objective=[],
        method="FBA",
        regression = True,
        threshold = None,
        parallel = False,
        input = None,
        verbose=False,
    ):
        if input != None:
            get_input(self, input)
        else:
            if check_warning(self, trainingfile, cobraname, mediumname, mediumsize) == 0:
                return
            self.cobraname = cobraname  # model cobra file
            self.mediumname = mediumname  # medium file
            self.mediumbound = mediumbound  # EB or UB
            self.mediumsize = mediumsize
            self.fixedmediumsize = fixedmediumsize
            self.fixedmediumvalue = fixedmediumvalue
            self.method = method
            self.regression = regression
            self.threshold = threshold
            self.parallel = parallel
            self.cobramodel = cobra.io.read_sbml_model(cobraname + ".xml")
            find_medium(self, mediumname, mediumsize, fixedmediumsize)
            # set objectve
            self.objective = (
                [get_objective(self.cobramodel)] if objective == [] else objective
            )


    def save(self, filename, X, y_true, y_cobra, verbose=False):
        """
        save cobra model in xml and parameter in npz (compressed npy)
        save parameters
        """
        np.savez_compressed(
            filename,
            cobraname=filename,
            mediumname=self.mediumname,
            mediumbound=self.mediumbound,
            mediumsize=self.mediumsize,
            fixedmediumsize=self.fixedmediumsize,
            fixedmediumvalue=self.fixedmediumvalue,
            objective=self.objective,
            method=self.method,
            size=self.size,
            medium=self.medium,
            X=X,
            gene=self.gene,
            genename=self.genename,
            mask = self.mask,
            regression = self.regression,
            threshold = self.threshold,
            parallel = self.parallel,
            y_true=y_true,
            y_cobra=y_cobra,
            y=self.y
            )
        # save cobra model
        cobra.io.write_sbml_model(self.cobramodel, filename + ".xml")


    def load(self, filename):
        """
        load parameters (compressed npy)
        """
        if not os.path.isfile(filename + ".npz"):
            print(f"{filename}.npz")
            sys.exit("file not found")
        loaded = np.load(filename + ".npz", allow_pickle=True)
        self.cobraname = str(loaded["cobraname"])
        self.mediumname = str(loaded["mediumname"])
        self.mediumbound = str(loaded["mediumbound"])
        self.mediumsize = loaded["mediumsize"]
        self.fixedmediumsize = loaded["fixedmediumsize"]
        self.fixedmediumvalue = loaded["fixedmediumvalue"]
        self.objective = loaded["objective"]
        self.method = str(loaded["method"])
        self.size = loaded["size"]
        self.medium = loaded["medium"]
        self.X = loaded["X"]
        self.gene = loaded["gene"]
        self.genename = loaded["genename"]
        self.regression = loaded["regression"]
        self.threshold = loaded["threshold"]
        self.parallel = loaded["parallel"]
        self.mask = loaded["mask"]
        self.y_true = loaded["y_true"]
        self.y_cobra = loaded["y_cobra"]
        self.y = loaded["y"]
        self.cobramodel = cobra.io.read_sbml_model(self.cobraname + ".xml")

    def printout(self, filename=""):
        """
        print out cobra model informations
        """
        if filename != "":
            sys.stdout = open(filename, "wb")
        print(f"model file name: {self.cobraname}")
        print(f"medium file name: {self.mediumname}")
        print(f"medium bound: {self.mediumbound}")
        print(f"mediumsize: {self.mediumsize}")
        print(f"fixed medium size: {self.fixedmediumsize}")
        print(f"fixed medium value: {self.fixedmediumvalue}")
        print(f"list of reactions in objective: {self.objective}")
        print(f"method: {self.method}")
        print(f"trainingsize: {self.size}")
        print(f"Training set X: {self.X.shape}")
        print(f"Gene set: {self.gene.shape}")
        print(f"Regression : {self.regression}")
        print(f"Parallel : {self.parallel}")
        print(f"Training set y_true: {self.y_true.shape}")
        print(f"Training set y_cobra: {self.y_cobra.shape}")
        if filename != "":
            sys.stdout.close()

###############################################################################
# Function for modelling with cobra
###############################################################################
def scaling(X,mediumsize,fixedmediumvalue):
    """
    Transform X from experiemental concentration to 
    upper bound intake flux for better fitting with cobra
    """
    if isinstance(X,np.ndarray) != True:
        X = np.float32(X.to_numpy())
#    X = X[:,:mediumsize]
    medium_X = X*fixedmediumvalue
#    medium_X[medium_X < 1.0e-4] = 0
    return medium_X


def knock_out(model,i,gene,genename):
    """
    Find the reactions with 0 fluxes and
    eliminate them in cobra model
    """
    if isinstance(genename,np.ndarray) != True:
        genename = np.array(genename)
    KO = genename[gene[i,]==0]
    for react in KO:
        model.reactions.get_by_id(react).knock_out()
    return model 


def knock_out_rebound(model,i,gene,genename, esp = 1e-1):
    """
    Find the reactions with 0 fluxes (KO) and
    rebound them in cobra model, used to set a very small bound for KO reactions
    """
    if isinstance(genename,np.ndarray) != True:
        genename = np.array(genename)
    KO = genename[gene[i,]==0]
    bound = (0,esp)
    for react in KO:
        model.reactions.get_by_id(react).bounds = bound
    return model


def set_medium(cobramodel,medium):
    """
    Update the medium of a CobraModel with the provided medium dictionary.
    If a medium component exists in both the original CobraModel's medium and the provided
    medium dictionary, its value will be replaced with the value from the dictionary.
    If a medium component is not present in the original medium, it will be added.

    Parameters:
        cobramodel: The CobraModel to update.
        medium (dict): A dictionary of medium components and their values.

    Returns:
        cobramodel: The updated Cobramodel with the new medium components.
    """
    medium_ori = cobramodel.medium
    medium_ori = {key: 0 for key in medium_ori}

    for key, value in medium.items():
        medium_ori[key] = value

    cobramodel.medium = medium_ori

    return cobramodel

def set_medium_strict(cobramodel,medium):
    """
    Update the medium of a CobraModel with the provided medium dictionary.
    set reaction upper bound = lower bound = fluxes-in

    Parameters:
        cobramodel: The CobraModel to update.
        medium (dict): A dictionary of medium components and their values.

    Returns:
        cobramodel: The updated Cobramodel with the new medium components.
    """
    for key, value in medium.items():
        bound = (value,value)
        cobramodel.reactions.get_by_id(key).bounds = bound

    return cobramodel
