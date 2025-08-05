import os
os.environ["RAY_DISABLE_GPU_AUTODETECT"] = "1"
import pandas as pd
import numpy as np
from cobra import Configuration
from cobra.io import read_sbml_model
from library.parallel import fit_cobra, run_cobra_parallel

Configuration().solver = "gurobi" 
df = pd.read_csv("Dataset_input\iML1515_EXP.csv")
n_sample = 100  

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

complete_media = X.columns.tolist()

n_experiment = len(df)
n_media_components = X.shape[1]

unique_counts = X.nunique(dropna=True)
constant_media = unique_counts[unique_counts <= 1].index.tolist()
n_constant_media = len(constant_media)
variable_media = unique_counts[unique_counts == 2].index.tolist()
n_variable_media = len(variable_media)

X_reduced = X.drop(columns=constant_media , errors="ignore")
print("X_reduced is duplicated ?", X_reduced.duplicated().any())

gem_model = read_sbml_model("Dataset_model/iML1515_EXP.xml")
objective = "BIOMASS_Ec_iML1515_core_75p37M"

def prior_sampling(names, lower = 0, upper = 1, n = 100, seed= 42):
    """
    Sample uniformly n times for each name between lower and upper bounds,
    and return as a DataFrame (columns=names, rows=samples).
    """
    np.random.seed(seed)
    data = {name: np.random.uniform(lower, upper, n) for name in names}
    return data


sample_variable_medium = pd.DataFrame(prior_sampling(variable_media,n = n_sample))

X_rep = pd.DataFrame(
    np.repeat(X.values, repeats=n_sample, axis=0),
    columns=X.columns
)

prior_rep = pd.DataFrame(
    np.tile(sample_variable_medium.values, (X.shape[0], 1)),
    columns=sample_variable_medium.columns
)

X_masked = X_rep.copy()
for col in variable_media:
    X_masked[col] = X_rep[col] * prior_rep[col]

print(f"Generated {X_masked.shape[0]} samples.")
print(f"last sample: {X_masked.iloc[-1,:]}")


# print(f"Generated {samples_df.shape[0]} samples.")
# i = 117
# print(np.array(X)[i,:])    
# y_cobra, warning = fit_cobra(i = i, 
#                 medium_X= np.array(X), 
#                 mediumname = complete_media, 
#                 cobramodel = gem_model, 
#                 gene = [], 
#                 genename = [] , 
#                 objective = objective)

y_cobra, warning = run_cobra_parallel(X = np.array(X_masked), 
                                    cobramodel = gem_model, 
                                    mediumsize = n_media_components, 
                                    mediumname = complete_media, 
                                    fixedmediumvalue = 1,  
                                    genename = [], 
                                    objective = objective)
print(f"Size of result: {len(warning)}")
output_df = pd.DataFrame(warning, columns=complete_media)
output_df['y_cobra'] = y_cobra
output_df.to_csv(f"Dataset_input/generated_iMN1515_{n_sample}.csv", index=False)

