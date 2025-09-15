import pandas as pd
print(pd.__file__)
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
from cobra import Configuration
from cobra.io import read_sbml_model
from library.parallel import fit_cobra
Configuration().solver = "glpk"
import jax.numpy as jnp
import bayesflow as bf
import matplotlib.pyplot as plt


size = 1
epochs = 50
batch_size = 128

df = pd.read_csv(f"Dataset_input\generated_iMN1515_{size}.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

complete_media = X.columns.tolist()
n_media_components = X.shape[1]

unique_counts = X.nunique(dropna=True)
constant_media = unique_counts[unique_counts <= 1].index.tolist()
n_constant_media = len(constant_media)
variable_media = unique_counts[unique_counts == 2].index.tolist()
n_variable_media = len(variable_media)


gem_model = read_sbml_model("Dataset_model/iML1515_EXP.xml")
objective = "BIOMASS_Ec_iML1515_core_75p37M"

def prior_sampling(names, lower = 0, upper = 1, n = 100, seed= None):
    """
    Sample uniformly n times for each name between lower and upper bounds,
    and return as a DataFrame (columns=names, rows=samples).
    """
    np.random.seed(seed)
    data = {name: np.random.uniform(lower, upper, n) for name in names}
    return data

def solver_fba(**kwargs):

    full_medium = {col: 1 for col in constant_media}
    for k, v in kwargs.items():
        full_medium[k] = v.item()
    try:
        y_cobra, warning = fit_cobra(
            i=0,
            medium_X=np.array([list(full_medium.values())]),
            mediumname=complete_media,
            cobramodel=gem_model,
            gene=[],
            genename=[],
            objective=objective
        )

        if y_cobra is None or np.isnan(y_cobra):
            y_cobra = np.nan
            warning = {f"design_{k}": np.nan for k in kwargs}
        else:
            warning = {
                f"design_{k}": 0 if warning[i] == 0 else 1
                for i, k in enumerate(complete_media)
                if k in kwargs
            }
    except Exception:
        y_cobra = np.nan
        warning = {f"design_{k}": np.nan for k in kwargs}
    
    return {"y_cobra": y_cobra, **warning}

sample = prior_sampling(variable_media, n=1)
result = solver_fba(**sample)
# print(f"Sample: {sample}")
# print(f"Result: {result}")

training_variables = variable_media + [f"design_{name}" for name in variable_media] + ["y_cobra"]

simulator = bf.simulators.make_simulator([prior_sampling, solver_fba])
adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(variable_media, into="inference_variables")
    .concatenate([f"design_{name}" for name in variable_media], into="inference_conditions")
    # .concatenate(["y_cobra"], into="summary_variables")
)

workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=bf.networks.CouplingFlow()
    )

df_reduced = df.drop(columns=constant_media, errors="ignore")

for col in variable_media:
    design_col = f"design_{col}"
    df_reduced[design_col] = df_reduced[col].apply(lambda x: 0 if x == 0 else (1 if 0 < x < 1 else x))


df_reduced = df_reduced[training_variables].dropna()
n_experiment = len(df_reduced)
training_data = {
    k: np.array(v).reshape(-1, 1) if np.array(v).ndim == 1 else np.array(v)
    for k, v in df_reduced.to_dict(orient="list").items()
}

history = workflow.fit_offline(
        training_data,  # Train data is a dictionary with values as 2 dimensions arrays 
        epochs=epochs,
        batch_size=batch_size,
        inference_variables=variable_media,
    )
f = bf.diagnostics.plots.loss(history)
# plt.show()

# model_path = f"Models/BED_model_{size}.h5"
# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# workflow.approximator.save(model_path, save_format="keras_v3")

# print(f"Model saved to {model_path}")
#############################
# Estimate log probability of the designs
exp_data = pd.read_csv("Dataset_input/iML1515_EXP.csv")
exp_X = exp_data.iloc[:, :-1]
exp_Y = exp_data.iloc[:, -1]

design_list = exp_X[variable_media].drop_duplicates().reset_index(drop=True)
n_designs = len(design_list)
n_sample_MC = 500
prior_sample = prior_sampling(variable_media, n=n_sample_MC)

def apply_design_mask(prior_sample: dict, design: pd.Series) -> dict:
    """
    Multiply each variable_media column in prior_sample by the corresponding value in design.
    Returns a new dict with the same keys and masked values.
    """
    masked = {}
    for col in prior_sample:
        res = np.array(prior_sample[col]) * design[col]
        masked[col] = res.reshape(-1, 1)
    return masked

mean_list = []
std_list = []

for i in range(n_designs):
    design = design_list.iloc[i]
    masked_prior = apply_design_mask(prior_sample, design)
    design_dict = {f"design_{col}": np.repeat(design[col], n_sample_MC).reshape(-1, 1) for col in variable_media}

    masked_prior.update(design_dict)
    log_prob = workflow.log_prob(data=masked_prior)
    print(f"Design {i+1}/{n_designs}: {np.mean(log_prob)} , {np.std(log_prob)}")
    mean_list.append(jnp.mean(log_prob))
    std_list.append(jnp.std(log_prob))
    





