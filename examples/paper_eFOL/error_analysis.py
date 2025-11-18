import os,time,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import numpy as np
from fol.loss_functions.thermal import ThermalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from utilities import *

# cleaning & logging
working_directory_name = 'nn_output_error_analysis'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(case_dir)
sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_2D.log"))

# create mesh_io
# problem setup
model_settings = {"L":1,"N":52,
                "left":1.0,"right":0.1}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["left"],"right":model_settings["right"]}}

thermal_loss_2d = ThermalLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,"loss_function_exponent":2,
                                                                        "beta":2,"c":4},
                                                                        fe_mesh=fe_mesh)

# create Fourier parametrization/control
x_freqs = np.array([3,5,7])
y_freqs = np.array([2,4,7])
z_freqs = np.array([0])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 10000
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict_efol_paper.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(os.path.join(os.path.dirname(__file__),f'fourier_control_dict_efol_paper.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

DON, FNO, FOL, iFOL, FEM = {}, {}, {}, {}, {}
with open('thermal_deeponet_pred_dict.pkl', 'rb') as f:
    DON = pickle.load(f)

with open('thermal_fno_pred_dict.pkl', 'rb') as f:
    FNO = pickle.load(f)

with open('thermal_fol_siren_pred_dict.pkl', 'rb') as f:
    FOL = pickle.load(f)

with open('thermal_ifol_pred_dict.pkl', 'rb') as f:
    iFOL = pickle.load(f)

with open('thermal_FE_T_dict.pkl', 'rb') as f:
    FEM = pickle.load(f)
# keys for dictionaries: FOL["exFOL_pred"], FNO["FNO_pred"], DNO["DeepONet_pred"]

fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                "maxiter":20,"load_incr":4}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()

FE = {}
fol_abs_error, fno_abs_error, deeponet_abs_error,ifol_abs_error = [],[],[],[]
for eval_id in range(9000,9200):
    FOL_T = FOL["exFOL_pred"][eval_id,:].reshape(-1)
    FNO_T = FNO["FNO_pred"][eval_id,:].reshape(-1)
    DON_T = DON["DeepONet_pred"][eval_id,:].reshape(-1)
    iFOL_T = iFOL["iFOL_pred"][eval_id,:].reshape(-1)
    # FE_T = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes())))

    FE_T = FEM[f'sample_{eval_id}']
    fol_abs_error.append(abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1)).flatten())
    fno_abs_error.append(abs(FNO_T.reshape(-1,1)-FE_T.reshape(-1,1)).flatten())
    deeponet_abs_error.append(abs(DON_T.reshape(-1,1)-FE_T.reshape(-1,1)).flatten())
    ifol_abs_error.append(abs(iFOL_T.reshape(-1,1)-FE_T.reshape(-1,1)).flatten())

# with open(f'thermal_FE_T_dict.pkl', 'wb') as f:
#     pickle.dump(FE,f)
    
fol_abs_err = np.array(fol_abs_error)
fno_abs_err = np.array(fno_abs_error)
deeponet_abs_err = np.array(deeponet_abs_error)
ifol_abs_err = np.array(ifol_abs_error)

fol_mae = np.mean(fol_abs_err,axis=1)
fol_max = np.max(fol_abs_err,axis=1)
fno_mae = np.mean(fno_abs_err,axis=1)
fno_max = np.max(fno_abs_err,axis=1)
deeponet_mae = np.mean(deeponet_abs_err,axis=1)
deeponet_max = np.max(deeponet_abs_err,axis=1)
ifol_mae = np.mean(ifol_abs_err,axis=1)
ifol_max = np.max(ifol_abs_err,axis=1)

model_names = ['eFOL','FNO','DeepONet','iFOL']
mae_errors = [fol_mae, fno_mae, deeponet_mae, ifol_mae]
max_errors = [fol_max, fno_max, deeponet_max, ifol_max]
# Combine all error arrays into one big NumPy array
all_errors = np.concatenate([np.array(mae_errors).flatten(),
                             np.array(max_errors).flatten()])

# Compute global min and max for y-axis
min_axis, max_axis = np.min(all_errors), np.max(all_errors)

mea_T_df = pd.DataFrame({
    model: mea.flatten() for model, mea in zip(model_names, mae_errors)
})
max_T_df = pd.DataFrame({
    model: mx.flatten() for model, mx in zip(model_names, max_errors)
})

fig, axes = plt.subplots(1, 2, figsize=(12,8), sharey=True)

mea_T_df.plot.box(ax=axes[0], showfliers=False, showmeans=True)
max_T_df.plot.box(ax=axes[1], showfliers=False, showmeans=True)

axes[0].set_title("MAE T",fontsize=16)
axes[1].set_title("Max T",fontsize=16)

for ax in axes:
    ax.set_ylabel("Error",fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_yscale("log")
    ax.set_ylim(0.1*min_axis, 10*max_axis)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(f'error_analysis_.png')
