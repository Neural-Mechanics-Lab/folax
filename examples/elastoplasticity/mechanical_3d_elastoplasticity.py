import sys
import os
import shutil
import jax
import numpy as np

from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss3DTetra
from fol.controls.fourier_control import FourierControl
from fol.mesh_input_output.mesh import Mesh
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.solvers.fe_nonlinear_residual_based_solver_with_history_update import FiniteElementNonLinearResidualBasedSolverWithStateUpdate
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import matplotlib.pyplot as plt
import pickle


def main(solve_FE=True, clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_tetra"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')

    # create fe-based loss function
    bc_dict = {"Ux":{"left":0.0},
                "Uy":{"left":0.0,"right":-0.06},
                "Uz":{"left":0.0}}
    
    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)


    material_dict = {"young_modulus": 3.0, "poisson_ratio": 0.3, "iso_hardening_parameter_1": 0.4, "iso_hardening_param_2" :10.0, "yield_limit" :0.2}
    mechanical_loss_3d = ElastoplasticityLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()
    fourier_control.Initialize()
    create_random_coefficients = True
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'fourier_control_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)

    eval_id = 69
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])


    # choose which sample to evaluate

    # classical FE solve (no ML)
    if solve_FE:
        fe_setting = {
            "linear_solver_settings": {
                "solver": "JAX-direct",
                "tol": 1e-6,
                "atol": 1e-6,
                "maxiter": 1000,
                "pre-conditioner": "ilu"
            },
            "nonlinear_solver_settings": {
                "rel_tol": 1e-5,
                "abs_tol": 1e-5,
                "maxiter": 100,
                "load_incr": 10
            }
        }

        nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverWithStateUpdate(
            "nonlinear_fe_solver",
            mechanical_loss_3d,
            fe_setting,
            history_plot_settings={"plot":True,"save_directory":case_dir}
        )
        nonlinear_fe_solver.Initialize()

        # Solve for the chosen K-field and zero initial guess
        load_steps_solutions, load_steps_states, solution_history_dict = nonlinear_fe_solver.Solve(
                        K_matrix[eval_id], np.zeros(3 * fe_mesh.GetNumberOfNodes()),return_all_steps=True)
        
        n_incr = fe_setting["nonlinear_solver_settings"]["load_incr"]
        FE_UVW=load_steps_solutions[n_incr-1,:]
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
        
    # finalize and export mesh data
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Defaults
    solve_FE = True
    clean_dir = False

    main(solve_FE, clean_dir)
