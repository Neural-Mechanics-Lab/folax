import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *
import requests
import zipfile

def prepare_net_params(case_dir):
        """
        Extract only the contents of 'folder_in_zip' from the ZIP archive
        and place them into 'extract_to'.
        """
        extract_to = case_dir
        folder_in_zip = "2d_hyperelastic/"  # ensure correct format

        url = "https://zenodo.org/records/17752752/files/NiN.zip?download=1"
        filename = "NiN.zip"

        fol_info(f"⬇ Downloading '{filename}' from Zenodo...")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # raises if e.g. 404, 403, etc.

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        with zipfile.ZipFile(filename, "r") as z:
            # Filter the files that start with the folder path
            members = [m for m in z.namelist() if m.startswith(folder_in_zip)]

            if not members:
                raise ValueError(f"Folder '{folder_in_zip}' not found inside ZIP.")

            fol_info(f"📦 Extracting {len(members)} files from '{folder_in_zip}'...")

            for member in members:
                # Compute final extraction path
                destination = os.path.join(extract_to, os.path.relpath(member, folder_in_zip))

                # Create directories if needed
                if member.endswith("/"):
                    os.makedirs(destination, exist_ok=True)
                else:
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    with z.open(member, "r") as src, open(destination, "wb") as dest:
                        dest.write(src.read())

            fol_info(f"✔ Extracted to: {os.path.abspath(extract_to)}")

# directory & save handling
working_directory_name = "2d_hyperelastic"   # should be the same dir that contains network parameters
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":41,
                "Ux_left":0.0,"Ux_right":0.5,
                "Uy_left":0.0,"Uy_right":0.5}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create fe-based loss function
bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

material_dict = {"young_modulus":1,"poisson_ratio":0.3}

mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                    "num_gp":2,
                                                                                    "material_dict":material_dict},
                                                    fe_mesh=fe_mesh)

mechanical_loss_2d.Initialize()

# define a control class which reconstrcut the input space from a reduced space
# identity control maps X: -> X
identity_control = IdentityControl('identity_control', control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()


# define fourier control to create synthethic microstructures
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
fourier_control.Initialize()

# load fourier coefficients and compute K
with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


# now we need to create, initialize and train ifol
ifol_settings_dict = {
    "characteristic_length": 64,
    "synthesizer_depth": 4,
    "activation_settings":{"type":"sin",
                            "prediction_gain":30,
                            "initialization_gain":1.0},
    "skip_connections_settings": {"active":False,"frequency":1},
    "latent_size":  8*64,
    "modulator_bias": False,
    "main_loop_transform": 1e-5,
    "latent_step_optimizer": 1e-4,
    "ifol_nn_latent_step_size": 1e-2
}

# design synthesizer & modulator NN for hypernetwork
characteristic_length = ifol_settings_dict["characteristic_length"]
synthesizer_nn = MLP(name="synthesizer_nn",
                    input_size=3,
                    output_size=2,
                    hidden_layers=[characteristic_length] * ifol_settings_dict["synthesizer_depth"],
                    activation_settings=ifol_settings_dict["activation_settings"],
                    skip_connections_settings=ifol_settings_dict["skip_connections_settings"])

latent_size = ifol_settings_dict["latent_size"]
modulator_nn = MLP(name="modulator_nn",
                input_size=latent_size,
                use_bias=ifol_settings_dict["modulator_bias"]) 

hyper_network = HyperNetwork(name="hyper_nn",
                            modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                            coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
#learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(ifol_settings_dict["main_loop_transform"]))
latent_step_optimizer = optax.chain(optax.adam(ifol_settings_dict["latent_step_optimizer"]))

# create fol
ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=fourier_control,
                                                        loss_function=mechanical_loss_2d,
                                                        flax_neural_network=hyper_network,
                                                        main_loop_optax_optimizer=main_loop_transform,
                                                        latent_step_optax_optimizer=latent_step_optimizer,
                                                        latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                        num_latent_iterations=3)
ifol.Initialize()

# split the data to train and test sets
otf_id = 0
train_set_otf = coeffs_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

train_start_id = 20
train_end_id = 8000
train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]     # for parametric training

test_start_id = 0
test_end_id = 20
test_set_pr = coeffs_matrix[test_start_id:test_end_id,:]

# OTF or Parametric 
parametric_learning = True
if parametric_learning:
    train_set = train_set_pr
    test_set = test_set_pr
    tests = range(test_start_id,test_end_id)
else:
    train_set = train_set_otf   
    test_set = train_set
    tests = [otf_id]
# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
train_settings_dict = {"batch_size": 1,
                        "num_epoch":5000,
                        "parametric_learning": parametric_learning,
                        "OTF_id": otf_id,
                        "train_start_id": train_start_id,
                        "train_end_id": train_end_id,
                        "test_start_id": test_start_id,
                        "test_end_id": test_end_id}

train_from_scratch = False
if train_from_scratch:
    ifol.Train(train_set=(train_set,),
                test_set=(test_set,),
                test_frequency=100,
                batch_size=train_settings_dict["batch_size"],
                convergence_settings={"num_epochs":train_settings_dict["num_epoch"],"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)
else:
    # download the params if needed 
    prepare_net_params(case_dir)
    # load the best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

for eval_id in tests:
    # predict the result from ifol
    ifol_uvw = np.array(ifol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'iFOL_U_{eval_id}'] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
    fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
    iFOL_stress = compute_stress_neohooke_quad(loss_function=mechanical_loss_2d, disp_field_vec=jnp.array(ifol_uvw), K_matrix=jnp.array(K_matrix[eval_id,:]))

    # solve FE here to compare the result
    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":1e-7,
                                                "maxiter":8,"load_incr":51}}
    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
    nonlin_fe_solver.Initialize()
    FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  

    fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))

    abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
    fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))

    FE_stress = compute_stress_neohooke_quad(loss_function=mechanical_loss_2d, disp_field_vec=jnp.array(FE_UVW), K_matrix=jnp.array(K_matrix[eval_id,:]))
    stress_error = abs(iFOL_stress.reshape(-1) - FE_stress.reshape(-1))
    fe_mesh[f"FE_FirstPiola_{eval_id}"] = FE_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f"iFOL_FirstPiola_{eval_id}"] = iFOL_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f"error_FirstPiola_{eval_id}"] = stress_error.reshape((fe_mesh.GetNumberOfNodes(), 3))


    # solve the Newton-Raphson initialized by ifol in one load increment
    fol_info("solve fe hybrid in one load step")
    nin_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                "maxiter":10,"load_incr":1}}
    nin_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nin_nonlin_fe_solver",mechanical_loss_2d,nin_setting)
    nin_nonlin_fe_solver.Initialize()
    try:    
        NiN_UVW = np.array(nin_nonlin_fe_solver.Solve(K_matrix[eval_id,:],ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes())))  
    except Exception as e:
        fol_info(f"Error occured {type(e).__name__}: e")
        NiN_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())

    fe_mesh[f'NiN_U_{eval_id}'] = NiN_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
    fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))

    # plot the result
    plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes()), hfe_sol_field=NiN_UVW,
                err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"/ifol_fe-nin_error_{eval_id}",
                fig_titles=['Elasticity Morph.','iFOL','FE-NIN','iFOL-FE Abs Diff.'])

# export the result in a .vtk file
fe_mesh.Finalize(export_dir=case_dir)
