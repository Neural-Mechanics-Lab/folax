import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.dirichlet_control import DirichletControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from mechanical3d_utilities import *

def main(ifol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "ifol_output_tpms_NH_AD"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # #call the function to create the mesh
    # fe_mesh = create_3D_box_mesh(Nx=11,Ny=11,Nz=11,Lx=1.,Ly=1.,Lz=1.,case_dir=case_dir)
    # fe_mesh.Initialize()

    # tpms geometry from a mesh file
    fe_mesh = Mesh("fol_io","cylindrical_fine_all_sides.med",os.path.join(os.path.abspath(__file__),'../../../meshes'))
    fe_mesh.Initialize()

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.2},
                "Uy":{"left":0.0,"right":0.0},
                "Uz":{"left":0.0,"right":0.0}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    loss_settings = {"dirichlet_bc_dict":bc_dict,"parametric_boundary_learning":True,"material_dict":material_dict}
    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,
                                                                                   fe_mesh=fe_mesh)
    mechanical_loss_3d.Initialize()


    # dirichlet control
    dirichlet_control_settings = {"learning_boundary":{"Ux":{'right'},"Uy":{"right"}}}
    dirichlet_control = DirichletControl(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                         fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)
    dirichlet_control.Initialize()
    print(dirichlet_control.num_control_vars)

    # solve FE here
    # shear BCs
    updated_bc_shear = bc_dict.copy()
    updated_bc_shear.update({"Ux":{"left":0.,"right":0.},
                        "Uy":{"left":0.,"right":0.2},
                        "Uz":{"left":0.,"right":0.}})
    
    # Compression BCs
    updated_bc_compression = bc_dict.copy()
    updated_bc_compression.update({"Ux":{"left":0.,"right":-0.2},
                        "Uy":{"left":0.,"right":0.},
                        "Uz":{"left":0.,"right":0.}})
    
    updated_bc_tension = bc_dict.copy()
    updated_bc_tension.update({"Ux":{"left":0.,"right":0.2},
                        "Uy":{"left":0.,"right":0.0},
                        "Uz":{"left":0.,"right":0.0}})


    updated_loss_setting = loss_settings.copy()
    updated_loss_setting.update({"dirichlet_bc_dict":updated_bc_tension})
    mechanical_loss_3d_updated = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=updated_loss_setting,
                                                                            fe_mesh=fe_mesh)
    mechanical_loss_3d_updated.Initialize()
    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                            "maxiter":8,"load_incr":21},
                "output_directory":case_dir}
    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d_updated,fe_setting)
    nonlin_fe_solver.Initialize()
    FE_UVW = np.array(nonlin_fe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),np.zeros(3*fe_mesh.GetNumberOfNodes())))
    plot_norm_iter(data=np.loadtxt(case_dir+"/res_norm_jax.txt"),plot_name=case_dir+'/res_norm_iter_FE_AD',type='1')
    
    fe_mesh[f'U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    FE_stress = get_stress(loss_function=mechanical_loss_3d_updated, disp_field_vec=FE_UVW, K_matrix=np.ones(fe_mesh.GetNumberOfNodes()))
    fe_mesh[f'P_FE'] = FE_stress.reshape((fe_mesh.GetNumberOfNodes(), 6))

    fe_mesh.Finalize(export_dir=case_dir,export_format='vtu')

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 100
    solve_FE = True
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("ifol_num_epochs="):
            try:
                ifol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("ifol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE = value.lower() == 'true'
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python thermal_fol.py ifol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)
    
    # Call the main function with the parsed values
    main(ifol_num_epochs, solve_FE,clean_dir)