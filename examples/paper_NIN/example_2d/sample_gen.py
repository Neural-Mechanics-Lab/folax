import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from mechanical2d_utilities import *

### Script's goal:
####### the following script is to 
####### create and plot tpms, circular inclusion, and fiber samples and save the pkl dictionary file
####### beside a txt file including all K_matrix so that one can have access to K_matrix.txt file

def main(solve_FE=False,ifol_num_epochs=10,clean_dir=False):
    # directory & save handling
    working_directory_name = "sample_gen_2D_tpms_81"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":81,
                    "Ux_left":0.0,"Ux_right":0.1,
                    "Uy_left":0.0,"Uy_right":0.1}

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
    
    
    
    # define settings to create K_matrix samples
    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1, "section_axis_value": 0.5,
                     "constant": 0., "threshold": 0.5, "coefficients":(2.,2.,2.)}
    
    sphere_settings = {"sphere_diameter": 0.2, "fiber_length": 0.4, "fiber_radius": 0.05,
                        "max": 1., "min": 0.1, "num_spheres": 40}

    # create a dictionary to save different topologies and the corresponding settings
    function_dict = {"tpms_diamond":(create_tpms_diamond, tpms_settings),
                      "tpms_gyroid":(create_tpms_gyroid, tpms_settings), 
                      "tpms_lindinoid":(create_tpms_lidinoid, tpms_settings), 
                      "tpms_schwarz_p":(create_tpms_schwarz_P, tpms_settings), 
                      "tpms_split_p":(create_tpms_split_p, tpms_settings), 
                      "sphere_lattice":(create_sphere_lattice, sphere_settings), 
                      "periodic_sphere":(create_random_periodic_sphere_field,sphere_settings), 
                      "periodic_fiber":(create_random_fiber_field, sphere_settings)}
    
    # save and plot?
    plot = input("Do you want to save plots? (Y/N): ").strip().lower()
    if plot == "y":
        plot = True
    elif plot == 'n':
        plot = False
    # save topologies as binary file using pickle
    section_axis_values = (0., 0.2, 0.4, 0.6, 0.8, 1.)
    sample_info_dict = {}
    file_path = os.path.join(working_directory_name,f"ifol_tpms_test_samples_res_{model_settings['N']}.pkl")

    print(f"You are about to overwrite the file: {file_path}")
    confirm = input("Do you want to proceed? (Y/N): ").strip().lower()

    if confirm == "y":
        with open (file_path, 'wb') as f:
            for function_name, dict_tuple in function_dict.items():
                sample_info_dict[function_name] = {}
                sample_info_dict[function_name]['settings'] = dict_tuple[1]
                print(f"Settings associate with '{function_name}' stored successfully.")
                
                sample_info_dict[function_name]['K_matrix'] = {}
                for axis_value in section_axis_values:
                    dict_tuple[1].update({"section_axis_value":axis_value})
                    K_matrix = dict_tuple[0](fe_mesh,dict_tuple[1])
                    sample_info_dict[function_name]['K_matrix'][str(axis_value)] = K_matrix
                    print(f"\t The corresponding topology successfully stored at z axis = {axis_value}.")
            
                if plot:
                    file_name = os.path.join(case_dir,function_name+".png")
                    plot_tpms_2d(
                    tpms_settings=dict_tuple[1],
                    model_settings=model_settings,
                    tpms_fn=dict_tuple[0],
                    fe_mesh=fe_mesh,
                    file_name=file_name,
                    section_values=(0., 0.2, 0.4, 0.6, 0.8, 1.)
                    )
            pickle.dump(sample_info_dict,f)
            print("Dictionary saved successfully!")
    elif confirm == "n":
        with open(file_path, 'rb') as f:
            sample_info_dict = pickle.load(f)
        print(f"The file {file_path} loaded and stored in the variable sample_info_dict!")
    else:
        print("Invalid input. Please enter Y or N.")
    
    tpms_K = tpms_to_K_matrix(sample_info_dict)
    
    K_matrix_file = os.path.join(case_dir, f"K_matrix_tpms_res_{model_settings['N']}.txt")
    np.savetxt(K_matrix_file, tpms_K)


    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 2000
    solve_FE = True
    clean_dir=False
    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                fol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("fol_num_epochs should be an integer.")
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
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(ifol_num_epochs, solve_FE,clean_dir)