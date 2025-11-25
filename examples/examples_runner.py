import subprocess
import os

# Define the scripts and their arguments
scripts_and_args = [
    ("fourier_feature_mapping_hypernetworks","implicit_learning_mechanical_2D.py", "clean_dir=true"),
    ("sensitivity_analysis","square_mechanical_adj_sa.py", "clean_dir=true"),
    ("implicit_control_solution_decoder_data_driven","meta_implicit_decoder_fourier_fields.py", "clean_dir=true","ifol_num_epochs=10"),
    ("implicit_parametric_learning_square","meta_implicit_pr_lr_mechanical_2D.py", "clean_dir=true","ifol_num_epochs=10"),
    ("implicit_parametric_learning_square","meta_alpha_implicit_pr_lr_mechanical_2D.py", "clean_dir=true","ifol_num_epochs=10"),
    ("thermal_box","thermal_box.py", "clean_dir=true","fol_num_epochs=10"),
    ("thermal_fol","thermal_fol.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_box","mechanical.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_box","mechanical_poly.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_box","neo_hooke_mechanical.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_box","neo_hooke_mechanical_hexa.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_hook","hook.py", "clean_dir=true","fol_num_epochs=10"),
    ("mechanical_square", "mechanical_2D_poly.py", "clean_dir=true", "fol_num_epochs=10"),
    ("mechanical_square", "neo_hooke_mechanical_2D.py", "clean_dir=true", "fol_num_epochs=10"),
    ("MNIST", "mnist_classifier.py", "clean_dir=true", "num_epochs=10")
]

# Run each script with its arguments
for script_dir, script_name, *args in scripts_and_args:
    try:
        # Change the current working directory to the script's directory
        os.chdir(script_dir)
        
        # Build the command to run the script with the provided arguments
        command = ["python3", script_name] + list(args)
        
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the script finished successfully
        if result.returncode == 0:
            print(f"Output from {script_dir}/{script_name}:\n{result.stdout}")
        else:
            print(f"Error from {script_dir}/{script_name}:\n{result.stderr}")
            print(f"Script {script_dir}/{script_name} failed with return code {result.returncode}.")
            break  # Stop executing further scripts if one fails

        # Change back to the original directory
        os.chdir('..')

    except Exception as e:
        print(f"An error occurred while running the script {script_dir}/{script_name}: {e}")
        os.chdir('..')  # Ensure we return to the original directory if there's an error