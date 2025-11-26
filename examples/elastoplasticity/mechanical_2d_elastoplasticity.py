import sys
import os
import shutil
import jax
import numpy as np

from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.solvers.fe_nonlinear_residual_based_solver_with_history_update import FiniteElementNonLinearResidualBasedSolverWithStateUpdate
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import matplotlib.pyplot as plt
import pickle, time



def main(solve_FE=True, clean_dir=False):
    # directory & save handling
    working_directory_name = 'mechanical_2d_elastoplasticity'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir, working_directory_name + ".log"))

    # problem setup
    model_settings = {
        "L": 1, "N": 40,
        "Ux_left": 0.0, "Ux_right": 0.08
        ,
        "Uy_left": 0.0, "Uy_right": 0.08
    }

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"], N=model_settings["N"])

    # create FE-based loss function (Dirichlet BCs and material params for stress viz)
    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]}
    }
    material_dict = {"young_modulus": 3.0, "poisson_ratio": 0.3, "iso_hardening_parameter_1": 0.4, "iso_hardening_param_2" :10.0, "yield_limit" :0.2}

    mechanical_loss_2d = ElastoplasticityLoss2DQuad(
        "mechanical_loss_2d",
        loss_settings={
            "dirichlet_bc_dict": bc_dict,
            "num_gp": 2,
            "material_dict": material_dict
        },
        fe_mesh=fe_mesh
    )

    with open(f'voroni_control_dict.pkl', 'rb') as f:
        voronoi_control_settings = pickle.load(f)
    voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    voronoi_control.Initialize()

    coeffs_matrix = voronoi_control_settings["coeffs_matrix"]
    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)

    # specify id of the K of interest
    eval_id = 25

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
                "load_incr": 100
            }
        }

        nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverWithStateUpdate(
            "nonlinear_fe_solver",
            mechanical_loss_2d,
            fe_setting,
            history_plot_settings={"plot":True,"save_directory":case_dir}
        )
        nonlinear_fe_solver.Initialize()


        # Solve for the chosen K-field and zero initial guess
        load_steps_solutions, load_steps_states, solution_history_dict = nonlinear_fe_solver.Solve(
                        K_matrix[eval_id], np.zeros(2 * fe_mesh.GetNumberOfNodes()),return_all_steps=True)

        # @Rishabh Please adjust the rest accordingly from here

        exit()

        # 3) Scatter-add to global nodes and average at shared nodes
        nelem = fe_mesh.GetNumberOfElements(mechanical_loss_2d.element_type)
        nnod  = fe_mesh.GetNumberOfNodes()
        conn  = fe_mesh.GetElementsNodes(mechanical_loss_2d.element_type)  
        gp_points, gp_weights = mechanical_loss_2d.fe_element.GaussIntegration2()  
        H = jnp.stack([mechanical_loss_2d.fe_element.ShapeFunctionsValues(p) for p in gp_points], axis=0)  # (4,4)

        def extrapolate_gp_to_nodes_vectorized(gp_data, H_matrix, conn_jax, nnod):
            """
                Vectorized extrapolation from Gauss points to nodes 
                
                Args:
                    gp_data: (nelem, ngp) array - values at Gauss points
                    H_matrix: (ngp, nnodes_per_elem) array - shape functions at Gauss points
                    conn_jax: (nelem, nnodes_per_elem) array - connectivity
                    nnod: int - total number of nodes
                
                Returns:
                    (nnod,) array - nodal values averaged at shared nodes
            """
            
            # Solve H.T @ nodal = gp for each element (avoids computing inverse)
            # Vectorized over all elements using vmap
            def solve_element(gp_elem):
                return jnp.linalg.solve(H_matrix.T, gp_elem)
            
            elem_nodal = jax.vmap(solve_element)(gp_data)  # (nelem, nnodes_per_elem)
            
            # Vectorized scatter-add using segment_sum (more efficient than .at[].add())
            node_indices = conn_jax.flatten()  # (nelem * nnodes_per_elem,)
            values = elem_nodal.flatten()      # (nelem * nnodes_per_elem,)
            
            # Accumulate contributions to each node
            nodal_sum = jax.ops.segment_sum(values, node_indices, num_segments=nnod)
            nodal_count = jnp.bincount(node_indices, length=nnod)
            
            return nodal_sum / jnp.maximum(nodal_count, 1.0)
        
        def extrapolate_gp_to_nodes_vectorized_tensor(gp_data, H_matrix, conn_jax, nnod):
            """
            Vectorized extrapolation for tensor quantities (e.g., strain, stress).
            
            Args:
                gp_data: (nelem, ngp, ...) array - tensor values at Gauss points
                H_matrix: (ngp, nnodes_per_elem) array
                conn_jax: (nelem, nnodes_per_elem) array
                nnod: int
            
            Returns:
                (nnod, ...) array - nodal tensor values
            """
            nelem, ngp = gp_data.shape[:2]
            extra_dims = gp_data.shape[2:]  # e.g., (2, 2) for strain
            nnodes_per_elem = H_matrix.shape[1]
            
            def solve_element(gp_elem):
                # gp_elem: (ngp, *extra_dims) -> (nnodes_per_elem, *extra_dims)
                flat_shape = (ngp, -1)
                gp_flat = gp_elem.reshape(flat_shape)  # (ngp, prod(extra_dims))
                nodal_flat = jnp.linalg.solve(H_matrix.T, gp_flat)  # (nnodes_per_elem, prod(extra_dims))
                return nodal_flat.reshape(nnodes_per_elem, *extra_dims)
            
            elem_nodal = jax.vmap(solve_element)(gp_data)  # (nelem, nnodes_per_elem, *extra_dims)
            
            # Flatten for scatter-add
            elem_nodal_flat = elem_nodal.reshape(-1, *extra_dims)  # (nelem*nnodes_per_elem, *extra_dims)
            node_indices = conn_jax.flatten()
            
            # Accumulate using segment_sum with extra dimensions
            nodal_sum = jax.ops.segment_sum(elem_nodal_flat, node_indices, num_segments=nnod)
            nodal_count = jnp.bincount(node_indices, length=nnod)
            
            # Broadcast count for division
            count_shape = (nnod,) + (1,) * len(extra_dims)
            nodal_count = nodal_count.reshape(count_shape)
            
            return nodal_sum / jnp.maximum(nodal_count, 1.0)
        
        def plot_mesh_res_1(vectors_list:list, file_name:str="plot",loss_settings:dict={}):
            fontsize = 16
            fig, axs = plt.subplots(3, 3, figsize=(20, 12))  # Adjusted to 4 columns

            # Plot the first entity in the first row
            data = vectors_list[0]
            N = int((data.reshape(-1, 1).shape[0]) ** 0.5)
            im = axs[0, 0].imshow(data.reshape(N, N), cmap='viridis', aspect='equal')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 0].set_title('Elasticity Morph.', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            # Plot the same entity with mesh grid in the first row, second column
            im = axs[0, 1].imshow(data.reshape(N, N), cmap='bone', aspect='equal',extent=[0, N, 0, N])
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 1].set_xticklabels([])  # Remove text on x-axis
            axs[0, 1].set_yticklabels([])  # Remove text on y-axis
            axs[0, 1].set_title(f'Mesh Grid: {N} x {N}', fontsize=fontsize)
            axs[0, 1].grid(True, color='red', linestyle='-', linewidth=1)  # Adding solid grid lines with red color
            axs[0, 1].xaxis.grid(True)
            axs[0, 1].yaxis.grid(True)

            x_ticks = np.linspace(0, N, N)
            y_ticks = np.linspace(0, N, N)
            axs[0, 1].set_xticks(x_ticks)
            axs[0, 1].set_yticks(y_ticks)

            cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            # Plot the fourth entity in the second row
            data = vectors_list[1][::2]
            im = axs[0, 2].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal',origin='lower')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            axs[0, 2].set_title(f'U FEM', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            # Plot the fourth entity in the second row
            data = vectors_list[1][1::2]
            im = axs[1,0].imshow(
            data.reshape(N, N),        # or see note below re: transpose
            cmap='coolwarm',
            aspect='equal',
            origin='lower',            # <-- key line
            )
            axs[1,0].set_xticks([])
            axs[1,0].set_yticks([])
            axs[1,0].set_title(f'V FEM', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            data = vectors_list[0]
            L = 1
            N = int((data.reshape(-1, 1).shape[0])**0.5)
            nu = loss_settings["poisson_ratio"]
            e = loss_settings["young_modulus"]

            # PLANE STRAIN constitutive constants (not plane stress!)
            factor = e / ((1 + nu) * (1 - 2*nu))
            c11 = factor * (1 - nu)
            c12 = factor * nu

            dx = L / (N - 1)
            # Get the data
            data = vectors_list[1]
            U_fem = data[::2]
            V_fem = data[1::2]
            coords = vectors_list[3]
            plastic_strain_nodal = vectors_list[2]

            # 1) Build a consistent node ordering by (y, x)
            x = coords[:, 0]
            y = coords[:, 1]
            idx = np.lexsort((x, y))

            # 2) Apply the SAME idx to all nodal fields
            U_sorted = U_fem[idx]
            V_sorted = V_fem[idx]
            exx_plastic_sorted = plastic_strain_nodal[idx, 0, 0]
            eyy_plastic_sorted = plastic_strain_nodal[idx, 1, 1]
            domain_map_sorted = vectors_list[0][idx]

            # 3) Build grid sizes from unique coordinates
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            Nx_grid = len(x_unique)
            Ny_grid = len(y_unique)

            # 4) Reshape EVERYTHING consistently as (Ny, Nx)
            U_grid = U_sorted.reshape(Ny_grid, Nx_grid)
            V_grid = V_sorted.reshape(Ny_grid, Nx_grid)
            exx_plastic = exx_plastic_sorted.reshape(Ny_grid, Nx_grid)
            eyy_plastic = eyy_plastic_sorted.reshape(Ny_grid, Nx_grid)
            domain_map_matrix = domain_map_sorted.reshape(Ny_grid, Nx_grid)

            # 5) Compute grid spacings from actual coordinates
            dx = np.diff(x_unique).mean() if Nx_grid > 1 else 1.0
            dy = np.diff(y_unique).mean() if Ny_grid > 1 else 1.0


            # 6) Total SMALL strains from displacement gradients
            dU_dx = np.gradient(U_grid, dx, axis=1)
            dV_dy = np.gradient(V_grid, dy, axis=0)

            exx_total = dU_dx
            eyy_total = dV_dy


            # 7) Elastic = Total − Plastic
            exx_elastic = exx_total - exx_plastic
            eyy_elastic = eyy_total - eyy_plastic


            # 8) Plane strain stresses
            factor = e / ((1 + nu) * (1 - 2*nu))
            c11 = factor * (1 - nu)
            c12 = factor * nu

            stress_xx_fem = domain_map_matrix * (c11 * exx_elastic + c12 * eyy_elastic)
            stress_yy_fem = domain_map_matrix * (c12 * exx_elastic + c11 * eyy_elastic)

            # 9) Plot
            im = axs[1, 1].imshow(stress_xx_fem, cmap='plasma', origin='lower')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 1].set_title('$\sigma_{xx}$, FEM', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            im = axs[1, 2].imshow(stress_yy_fem, cmap='plasma', origin='lower')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
            axs[1, 2].set_title('$\sigma_{yy}$, FEM', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

            # ---- compute equivalent strain per node ----
            strain_nodes=vectors_list[2]
            #eq_nodes=von_mises_equivalent_strain(strain_nodes)
            eq_nodes=strain_nodes[:,0,0]
            # --- sort nodes onto a structured (Ny, Nx) grid ---
            # sort by y first, then x, so within each row (fixed y) x increases
            coords=vectors_list[3]
            U_full=vectors_list[4]
            idx = np.lexsort((coords[:, 0], coords[:, 1]))
            coords_s   = coords[idx]
            U_s        = U_full[idx]
            vals_s     = np.asarray(eq_nodes, dtype=float)[idx]

            # infer grid shape from unique coordinates
            x_unique = np.unique(coords_s[:, 0])
            y_unique = np.unique(coords_s[:, 1])
            Nx, Ny = len(x_unique), len(y_unique)


            # reshape to (Ny, Nx): y is slow index (rows), x is fast index (cols)
            Z   = vals_s.reshape(Ny, Nx)
            Ux  = U_s[:, 0].reshape(Ny, Nx)
            Uy  = U_s[:, 1].reshape(Ny, Nx)

            # build *undeformed* grid from unique x/y
            X, Y = np.meshgrid(x_unique, y_unique, indexing='xy')

            # deformed grid
            def_scale = 2.0
            X_def = X + def_scale * Ux
            Y_def = Y + def_scale * Uy

            contour = axs[2,0].contourf(X_def, Y_def, Z, levels=20, cmap='viridis')
            cbar = fig.colorbar(contour, ax=axs[2,0],pad=0.02, shrink=0.7)
            cbar.set_label(r'$\varepsilon_{\mathrm{xx}}$')

            axs[2,0].set_aspect('equal', adjustable='box')
            axs[2,0].set_xlabel('x'); axs[1,2].set_ylabel('y')
            axs[2,0].set_title('Plastic strain in XX direction')

            eq_nodes=strain_nodes[:,1,1]
        # --- sort nodes onto a structured (Ny, Nx) grid ---
            # sort by y first, then x, so within each row (fixed y) x increases
            coords=vectors_list[3]
            U_full=vectors_list[4]
            idx = np.lexsort((coords[:, 0], coords[:, 1]))
            coords_s   = coords[idx]
            U_s        = U_full[idx]
            vals_s     = np.asarray(eq_nodes, dtype=float)[idx]

            # infer grid shape from unique coordinates
            x_unique = np.unique(coords_s[:, 0])
            y_unique = np.unique(coords_s[:, 1])
            Nx, Ny = len(x_unique), len(y_unique)


            # reshape to (Ny, Nx): y is slow index (rows), x is fast index (cols)
            Z   = vals_s.reshape(Ny, Nx)
            Ux  = U_s[:, 0].reshape(Ny, Nx)
            Uy  = U_s[:, 1].reshape(Ny, Nx)

            # build *undeformed* grid from unique x/y
            X, Y = np.meshgrid(x_unique, y_unique, indexing='xy')

            # deformed grid
            def_scale = 2.0
            X_def = X + def_scale * Ux
            Y_def = Y + def_scale * Uy


            contour = axs[2,1].contourf(X_def, Y_def, Z, levels=20, cmap='viridis')
            cbar = fig.colorbar(contour, ax=axs[2,1],pad=0.02, shrink=0.7)
            cbar.set_label(r'$\varepsilon_{\mathrm{yy}}$')

            axs[2,1].set_aspect('equal', adjustable='box')
            axs[2,1].set_xlabel('x'); axs[2,1].set_ylabel('y')
            axs[2,1].set_title('Plastic strain in YY direction')



            # Simple scatter plot with interpolation
            scatter = axs[2,2].tricontourf(coords[:, 0], coords[:, 1], np.array(vectors_list[5]), 
                                    levels=20, cmap='plasma')
            cbar = plt.colorbar(scatter, ax=axs[2,2])
            cbar.set_label('Accumulated Plastic Strain (ξ)', fontsize=12)

            # Add contour lines
            axs[2,2].tricontour(coords[:, 0], coords[:, 1], np.array(vectors_list[5]), 
                        levels=20, colors='k', alpha=0.3, linewidths=0.5)

            # Plot mesh edges
            for e in range(nelem):
                nodes = conn[e]
                for i in range(4):
                    n1 = nodes[i]
                    n2 = nodes[(i + 1) % 4]
                    axs[2,2].plot([coords[n1, 0], coords[n2, 0]],
                            [coords[n1, 1], coords[n2, 1]],
                            'k-', linewidth=0.3, alpha=0.5)

            axs[2,2].set_xlabel('X', fontsize=12)
            axs[2,2].set_ylabel('Y', fontsize=12)
            axs[2,2].set_title('Accumulated Plastic Strain (ξ) - Final Increment', 
                        fontsize=14, fontweight='bold')
            axs[2,2].set_aspect('equal')
            axs[2,2].grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(file_name, dpi=300)

        def plot_all_convergence_metrics(convergence_history, file_name, abs_tol=None, rel_tol=None):
            """
            Convergence plot with 2 subplots: Iterations per Load Step and Final Residual vs Load Factor
            """
            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
            
            load_steps = [h['load_step'] for h in convergence_history]
            iterations = [h['iterations'] for h in convergence_history]
            final_residuals = [h['final_residual'] for h in convergence_history]
            load_factors = [h['load_factor'] for h in convergence_history]
            
            # 1. Iterations per load step (bar chart)
            ax1 = fig.add_subplot(gs[0, 0])
            bars = ax1.bar(load_steps, iterations, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Load Step', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Iterations', fontsize=11, fontweight='bold')
            ax1.set_title('Iterations per Load Step', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            for bar, itr in zip(bars, iterations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(itr)}', ha='center', va='bottom', fontsize=9)
            
            # 2. Final residual vs load factor
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(load_factors, final_residuals, 'o-', color='darkgreen',
                    linewidth=2, markersize=8, markerfacecolor='lightgreen')
            if abs_tol:
                ax2.axhline(y=abs_tol, color='r', linestyle='--', linewidth=2, label='Abs Tol')
            ax2.set_xlabel('Load Factor (λ)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Final Residual Norm', fontsize=11, fontweight='bold')
            ax2.set_title('Final Residual vs Load Factor', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            if abs_tol:
                ax2.legend()
            
            # Add summary statistics
            total_itr = sum(iterations)
            avg_itr = np.mean(iterations)
            stats_text = f'Total: {total_itr} | Avg: {avg_itr:.1f} | Max: {max(iterations)}'
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()


        def plot_residual_convergence(convergence_history, file_name, rel_tol=None):
            """
            Separate plot for residual norm vs iteration (cumulative x-axis across all load steps)
            """
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Color map for different load steps
            colors = plt.cm.viridis(np.linspace(0, 1, len(convergence_history)))
            
            # Plot each step one after another on the x-axis
            x_start = 1
            for i, history in enumerate(convergence_history):
                residuals = np.asarray(history['residuals'])
                n = len(residuals)
                x = np.arange(x_start, x_start + n)  # cumulative x
                ax.semilogy(
                    x, residuals, 'o-', color=colors[i],
                    linewidth=2, markersize=5, alpha=0.8,
                    label=f"Step {history['load_step']}"
                )
                x_start += n  # next step continues where this one ended
            
            # Tolerance line
            if rel_tol is not None:
                ax.axhline(y=rel_tol, color='r', linestyle='--', linewidth=2, label='Rel Tol')
            
            # Labels & title
            ax.set_xlabel('Cumulative Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residual Norm', fontsize=12, fontweight='bold')
            ax.set_title('Residual Convergence Curves (Log Scale)', fontsize=14, fontweight='bold')
            
            
            # Grid
            ax.grid(True, which='both', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()




        def plot_reac_disp(vectors_list: list, state_snapshots: list, conn, H_matrix,
                                displacement_history, file_name: str = "plot", 
                                loss_settings: dict = {}):
            """
            OPTIMIZED: Uses vectorized JAX operations instead of Python loops
            Computes reaction force on LEFT boundary (fixed side)
            and plots vs equivalent displacement sqrt(Ux^2 + Uy^2) on RIGHT side
            """
            
            # Extract material properties
            nu = loss_settings["poisson_ratio"]
            e = loss_settings["young_modulus"]
            
            # Material stiffness (plane strain)
            factor = e / ((1 + nu) * (1 - 2*nu))
            c11 = factor * (1 - nu)
            c12 = factor * nu
            c33 = factor * (1 - 2*nu) / 2
            
            # Extract from vectors_list
            coords = vectors_list[3]
            Ux_right = vectors_list[6]
            Uy_right = vectors_list[7]
            
            # Get mesh info
            nelem = conn.shape[0]
            nnod = coords.shape[0]
            
            # Convert conn to JAX array for vectorized operations
            conn_jax = jnp.array(conn, dtype=jnp.int32)
            
            # Boundary nodes
            y_coords = coords[:, 1]
            x_coords = coords[:, 0]
            x_min, x_max = x_coords.min(), x_coords.max()
            
            tol = 1e-6
            left_nodes = np.where(np.abs(x_coords - x_min) < tol)[0]
            right_nodes = np.where(np.abs(x_coords - x_max) < tol)[0]
            
            
            # ==================================================================
            # PRE-COMPUTE: Strain computation function (vectorized)
            # ==================================================================
            def compute_total_strain_at_nodes_vectorized(U_k, coords, conn_jax):
                """Vectorized strain computation - much faster than element loops"""
                nelem = conn_jax.shape[0]
                nnod = coords.shape[0]
                
                # Reshape displacement vector
                U_nodes = U_k.reshape((nnod, 2))
                
                # Get element nodal coordinates and displacements
                elem_coords = coords[conn_jax]  # (nelem, 4, 2)
                elem_U = U_nodes[conn_jax]       # (nelem, 4, 2)
                
                # Compute strain at element centroids using simple averaging
                # This is approximate but much faster than full FE strain computation
                # For more accuracy, you'd need to implement full shape function derivatives
                
                # Average nodal strains (simple finite difference approach)
                nodal_exx = jnp.zeros(nnod)
                nodal_eyy = jnp.zeros(nnod)
                nodal_exy = jnp.zeros(nnod)
                nodal_count = jnp.zeros(nnod)
                
                # For each element, compute approximate strain and distribute to nodes
                for e in range(nelem):
                    nodes = conn_jax[e]
                    x = elem_coords[e, :, 0]
                    y = elem_coords[e, :, 1]
                    ux = elem_U[e, :, 0]
                    uy = elem_U[e, :, 1]
                    
                    # Simple finite difference (this is a rough approximation)
                    dx = x.max() - x.min()
                    dy = y.max() - y.min()
                    
                    if dx > 1e-10:
                        exx = (ux.max() - ux.min()) / dx
                    else:
                        exx = 0.0
                        
                    if dy > 1e-10:
                        eyy = (uy.max() - uy.min()) / dy
                    else:
                        eyy = 0.0
                        
                    exy = 0.0
                    if dx > 1e-10 and dy > 1e-10:
                        exy = 0.5 * ((ux.max() - ux.min()) / dy + (uy.max() - uy.min()) / dx)
                    
                    # Distribute to element nodes
                    for a in range(4):
                        n = nodes[a]
                        nodal_exx = nodal_exx.at[n].add(exx)
                        nodal_eyy = nodal_eyy.at[n].add(eyy)
                        nodal_exy = nodal_exy.at[n].add(exy)
                        nodal_count = nodal_count.at[n].add(1.0)
                
                nodal_exx = nodal_exx / jnp.maximum(nodal_count, 1.0)
                nodal_eyy = nodal_eyy / jnp.maximum(nodal_count, 1.0)
                nodal_exy = nodal_exy / jnp.maximum(nodal_count, 1.0)
                
                return nodal_exx, nodal_eyy, nodal_exy
            
            # ==================================================================
            # OPTIMIZED: Vectorized scatter-add for plastic strain
            # ==================================================================
            equiv_disp_history = []
            reaction_x_history = []
            reaction_y_history = []
            reaction_total_history = []
            
            # Process each load increment
            for k, U_k in enumerate(displacement_history):
                
                # ==================================================================
                # STEP 1: Extrapolate PLASTIC strain from GP to nodes (VECTORIZED)
                # ==================================================================
                state_k = state_snapshots[k]  # (nelem, ngp, nstate)
                exx_plastic_gp = state_k[..., 0]
                eyy_plastic_gp = state_k[..., 1]
                exy_plastic_gp = state_k[..., 2]
                
                # Vectorized extrapolation
                nodal_exx_plastic = extrapolate_gp_to_nodes_vectorized(
                    exx_plastic_gp, H_matrix, conn_jax, nnod)
                nodal_eyy_plastic = extrapolate_gp_to_nodes_vectorized(
                    eyy_plastic_gp, H_matrix, conn_jax, nnod)
                nodal_exy_plastic = extrapolate_gp_to_nodes_vectorized(
                    exy_plastic_gp, H_matrix, conn_jax, nnod)
                
                # ==================================================================
                # STEP 2: Compute TOTAL strain at nodes from displacements
                # ==================================================================
                nodal_exx_total, nodal_eyy_total, nodal_exy_total = \
                    compute_total_strain_at_nodes_vectorized(U_k, coords, conn_jax)
                
                # ==================================================================
                # STEP 3: Compute ELASTIC strain at nodes
                # ==================================================================
                nodal_exx_elastic = nodal_exx_total - nodal_exx_plastic
                nodal_eyy_elastic = nodal_eyy_total - nodal_eyy_plastic
                nodal_exy_elastic = nodal_exy_total - nodal_exy_plastic
                
                # ==================================================================
                # STEP 4: Compute STRESS at nodes from elastic strain
                # ==================================================================
                nodal_sxx = c11 * nodal_exx_elastic + c12 * nodal_eyy_elastic
                nodal_syy = c12 * nodal_exx_elastic + c11 * nodal_eyy_elastic
                nodal_sxy = 2 * c33 * nodal_exy_elastic
                
                # Convert to numpy for integration
                nodal_sxx = np.array(nodal_sxx)
                nodal_syy = np.array(nodal_syy)
                nodal_sxy = np.array(nodal_sxy)
                
                # ==================================================================
                # STEP 5: Integrate reaction force on LEFT boundary
                # ==================================================================
                reaction_x = 0.0
                reaction_y = 0.0
                
                if len(left_nodes) > 0:
                    y_left = coords[left_nodes, 1]
                    sxx_left = nodal_sxx[left_nodes]
                    sxy_left = nodal_sxy[left_nodes]
                    
                    # Sort by y-coordinate
                    sort_idx = np.argsort(y_left)
                    y_left = y_left[sort_idx]
                    sxx_left = sxx_left[sort_idx]
                    sxy_left = sxy_left[sort_idx]
                    
                    # Integrate
                    reaction_x = -np.trapezoid(sxx_left, y_left)
                    reaction_y = -np.trapezoid(sxy_left, y_left)
                
                reaction_total = np.sqrt(reaction_x**2 + reaction_y**2)
                
                # ==================================================================
                # STEP 6: Compute equivalent displacement on RIGHT boundary
                # ==================================================================
                if len(right_nodes) > 0:
                    Ux_right_nodes = U_k[2*right_nodes]
                    Uy_right_nodes = U_k[2*right_nodes + 1]
                    Ux_mean = float(Ux_right_nodes.mean())
                    Uy_mean = float(Uy_right_nodes.mean())
                    U_equiv = np.sqrt(Ux_mean**2 + Uy_mean**2)
                else:
                    scale_factor = (k + 1) / len(displacement_history)
                    Ux_mean = scale_factor * Ux_right
                    Uy_mean = scale_factor * Uy_right
                    U_equiv = np.sqrt(Ux_mean**2 + Uy_mean**2)
                
                equiv_disp_history.append(float(U_equiv))
                reaction_x_history.append(float(reaction_x))
                reaction_y_history.append(float(reaction_y))
                reaction_total_history.append(float(reaction_total))
                
            
            # ==================================================================
            # STEP 7: Create single plot - Total Reaction vs Equivalent Displacement
            # ==================================================================
            plt.figure(figsize=(10, 7))
            
            # Plot with dots at data points and connecting lines (no hollow circles)
            plt.plot(equiv_disp_history, reaction_total_history, '-', linewidth=2, 
                    color='#2E86AB', label='Total Reaction Force')
            plt.plot(equiv_disp_history, reaction_total_history, '.', markersize=8, 
                    color='#2E86AB')
            
            plt.xlabel('Equivalent Displacement √(Ux² + Uy²)', fontsize=13, fontweight='bold')
            plt.ylabel('Total Reaction Force (Left Boundary)', fontsize=13, fontweight='bold')
            plt.title('Total Reaction Force vs Equivalent Displacement', fontsize=15, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
            plt.legend(fontsize=11, loc='best')
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            

        snap = np.stack(state_snapshots, axis=0)

        # Use only the first three comps (0: exx, 1: eyy, 2: exy). Ignore comp 3+.
        exx = snap[..., 0]                # (n_inc, n_elem, n_gp)
        eyy = snap[..., 1]
        exy = snap[..., 2]             

        # Allocate output
        e_hist = np.zeros(snap.shape[:3] + (2, 2), dtype=snap.dtype)

        # Fill symmetric 2×2 tensor
        e_hist[..., 0, 0] = exx
        e_hist[..., 1, 1] = eyy
        e_hist[..., 0, 1] = exy         
        e_hist[..., 1, 0] = exy


        # 1) Build E from your own FE element APIs (robust to reordering)
        
        last_inc = e_hist[-1]
        strain_nodes = extrapolate_gp_to_nodes_vectorized_tensor(last_inc, H, conn, nnod)
        coords = np.asarray(fe_mesh.GetNodesCoordinates())[:, :2]                 # (nnodes, 2)
        U_full = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))                   # (nnodes, 2)        
        fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))
        

        xi_history = [snapshot[..., 3] for snapshot in state_snapshots]
        xi_history = np.array(xi_history)
        xi_final = xi_history[-1]  # (n_elem, n_gp)
        xi_nodes = extrapolate_gp_to_nodes_vectorized(xi_final, H, conn, nnod)
        vectors_list_U = [K_matrix[eval_id],FE_UV,strain_nodes,coords,U_full,xi_nodes]
        

        plot_mesh_res_1(
            vectors_list_U,
            file_name=os.path.join(case_dir, 'plot_inputs_outputs.png'),
            loss_settings=material_dict
        )

        n_incr = fe_setting["nonlinear_solver_settings"]["load_incr"]
        Ux_right = float(model_settings["Ux_right"])
        Uy_right = float(model_settings["Uy_right"])

        # MODIFIED: Pass both Ux_right and Uy_right
        vectors_list_U_1 = [
            K_matrix[eval_id], FE_UV, strain_nodes, coords, U_full, 
            n_incr, Ux_right, Uy_right
        ]
        
        plot_reac_disp(
            vectors_list_U_1, state_snapshots, conn, H, displacement_history,
            file_name=os.path.join(case_dir, 'plot_reaction_force.png'),
            loss_settings=material_dict
        )
        # NEW: Comprehensive convergence plots
        abs_tol = nonlinear_fe_solver.nonlinear_solver_settings.get("abs_tol", None)
        rel_tol = nonlinear_fe_solver.nonlinear_solver_settings.get("rel_tol", None)

           # First plot: 2 metrics
        plot_all_convergence_metrics(
            convergence_history,
            file_name=os.path.join(case_dir, 'convergence_metrics.png'),
            abs_tol=abs_tol,
            rel_tol=rel_tol
        )
        
        # Second plot: Residual curves
        plot_residual_convergence(
      
            convergence_history,
            file_name=os.path.join(case_dir, 'residual_convergence.png'),
            rel_tol=rel_tol
        )
        
    # finalize and export mesh data
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Defaults
    solve_FE = True
    clean_dir = False

    main(solve_FE, clean_dir)
