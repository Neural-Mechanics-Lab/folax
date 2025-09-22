
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math
import gmsh
import meshio
import os
import shutil
from fol.mesh_input_output.mesh import Mesh
import copy
from fol.loss_functions.fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

def plot_mesh_vec_data(L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
                       block_bool=False, colour_bar=True, colour_bar_name=None,
                       X_axis_name=None, Y_axis_name=None, show=False, file_name=None):
    num_vectors = len(vectors_list)
    if num_vectors < 1 or num_vectors > 4:
        raise ValueError("vectors_list must contain between 1 and 4 elements.")

    if subplot_titles is not None and len(subplot_titles) != num_vectors:
        raise ValueError("subplot_titles must have the same number of elements as vectors_list if provided.")

    # Determine the grid size for the subplots
    grid_size = math.ceil(math.sqrt(num_vectors))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 5*grid_size), squeeze=False)
    
    # Flatten the axs array and hide unused subplots if any
    axs = axs.flatten()
    for ax in axs[num_vectors:]:
        ax.axis('off')

    for i, squared_mesh_vec_data in enumerate(vectors_list):
        N = int((squared_mesh_vec_data.reshape(-1, 1).shape[0])**0.5)
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L])

        if subplot_titles is not None:
            axs[i].set_title(subplot_titles[i])
        else:
            axs[i].set_title(f'Plot {i+1}')

        if colour_bar:
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        if X_axis_name is not None:
            axs[i].set_xlabel(X_axis_name)

        if Y_axis_name is not None:
            axs[i].set_ylabel(Y_axis_name)

    if fig_title is not None:
        plt.suptitle(fig_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show(block=block_bool)

    if file_name is not None:
        plt.savefig(file_name)

def plot_data_input(input_morph, num_columns, filename):

    N = int(input_morph.shape[1]**0.5)
    L = 1

    # Calculate the number of rows based on the number of columns and the length of input_morph
    num_rows = int(np.ceil(len(input_morph) / num_columns))

    # Create a new figure with variable subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows))

    # Flatten the axes array to handle variable numbers of subplots
    axes = axes.flatten()

    # Loop through the input_morph rows and plot each row in a separate subplot
    for i in range(len(input_morph)):
        ax = axes[i] if i < len(axes) else None  # Handle cases with fewer subplots than data
        if ax:
            Z = input_morph[i].reshape(N, N)  # Reshape the vectorized Z to a 2D array
            min_val = np.min(Z)
            max_val = np.max(Z)
            im = ax.imshow(Z, cmap='viridis', extent=[0, L, 0, L], vmin=min_val, vmax=max_val)
            ax.set_title(f'Row {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a color bar at the top of the figure
    cbar_ax = fig.add_axes([0.3, 1.02, 0.4, 0.005])  # Define position and size of the color bar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    # Remove any unused subplots
    for i in range(len(input_morph), len(axes)):
        fig.delaxes(axes[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the plot as a PDF and PNG file with user-defined filename
    plt.savefig(f'{filename}.png')

def create_2D_square_model_info_thermal(L,N,T_left,T_right):
    # FE init starts here
    Ne = N - 1  # Number of elements in each direction
    nx = Ne + 1  # Number of nodes in the x-direction
    ny = Ne + 1  # Number of nodes in the y-direction
    ne = Ne * Ne    # Total number of elements
    # Generate mesh coordinates
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, L, ny)
    X, Y = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = jnp.zeros((Y.shape[-1]))
    # Gauss quadrature points and weights (for a 2x2 integration)
    # Create a matrix to store element nodal information
    elements_nodes = jnp.zeros((ne, 4), dtype=int)
    # Fill in the elements_nodes with element and node numbers
    for i in range(Ne):
        for j in range(Ne):
            e = i * Ne + j  # Element index
            # Define the nodes of the current element
            nodes = jnp.array([i * (Ne + 1) + j, i * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j])
            # Store element and node numbers in the matrix
            elements_nodes = elements_nodes.at[e].set(nodes) # Node numbers

    element_ids = jnp.arange(0,elements_nodes.shape[0])

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    left_boundary_nodes_values = T_left * jnp.ones(left_boundary_nodes.shape)
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary
    right_boundary_nodes_values = T_right * jnp.ones(right_boundary_nodes.shape)
    boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    boundary_values = jnp.concatenate([left_boundary_nodes_values, right_boundary_nodes_values])
    non_boundary_nodes = []
    for i in range(N*N):
        if not jnp.any(boundary_nodes == i):
            non_boundary_nodes.append(i)
    non_boundary_nodes = jnp.array(non_boundary_nodes)

    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":element_ids,"elements_nodes":elements_nodes}
    dofs_dict = {"T":{"non_dirichlet_nodes_ids":non_boundary_nodes,"dirichlet_nodes_ids":boundary_nodes,"dirichlet_nodes_dof_value":boundary_values}}
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict}

def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, case_dir):

    cell_type = 'hexahedron'
    degree= 1
    msh_dir = case_dir
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box.msh')

    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    domain_x = Lx
    domain_y = Ly
    domain_z = Lz

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    if cell_type.startswith('tetra'):
        Rec2d = False  # tris or quads
        Rec3d = False  # tets, prisms or hexas
    else:
        Rec2d = True
        Rec3d = True
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    meshio_obj = meshio.Mesh(points=points, cells={cell_type: cells})

    return meshio_obj

def create_3D_box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir):

    # create empty fe mesh object
    fe_mesh = Mesh("box_io","box.")

    settings = box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir)
    fe_mesh.node_ids = jnp.arange(len(settings.points))
    fe_mesh.nodes_coordinates = jnp.array(settings.points)

    left_mask = jnp.isclose(fe_mesh.nodes_coordinates[:,0], 0.0, atol=1e-5)
    right_mask = jnp.isclose(fe_mesh.nodes_coordinates[:,0], Lx, atol=1e-5)

    left_boundary_node_ids = fe_mesh.node_ids[left_mask]
    right_boundary_node_ids = fe_mesh.node_ids[right_mask]

    fe_mesh.elements_nodes = {"hexahedron":jnp.array(settings.cells_dict['hexahedron'])}

    fe_mesh.node_sets = {"left":left_boundary_node_ids,
                         "right":right_boundary_node_ids}
    
    fe_mesh.mesh_io = meshio.Mesh(fe_mesh.nodes_coordinates,fe_mesh.elements_nodes)

    fe_mesh.is_initialized = True

    return fe_mesh

def create_2D_square_mesh(L,N):

    # create empty fe mesh object
    fe_mesh = Mesh("square_io","square.")

    # FE init starts here
    Ne = N - 1  # Number of elements in each direction
    nx = Ne + 1  # Number of nodes in the x-direction
    ny = Ne + 1  # Number of nodes in the y-direction
    ne = Ne * Ne    # Total number of elements
    # Generate mesh coordinates
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, L, ny)
    X, Y = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = jnp.zeros((Y.shape[-1]))

    fe_mesh.node_ids = jnp.arange(Y.shape[-1])
    fe_mesh.nodes_coordinates = jnp.stack((X,Y,Z), axis=1)

    # Create a matrix to store element nodal information
    elements_nodes = jnp.zeros((ne, 4), dtype=int)
    # Fill in the elements_nodes with element and node numbers
    for i in range(Ne):
        for j in range(Ne):
            e = i * Ne + j  # Element index
            # Define the nodes of the current element
            nodes = jnp.array([i * (Ne + 1) + j, i * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j])
            # Store element and node numbers in the matrix
            elements_nodes = elements_nodes.at[e].set(nodes) # Node numbers

    fe_mesh.elements_nodes = {"quad":elements_nodes}

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary

    fe_mesh.node_sets = {"left":left_boundary_nodes,
                         "right":right_boundary_nodes}
    
    fe_mesh.mesh_io = meshio.Mesh(fe_mesh.nodes_coordinates,fe_mesh.elements_nodes)

    fe_mesh.is_initialized = True

    return fe_mesh

def create_random_fourier_samples(fourier_control,numberof_sample):
    N = int(fourier_control.GetNumberOfControlledVariables()**0.5)
    num_coeffs = fourier_control.GetNumberOfVariables()
    coeffs_matrix = np.zeros((0,num_coeffs))
    for i in range (numberof_sample):
        coeff_vec = np.random.normal(size=num_coeffs)
        coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # also add uniform dstibuted K of value 0.5
    coeff_vec = 1e-4 * np.zeros((num_coeffs))
    coeff_vec[0] = 10
    coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))
    K_matrix = np.vstack((K_matrix,fourier_control.ComputeControlledVariables(coeff_vec)))
    # plot_data_input(K_matrix,10,'K distributions')    

    return coeffs_matrix,K_matrix


def create_random_voronoi_samples(voronoi_control,number_of_sample,dim=2):
    number_seeds = voronoi_control.number_of_seeds
    rangeofValues = voronoi_control.E_values
    numberofVar = voronoi_control.num_control_vars
    coeffs_matrix = np.zeros((0,numberofVar))
    
    for _ in range(number_of_sample):
        x_coords = np.random.rand(number_seeds)
        y_coords = np.random.rand(number_seeds)
        if dim == 3:
            z_coords = np.random.rand(number_seeds)
        
        
        if isinstance(rangeofValues, tuple):
            E_values = np.random.uniform(rangeofValues[0],rangeofValues[-1],number_seeds)
        if isinstance(rangeofValues, list):
            E_values = np.random.choice(rangeofValues, size=number_seeds)
        
        Kcoeffs = np.zeros((0,numberofVar))
        if dim == 3:
            Kcoeffs = np.concatenate((x_coords.reshape(1,-1), y_coords.reshape(1,-1), 
                                  z_coords.reshape(1,-1), E_values.reshape(1,-1)), axis=1)
        else:
            Kcoeffs = np.concatenate((x_coords.reshape(1,-1), y_coords.reshape(1,-1), 
                                      E_values.reshape(1,-1)), axis=1)
        
        coeffs_matrix = np.vstack((coeffs_matrix,Kcoeffs))
    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)
    return coeffs_matrix,K_matrix

def create_clean_directory(case_dir):
    # Check if the directory exists
    if os.path.exists(case_dir):
        # Remove the directory and all its contents
        shutil.rmtree(case_dir)
    
    # Create the new directory
    os.makedirs(case_dir)

def plot_mesh_res(vectors_list:list, file_name:str="plot",dir:str="U"):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # Adjusted to 4 columns

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
    im = axs[0, 1].imshow(data.reshape(N, N), cmap='bone', aspect='equal')
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

    # Zoomed-in region
    zoomed_min = int(0.2*N)
    zoomed_max = int(0.4*N)
    zoom_region = data.reshape(N, N)[zoomed_min:zoomed_max, zoomed_min:zoomed_max]
    im = axs[0, 2].imshow(zoom_region, cmap='bone', aspect='equal')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_xticklabels([])  # Remove text on x-axis
    axs[0, 2].set_yticklabels([])  # Remove text on y-axis
    axs[0, 2].set_title(f'Zoomed-in: $x \in [{zoomed_min/N:.2f}, {zoomed_max/N:.2f}], y \in [{zoomed_min/N:.2f}, {zoomed_max/N:.2f}]$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the mesh grid
    axs[0, 2].xaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].yaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].grid(color='red', linestyle='-', linewidth=2)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[0].reshape(N, N)
    axs[0, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label='Elasticity', color='black')
    axs[0, 3].set_xlim([0, 1])
    #axs[0, 3].set_ylim([min(U1[y_idx, :].min()), max(U1[y_idx, :].max())])
    axs[0, 3].set_aspect(aspect='auto')
    axs[0, 3].set_title('Cross-section of E at y=0.5', fontsize=fontsize)
    axs[0, 3].legend(fontsize=fontsize)
    axs[0, 3].grid(True)
    axs[0, 3].set_xlabel('x', fontsize=fontsize)
    axs[0, 3].set_ylabel('E', fontsize=fontsize)


    # Plot the second entity in the second row
    data = vectors_list[1]
    im = axs[1, 0].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title(f'${dir}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the fourth entity in the second row
    data = vectors_list[2]
    im = axs[1, 1].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title(f'${dir}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the absolute difference between vectors_list[1] and vectors_list[3] in the third row, second column
    diff_data_1 = np.abs(vectors_list[1] - vectors_list[2])
    im = axs[1, 2].imshow(diff_data_1.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title(f'Abs. Difference ${dir}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[1].reshape(N, N)
    U2 = vectors_list[2].reshape(N, N)
    axs[1, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label=f'{dir} FOL', color='blue')
    axs[1, 3].plot(np.linspace(0, 1, N), U2[y_idx, :], label=f'{dir} FEM', color='red')
    axs[1, 3].set_xlim([0, 1])
    axs[1, 3].set_ylim([min(U1[y_idx, :].min(), U2[y_idx, :].min()), max(U1[y_idx, :].max(), U2[y_idx, :].max())])
    axs[1, 3].set_aspect(aspect='auto')
    axs[1, 3].set_title(f'Cross-section of {dir} at y=0.5', fontsize=fontsize)
    axs[1, 3].legend(fontsize=fontsize)
    axs[1, 3].grid(True)
    axs[1, 3].set_xlabel('x', fontsize=fontsize)
    axs[1, 3].set_ylabel(f'{dir}', fontsize=fontsize)

    plt.tight_layout()

    # Save the figure in multiple formats
    plt.savefig(file_name, dpi=300)
    # plt.savefig(plot_name+'.pdf')


def plot_mesh_grad_res_mechanics(vectors_list:list, file_name:str="plot", loss_settings:dict={}):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))

    data = vectors_list[0]
    L = 1
    N = int((data.reshape(-1, 1).shape[0])**0.5)
    nu = loss_settings["poisson_ratio"]
    e = loss_settings["young_modulus"]
    mu = e / (2*(1+nu))
    lambdaa = nu * e / ((1+nu)*(1-2*nu))
    c1 = e / (1 - nu**2)

    dx = L / (N - 1)

    U_fem = vectors_list[2][::2]
    V_fem = vectors_list[2][1::2]
    domain_map_matrix = vectors_list[0].reshape(N, N)
    dU_dx_fem = np.gradient(U_fem.reshape(N, N), dx, axis=1)
    dV_dy_fem = np.gradient(V_fem.reshape(N, N), dx, axis=0)
    stress_xx_fem = domain_map_matrix * c1 * (dU_dx_fem + nu * dV_dy_fem) # plain stress condition
    stress_yy_fem = domain_map_matrix * c1 * (nu * dU_dx_fem + dV_dy_fem) # plain stress condition

    im = axs[0, 1].imshow(stress_xx_fem, cmap='plasma')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_title('$\sigma_{xx}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    im = axs[1, 1].imshow(stress_yy_fem, cmap='plasma')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title('$\sigma_{yy}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    U_fol = vectors_list[1][::2]
    V_fol = vectors_list[1][1::2]
    dU_dx_fol = np.gradient(U_fol.reshape(N, N), dx, axis=1)
    dV_dy_fol = np.gradient(V_fol.reshape(N, N), dx, axis=0)
    stress_xx_fol = domain_map_matrix * c1 * (dU_dx_fol + nu * dV_dy_fol) # plain stress condition
    stress_yy_fol = domain_map_matrix * c1 * (nu * dU_dx_fol + dV_dy_fol) # plain stress condition

    min_v = np.min(stress_xx_fem)
    max_v = np.max(stress_xx_fem)
    im = axs[0, 0].imshow(stress_xx_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('$\sigma_{xx}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    min_v = np.min(stress_yy_fem)
    max_v = np.max(stress_yy_fem)
    im = axs[1, 0].imshow(stress_yy_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title('$\sigma_{yy}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    diff_data_2 = np.abs(stress_xx_fem - stress_xx_fol)
    im = axs[0, 2].imshow(diff_data_2, cmap='plasma')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_title('Abs. Difference $\sigma_{xx}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    diff_data_2 = np.abs(stress_yy_fem - stress_yy_fol)
    im = axs[1, 2].imshow(diff_data_2, cmap='plasma')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title('Abs. Difference $\sigma_{yy}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    # Extract cross-sections at y = 0.5
    y_index = N // 2
    stress_x_cross_fem = stress_xx_fem[y_index, :]
    stress_y_cross_fem = stress_yy_fem[y_index, :]
    stress_x_cross_fol = stress_xx_fol[y_index, :]
    stress_y_cross_fol = stress_yy_fol[y_index, :]

    # Plot cross-sections in the fourth column
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fem, label='FEM', color='r')
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fol, label='FOL', color='b')
    axs[0, 3].set_title('Cross-section $\sigma_{xx}$', fontsize=fontsize)
    axs[0, 3].legend()

    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fem, label='FEM', color='r')
    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fol, label='FOL', color='b')
    axs[1, 3].set_title('Cross-section $\sigma_{yy}$', fontsize=fontsize)
    axs[1, 3].legend()

    # Save cross-section data to a text file
    file_dir = os.path.join(os.path.dirname(os.path.abspath(file_name)),'cross_section_data.txt')
    with open(file_dir, 'w') as f:
        f.write('x, stress_x_fem, stress_x_fol, stress_y_fem, stress_y_fol, stress_xy_fem, stress_xy_fol\n')
        for i in range(N):
            f.write(f'{i*dx}, {stress_x_cross_fem[i]}, {stress_x_cross_fol[i]}, {stress_y_cross_fem[i]}, {stress_y_cross_fol[i]}\n')


    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    # plt.savefig(plot_name+'.pdf')


def UpdateDefaultDict(default_dict:dict,given_dict:dict):
    filtered_update = {k: given_dict[k] for k in default_dict if k in given_dict}
    updated_dict = copy.deepcopy(default_dict)
    updated_dict.update(filtered_update)
    return updated_dict

    
# -------------------------------------------------------------------------
#  Dirichlet-BC helper: JAX-FEM torsion of a 1×1×1 cube
# -------------------------------------------------------------------------
import jax.numpy as jnp

def build_twist_dirichlet(mesh,
                          left_name="left",
                          right_name="right",
                          theta_deg=60.0):
    """
    Returns a bc_dict whose keys are *group names* (strings).
    Each node on the left or right face is added to mesh.node_sets
    as a group named str(node_id), so MechanicalLoss3DTetra can apply
    a constant value per ‘group’ without any code changes.
    """
    # 1. node IDs & coordinates
    left_ids  = mesh.GetNodeSet(left_name)
    right_ids = mesh.GetNodeSet(right_name)
    coords    = mesh.GetNodesCoordinates()

    y, z  = coords[left_ids, 1], coords[left_ids, 2]
    theta = jnp.deg2rad(theta_deg)
    c, s  = jnp.cos(theta), jnp.sin(theta)

    # 2. target positions after 60° rotation, then half-step
    y_rot = 0.5 + (y - 0.5) * c - (z - 0.5) * s
    z_rot = 0.5 + (y - 0.5) * s + (z - 0.5) * c

    ux_left = jnp.zeros_like(y)
    uy_left = 0.5 * (y_rot - y)
    uz_left = 0.5 * (z_rot - z)

    # 3. make one-node groups and build BC maps
    bc_ux, bc_uy, bc_uz = {}, {}, {}
    to_py = lambda v: float(v)

    for nid, ux, uy, uz in zip(left_ids, ux_left, uy_left, uz_left):
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])    # create group
        bc_ux[g] = to_py(ux)
        bc_uy[g] = to_py(uy)
        bc_uz[g] = to_py(uz)

    # right face: all zeros
    for nid in right_ids:
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])
        bc_ux[g] = 0.0
        bc_uy[g] = 0.0
        bc_uz[g] = 0.0

    return {"Ux": bc_ux, "Uy": bc_uy, "Uz": bc_uz}



# -------------------------------------------------------------------------
#  Dirichlet-BC helper: uniform stretch / compression for a 1×1×1 cube
# -------------------------------------------------------------------------
import jax.numpy as jnp

def build_uniform_stretch_dirichlet(mesh,
                                    left_name="left",
                                    right_name="right",
                                    stretch_pct=5.0,
                                    axis="x",
                                    half_step=True):
    """
    Parameters
    ----------
    mesh : FOL mesh object with GetNodeSet() & GetNodesCoordinates().
    left_name, right_name : str
        Opposite faces (node-set names) between which the stretch is applied.
        The *right* face is kept fixed (all zeros), the *left* face is displaced.
    stretch_pct : float
        Percentage of stretch relative to the unit cube side.
        +5.0  ➜  +5 % tension;   –2.0 ➜  2 % compression.
    axis : {'x','y','z', 0,1,2}
        Principal axis along which to stretch.
    half_step : bool
        `True` reproduces the 0.5 factor used in build_twist_dirichlet
        (handy for incremental loading).  Set to False for the full step.

    Returns
    -------
    dict with keys {"Ux","Uy","Uz"} mapping each one-node group to a float.
    """
    # -- 1. node IDs & coordinates ------------------------------------------------
    left_ids  = mesh.GetNodeSet(left_name)
    right_ids = mesh.GetNodeSet(right_name)
    coords    = mesh.GetNodesCoordinates()

    # ensure axis is an int 0,1,2
    axis_map  = {"x": 0, "y": 1, "z": 2}
    ax        = axis_map[axis] if isinstance(axis, str) else int(axis)

    # -- 2. displacement to impose on the left face ------------------------------
    # |Δ| = ε · L, with L = 1 for a 1×1×1 cube
    eps   = stretch_pct / 100.0           # engineering strain
    delta = eps * 1.0                     # side length = 1
    factor = 0.5 if half_step else 1.0    # match twist helper

    # initialise zero arrays with same length as left_ids
    zeros = jnp.zeros_like(coords[left_ids, 0])
    ux_left, uy_left, uz_left = zeros, zeros, zeros

    if ax == 0:
        ux_left = factor * delta * jnp.ones_like(zeros)
    elif ax == 1:
        uy_left = factor * delta * jnp.ones_like(zeros)
    elif ax == 2:
        uz_left = factor * delta * jnp.ones_like(zeros)
    else:
        raise ValueError("axis must be 0, 1, 2 or 'x', 'y', 'z'")

    # -- 3. make one-node groups and build BC maps --------------------------------
    bc_ux, bc_uy, bc_uz = {}, {}, {}
    to_py = lambda v: float(v)            # JAX scalar ➜ Python float

    for nid, ux, uy, uz in zip(left_ids, ux_left, uy_left, uz_left):
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])   # create group
        bc_ux[g] = to_py(ux)
        bc_uy[g] = to_py(uy)
        bc_uz[g] = to_py(uz)

    # right face: fully fixed
    for nid in right_ids:
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])
        bc_ux[g] = 0.0
        bc_uy[g] = 0.0
        bc_uz[g] = 0.0

    return {"Ux": bc_ux, "Uy": bc_uy, "Uz": bc_uz}



# -------------------------------------------------------------------------
#  Dirichlet-BC helper: simple shear of a 1×1×1 cube
# -------------------------------------------------------------------------
import jax.numpy as jnp

def build_simple_shear_dirichlet(mesh,
                                 left_name="left",
                                 right_name="right",
                                 shear_pct=25.0,
                                 axis="x",
                                 disp_dir="y",
                                 half_step=True):
    """
    Parameters
    ----------
    mesh : FOL mesh object
    left_name, right_name : str
        Opposite faces *normal* to `axis`.  `right_name` is fixed (all zeros);
        `left_name` receives a tangential displacement.
    shear_pct : float
        γ in percent.  +25 → 25 % shear; −10 → opposite shear of 10 %.
    axis : {'x','y','z', 0,1,2}
        Normal direction between the two faces (the “gradient” axis).
    disp_dir : {'x','y','z', 0,1,2}
        Direction **within** the face along which it slides.
        Must be different from `axis`.
    half_step : bool
        Retains the 0.5 factor of the twist helper so you can load in two
        increments if desired.  Set False for the full step.

    Returns
    -------
    bc_dict : {"Ux": {group: value}, "Uy": {…}, "Uz": {…}}
        Ready for MechanicalLoss3DTetra.
    """
    # -- sanity checks -----------------------------------------------------------
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map[axis] if isinstance(axis, str) else int(axis)
    dx = axis_map[disp_dir] if isinstance(disp_dir, str) else int(disp_dir)
    if ax == dx:
        raise ValueError("disp_dir must be orthogonal to axis")

    # -- 1. node IDs -------------------------------------------------------------
    left_ids  = mesh.GetNodeSet(left_name)
    right_ids = mesh.GetNodeSet(right_name)

    # -- 2. prescribed displacement (fixed version) ------------------------------
    gamma  = shear_pct / 100.0          # engineering shear γ
    delta  = gamma * 1.0                # side length L = 1
    factor = 0.5 if half_step else 1.0
    shift  = factor * delta             # constant shift

    zeros = jnp.zeros_like(mesh.GetNodesCoordinates()[left_ids, 0])
    ux_left, uy_left, uz_left = zeros, zeros, zeros

    if   dx == 0: ux_left = jnp.full_like(zeros, shift)  # ← keep array shape
    elif dx == 1: uy_left = jnp.full_like(zeros, shift)
    elif dx == 2: uz_left = jnp.full_like(zeros, shift)

    # -- 3. build one-node groups ------------------------------------------------
    bc_ux, bc_uy, bc_uz = {}, {}, {}
    to_py = lambda v: float(v)

    for nid, ux, uy, uz in zip(left_ids, ux_left, uy_left, uz_left):
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])
        bc_ux[g] = to_py(ux)
        bc_uy[g] = to_py(uy)
        bc_uz[g] = to_py(uz)

    # right face: fully fixed
    for nid in right_ids:
        g = str(int(nid))
        mesh.node_sets[g] = jnp.array([nid])
        bc_ux[g] = 0.0
        bc_uy[g] = 0.0
        bc_uz[g] = 0.0

    return {"Ux": bc_ux, "Uy": bc_uy, "Uz": bc_uz}

def create_S_mat(S:jnp.ndarray):
    if S.size == 3:
        S_mat = jnp.zeros((2,2))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[2,0])
        S_mat = S_mat.at[1,0].set(S[2,0])
        S_mat = S_mat.at[1,1].set(S[1,0])
    elif S.size == 6:
        S_mat = jnp.zeros((3,3))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[5,0])
        S_mat = S_mat.at[0,2].set(S[4,0])
        S_mat = S_mat.at[1,0].set(S[5,0])
        S_mat = S_mat.at[1,1].set(S[1,0])
        S_mat = S_mat.at[1,2].set(S[3,0])
        S_mat = S_mat.at[2,0].set(S[4,0])
        S_mat = S_mat.at[2,1].set(S[3,0])
        S_mat = S_mat.at[2,2].set(S[2,0])
    return S_mat

# @partial(jit)
def ComputeElementStressAtGauss(loss_function,xyze,de,uvwe):
    @jit
    def compute_at_gauss_point(gp_point):
        N_vec = loss_function.fe_element.ShapeFunctionsValues(gp_point)
        DN_DX_T = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point).T
        J = loss_function.fe_element.Jacobian(xyze,gp_point)

        e_at_gauss = jnp.dot(N_vec, de.squeeze())
        k_at_gauss = e_at_gauss / (3 * (1 - 2*loss_function.v))
        mu_at_gauss = e_at_gauss / (2 * (1 + loss_function.v))

        _,F,_ = loss_function.CalculateKinematics(DN_DX_T,uvwe)
        _,S,_ = loss_function.material_model.evaluate(F,k_at_gauss,mu_at_gauss)
        S_mat = create_S_mat(S)
        P = jnp.dot(F,S_mat)    # first Piola Kirchhoff stress tensor
        P = loss_function.material_model.TensorToVoigt(P)
        return P.flatten()
    gp_points,_ = loss_function.fe_element.GetIntegrationData()
    stress_at_gauss = jax.vmap(compute_at_gauss_point,in_axes=(0,))(gp_points)
    return stress_at_gauss
    
def compute_nodal_stress_from_gauss(total_stress: jnp.ndarray, element_nodes: np.ndarray, num_nodes: int) -> jnp.ndarray:
    """
    Args:
        total_stress: jnp.ndarray of shape (num_elements, 4, 3) - stress at Gauss points
        element_nodes: np.ndarray of shape (num_elements, 4) - node indices per element
        num_nodes: total number of nodes in the mesh

    Returns:
        nodal_stress: jnp.ndarray of shape (num_nodes, 3) - averaged stress per node
    """
    # Accumulators
    stress_accumulator = np.zeros((num_nodes, 6), dtype=np.float32)
    count_accumulator = np.zeros((num_nodes,), dtype=np.int32)

    num_elements, num_gauss, _ = total_stress.shape

    # For each element
    for e in range(num_elements):
        nodes = element_nodes[e]  # (4,) node indices
        gauss_stresses = total_stress[e]  # (4, 3)

        # Assuming Gauss point i corresponds to node i (common in reduced integration)
        for i in range(4):
            node_id = nodes[i]
            stress_accumulator[node_id] += np.array(gauss_stresses[i])
            count_accumulator[node_id] += 1

    # Avoid divide-by-zero
    count_accumulator[count_accumulator == 0] = 1
    nodal_stress = stress_accumulator / count_accumulator[:, None]  # shape: (num_nodes, 3)
    return jnp.array(nodal_stress)


def get_stress(loss_function:FiniteElementLoss, disp_field_vec:np.ndarray, K_matrix:np.ndarray):
    element_nodes = loss_function.fe_mesh.GetElementsNodes(loss_function.element_type)
    total_control_vars = K_matrix.reshape(-1,1)

    # @partial(jit, static_argnums=(0,))
    def ComputeElementStressVmapCompatible(element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
            return ComputeElementStressAtGauss(loss_function,
                                               xyz[elements_nodes[element_id],:],
                                            full_control_vector[elements_nodes[element_id]],
                                            full_dof_vector[((loss_function.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                            jnp.arange(loss_function.number_dofs_per_node))].reshape(-1,1))

    # @partial(jit)
    def ComputeElementsStresses(total_control_vars:jnp.array,total_primal_vars:jnp.array):
        # parallel calculation of energies
        return jax.vmap(ComputeElementStressVmapCompatible,(0,None,None,None,None)) \
                        (loss_function.fe_mesh.GetElementsIds(loss_function.element_type),
                        loss_function.fe_mesh.GetElementsNodes(loss_function.element_type),
                        loss_function.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars)
    # @partial(jit)
    def ComputeTotalStress(total_control_vars:jnp.array,total_primal_vars:jnp.array):
        return ComputeElementsStresses(total_control_vars.reshape(-1,1),total_primal_vars)
    

    total_stress = ComputeTotalStress(jnp.array(total_control_vars),jnp.array(disp_field_vec))
    num_nodes = loss_function.fe_mesh.GetNumberOfNodes()
    nodal_stress = compute_nodal_stress_from_gauss(total_stress, element_nodes, num_nodes)

    return nodal_stress.flatten()