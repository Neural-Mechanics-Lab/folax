"""
 Authors: Rishabh Arora, https://github.com/rishabharora236-cell
 Date: Oct, 2025
 License: FOL/LICENSE
"""
from  .mechanical import MechanicalLoss
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh
from fol.constitutive_material_models.plasticity import J2Plasticity 

class ElastoplasticityLoss(MechanicalLoss):

    def Initialize(self) -> None:  

        super().Initialize() 
        mat_dict = self.loss_settings["material_dict"]

        self.material_model = J2Plasticity(
            E=mat_dict["young_modulus"],
            nu=mat_dict["poisson_ratio"],
            yield_stress=mat_dict["yield_limit"],
            hardening_modulus=mat_dict["iso_hardening_parameter_1"],
            hardening_exponent=mat_dict["iso_hardening_param_2"] # Use .get() for safety
        )

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uvwe,element_state_gps):
        @jit
        def compute_at_gauss_point(gp_point,gp_weight,gp_state_vector,uvwe):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            strain_gp = B_mat @ uvwe
            # Determine dimensionality based on strain vector length
            n_strain_components = strain_gp.shape[0]
            
            # Construct strain matrix based on dimensionality
            if n_strain_components == 3:  # 2D case (ε_xx, ε_yy, γ_xy)
                strain_matrix = jnp.array([
                    [strain_gp[0], strain_gp[2]],  # [ε_xx, ε_xy]
                    [strain_gp[2], strain_gp[1]]   # [ε_xy, ε_yy]
                ])
            elif n_strain_components == 6:  # 3D case (ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz)
                strain_matrix = jnp.array([
                    [strain_gp[0], strain_gp[3], strain_gp[5]],  # [ε_xx, ε_xy, ε_xz]
                    [strain_gp[3], strain_gp[1], strain_gp[4]],  # [ε_xy, ε_yy, ε_yz]
                    [strain_gp[5], strain_gp[4], strain_gp[2]]   # [ε_xz, ε_yz, ε_zz]
                ])
            strain_matrix_array= strain_matrix.squeeze()
            stress_gp_v,gp_state_up = self.material_model.evaluate(strain_matrix_array, gp_state_vector)
            stress_gp_v = stress_gp_v.reshape(n_strain_components,1)
            gp_f_int = (gp_weight * detJ * (B_mat.T @ stress_gp_v))
            #gp_stiffness = gp_weight * detJ * (B_mat.T @ (tgMM @ B_mat)) use when computing tangent at gauss point level
            gp_f_body = (gp_weight * detJ * (N_mat.T @ self.body_force))
            
            return gp_f_body,gp_f_int,gp_state_up

        gp_points,gp_weights = self.fe_element.GetIntegrationData()

        f_gps,f_gps_int,gps_state = jax.vmap(compute_at_gauss_point,in_axes=(0,0,0,None))(gp_points,gp_weights,element_state_gps,uvwe)
        #Se = jnp.sum(k_gps, axis=0, keepdims=False) use when computing tangent at gauss point level
        Fe = jnp.sum(f_gps, axis=0, keepdims=False)
        Fe_int= jnp.sum(f_gps_int, axis=0)
        residual = (Fe_int-Fe)
        
        #Compute element tangent 
        def compute_residual_flat(u_flat):
            u = u_flat.reshape(-1, 1)  # Reshape back to column vector
            f_gps, f_gps_int, _ = jax.vmap(
                compute_at_gauss_point, 
                in_axes=(0, 0, 0, None)
            )(gp_points, gp_weights, element_state_gps, u)
            
            Fe = jnp.sum(f_gps, axis=0, keepdims=False)
            Fe_int = jnp.sum(f_gps_int, axis=0)
            residual = (Fe_int - Fe)
            return residual.flatten()  # Return as 1D array
        
        # Compute residual
        uvwe_flat = uvwe.flatten()
        residual_flat = compute_residual_flat(uvwe_flat)
        residual = residual_flat.reshape(-1, 1)
        
        # Compute tangent stiffness using automatic differentiation
        Se = jax.jacfwd(compute_residual_flat)(uvwe_flat)
        element_residuals = jax.lax.stop_gradient(residual)
        return  ((uvwe.T @ element_residuals)[0,0]),gps_state, residual, Se
    
    def ComputeElementResidualAndJacobian(
        self,
        elem_xyz: jnp.array,
        elem_controls: jnp.array,
        elem_dofs: jnp.array,
        elem_BC: jnp.array,
        elem_mask_BC: jnp.array,
        transpose_jac: bool,
        elem_state_gps: jnp.array
    ):
        """
        Compute element residual and jacobian, with optional state update.
        """

        _, elem_state_up_gps, re, ke = self.ComputeElement(
            elem_xyz,
            elem_controls,
            elem_dofs,
            elem_state_gps
        )

        index = jnp.asarray(transpose_jac, dtype=jnp.int32)

        # Define the two branches for switch
        branches = [
            lambda _: ke,                  # Case 0: No transpose
            lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
        ]

        # Apply the switch operation
        ke = jax.lax.switch(index, branches, None)

        # Apply Dirichlet boundary conditions
        r_e, k_e = self.ApplyDirichletBCOnElementResidualAndJacobian(re, ke, elem_BC, elem_mask_BC)

        return r_e, k_e, elem_state_up_gps

    def ComputeElementResidualAndJacobianVmapCompatible(self,element_id:jnp.integer,
                                                        elements_nodes:jnp.array,
                                                        xyz:jnp.array,
                                                        full_control_vector:jnp.array,
                                                        full_dof_vector:jnp.array,
                                                        full_dirichlet_BC_vec:jnp.array,
                                                        full_mask_dirichlet_BC_vec:jnp.array,
                                                        transpose_jac:bool,
                                                        full_state_gps: jnp.array):
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      full_control_vector[elements_nodes[element_id]],
                                                      full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      transpose_jac,
                                                      full_state_gps[element_id, :, :])

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeJacobianMatrixAndResidualVector(
        self,
        total_control_vars: jnp.array,
        total_primal_vars: jnp.array,
        old_state_gps: jnp.array,
        transpose_jacobian: bool = False
    ):
        BC_vector = jnp.ones((self.total_number_of_dofs))
        BC_vector = BC_vector.at[self.dirichlet_indices].set(0)
        mask_BC_vector = jnp.zeros((self.total_number_of_dofs))
        mask_BC_vector = mask_BC_vector.at[self.dirichlet_indices].set(1)

        num_nodes_per_elem = len(self.fe_mesh.GetElementsNodes(self.element_type)[0])
        element_matrix_size = self.number_dofs_per_node * num_nodes_per_elem
        elements_jacobian_flat = jnp.zeros(
            self.fe_mesh.GetNumberOfElements(self.element_type) * element_matrix_size * element_matrix_size
        )

        template_element_indices = jnp.arange(0, self.adjusted_batch_size)
        template_elem_res_indices = jnp.arange(0, element_matrix_size, self.number_dofs_per_node)
        template_elem_jac_indices = jnp.arange(0, self.adjusted_batch_size * element_matrix_size * element_matrix_size)

        residuals_vector = jnp.zeros((self.total_number_of_dofs))

        new_state_gps = jnp.zeros_like(old_state_gps)

        @jit
        def fill_arrays(batch_index, batch_arrays):

            glob_res_vec, elem_jac_vec, new_state_buf = batch_arrays

            batch_element_indices = (batch_index * self.adjusted_batch_size) + template_element_indices
            batch_elem_jac_indices = (batch_index * self.adjusted_batch_size * element_matrix_size**2) + template_elem_jac_indices

            element_nodes = self.fe_mesh.GetElementsNodes(self.element_type)
            node_coords = self.fe_mesh.GetNodesCoordinates()

            batch_elements_residuals, batch_elements_stiffness, batch_state_up_gps = jax.vmap(
                self.ComputeElementResidualAndJacobianVmapCompatible, (0, None, None, None, None, None, None, None, None)
            )(
                batch_element_indices,
                element_nodes,
                node_coords,
                total_control_vars,
                total_primal_vars,
                BC_vector,
                mask_BC_vector,
                transpose_jacobian,
                old_state_gps,
            )

            elem_jac_vec = elem_jac_vec.at[batch_elem_jac_indices].set(batch_elements_stiffness.ravel())

            @jit
            def fill_res_vec(dof_idx, glob_res_vec):
                glob_res_vec = glob_res_vec.at[
                    self.number_dofs_per_node * element_nodes[batch_element_indices] + dof_idx
                ].add(jnp.squeeze(batch_elements_residuals[:, template_elem_res_indices + dof_idx]))
                return glob_res_vec

            glob_res_vec = jax.lax.fori_loop(0, self.number_dofs_per_node, fill_res_vec, glob_res_vec)

            new_state_buf = new_state_buf.at[batch_element_indices, :, :].set(batch_state_up_gps)
            return glob_res_vec, elem_jac_vec, new_state_buf

        # Run loop
        residuals_vector, elements_jacobian_flat, new_state_gps = jax.lax.fori_loop(
            0, self.num_element_batches, fill_arrays, (residuals_vector, elements_jacobian_flat, new_state_gps)
        )

        # Assemble sparse Jacobian
        jacobian_indices = jax.vmap(self.ComputeElementJacobianIndices)(
            self.fe_mesh.GetElementsNodes(self.element_type)
        ).reshape(-1, 2)

        sparse_jacobian = sparse.BCOO(
            (elements_jacobian_flat, jacobian_indices),
            shape=(self.total_number_of_dofs, self.total_number_of_dofs),
        )

        return new_state_gps, sparse_jacobian, residuals_vector

class ElastoplasticityLoss2DQuad(ElastoplasticityLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],  
                               "element_type":"quad"},fe_mesh)

class ElastoplasticityLoss3DTetra(ElastoplasticityLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],  
                               "element_type":"tetra"},fe_mesh)