"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Sep, 2025
 License: FOL/LICENSE
"""
try:
    import fol_ffi_functions
    _HAS_FOL_FFI_LIB = True
except ImportError:
    _HAS_FOL_FFI_LIB = False

from  fol.loss_functions.fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from jax.experimental import sparse

if _HAS_FOL_FFI_LIB:
    from fol_ffi_functions import kr_small_displacement_element
    jax.ffi.register_ffi_target("compute_nodal_residuals", kr_small_displacement_element.compute_nodal_residuals(), platform="CUDA")
    jax.ffi.register_ffi_type_id("compute_nodal_residuals", kr_small_displacement_element.type_id(), platform="CUDA")
    jax.ffi.register_ffi_target("compute_elements", kr_small_displacement_element.compute_elements(), platform="CUDA")

class KratosSmallDisplacement3DTetra(FiniteElementLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not _HAS_FOL_FFI_LIB:
            fol_error(" fol_ffi_functions is not available, install by running install script under ffi_functions folder!")
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],  
                               "element_type":"tetra"},fe_mesh)

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize()

        self.material_settings = {"poisson_ratio":0.3,"young_modulus":1.0}
        if "material_dict" in self.loss_settings.keys():
            self.material_settings = self.loss_settings["material_dict"]

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,te,body_force=0):
        fol_error(" is not implemented for KratosSmallDisplacement3DTetra!")

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        total_primal_vars = total_primal_vars.reshape(1,-1)
        batch_size = 1
        nodal_res_type = jax.ShapeDtypeStruct((batch_size,self.number_dofs_per_node*self.fe_mesh.GetNumberOfNodes()), total_primal_vars.dtype)
        nodal_res = jax.ffi.ffi_call("compute_nodal_residuals", nodal_res_type, vmap_method="legacy_vectorized")(self.fe_mesh.GetNodesCoordinates(),
                                                                                self.fe_mesh.GetElementsNodes(self.element_type),
                                                                                jnp.array([self.material_settings["poisson_ratio"],
                                                                                           self.material_settings["young_modulus"]]),total_primal_vars)
        return (total_primal_vars @ nodal_res.T)[0,0]

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeJacobianMatrixAndResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array,transpose_jacobian:bool=False):

        BC_vector = jnp.ones((self.total_number_of_dofs))
        BC_vector = BC_vector.at[self.dirichlet_indices.astype(jnp.int32)].set(0)
        mask_BC_vector = jnp.zeros((self.total_number_of_dofs))
        mask_BC_vector = mask_BC_vector.at[self.dirichlet_indices.astype(jnp.int32)].set(1)

        total_num_elements = self.fe_mesh.GetNumberOfElements(self.element_type)
        num_nodes_per_elem = len(self.fe_mesh.GetElementsNodes(self.element_type)[0])
        element_matrix_size = self.number_dofs_per_node * num_nodes_per_elem

        lhs_type = jax.ShapeDtypeStruct((total_num_elements,element_matrix_size,element_matrix_size), total_primal_vars.dtype)
        rhs_type = jax.ShapeDtypeStruct((total_num_elements,element_matrix_size), total_primal_vars.dtype)
        lhs,res = jax.ffi.ffi_call("compute_elements", (lhs_type, rhs_type))(self.fe_mesh.GetNodesCoordinates(),
                                                                            self.fe_mesh.GetElementsNodes(self.element_type),
                                                                            jnp.array([self.material_settings["poisson_ratio"],
                                                                                        self.material_settings["young_modulus"]]),
                                                                            total_primal_vars)
        @jax.jit
        def Proccess(ke:jnp.array,
                     re:jnp.array,
                     elem_BC:jnp.array,
                     elem_mask_BC:jnp.array):
                     
            index = jnp.asarray(transpose_jacobian, dtype=jnp.int32)
            # Define the two branches for switch
            branches = [
                lambda _: ke,                  # Case 0: No transpose
                lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
            ]
            # Apply the switch operation
            ke = jax.lax.switch(index, branches, None)
            return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)

        @jax.jit
        def ProccessVmapCompatible(element_id:jnp.integer,
                                    elements_nodes:jnp.array,
                                    elements_stiffness:jnp.array,
                                    elements_residuals:jnp.array,
                                    full_dirichlet_BC_vec:jnp.array,
                                    full_mask_dirichlet_BC_vec:jnp.array):

            return Proccess(elements_stiffness[element_id],
                                 elements_residuals[element_id],
                                full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

        elements_residuals, elements_stiffness = jax.vmap(ProccessVmapCompatible, (0,None,None,None,None,None))(self.fe_mesh.GetElementsIds(self.element_type),
                                                                                                            self.fe_mesh.GetElementsNodes(self.element_type),
                                                                                                            lhs,
                                                                                                            res,
                                                                                                            BC_vector,
                                                                                                            mask_BC_vector)


        # first compute the global residual vector
        residuals_vector = jnp.zeros((self.total_number_of_dofs))
        for dof_idx in range(self.number_dofs_per_node):
            residuals_vector = residuals_vector.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)+dof_idx].add(jnp.squeeze(elements_residuals[:,dof_idx::self.number_dofs_per_node]))

        # second compute the global jacobian matrix  
        jacobian_data = jnp.ravel(elements_stiffness)
        jacobian_indices = jax.vmap(self.ComputeElementJacobianIndices)(self.fe_mesh.GetElementsNodes(self.element_type)) # Get the indices
        jacobian_indices = jacobian_indices.reshape(-1,2)
        
        sparse_jacobian = sparse.BCOO((jacobian_data,jacobian_indices),shape=(self.total_number_of_dofs,self.total_number_of_dofs))
        
        return sparse_jacobian, residuals_vector
