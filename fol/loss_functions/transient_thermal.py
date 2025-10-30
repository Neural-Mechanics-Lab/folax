"""
 Authors: Yusuke Yamazaki
          Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2025
 License: FOL/LICENSE
"""
from  .thermal import ThermalLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *

class TransientThermalLoss(ThermalLoss):

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 

        self.default_material_settings = {"rho":1.0,"cp":1.0,"k0":np.ones((self.fe_mesh.GetNumberOfNodes())),"beta":0.0,"c":1.0}
        self.default_time_integration_settings = {"method":"implicit-euler","time_step":None}

        if "material_dict" in self.loss_settings.keys():
            self.material_settings = UpdateDefaultDict(self.default_material_settings,self.loss_settings["material_dict"])
            if "k0" in self.loss_settings["material_dict"].keys():
                input_k0_shape = self.loss_settings["material_dict"]["k0"].shape
                if  input_k0_shape != self.default_material_settings["k0"].shape:
                    fol_error(f"provided k0({input_k0_shape}) in the material_dict does not match the mesh with {self.fe_mesh.GetNumberOfNodes()} nodes !")
                else:
                    self.material_settings["k0"] = jnp.array(self.material_settings["k0"])
        else:
            self.material_settings = self.default_material_settings

        self.time_integration_settings = UpdateDefaultDict(self.default_time_integration_settings,self.loss_settings["time_integration_dict"])
        if self.time_integration_settings["time_step"] == None:
            fol_error("time step should be provided in the time_integration_dict ")
    
    def ComputeElement(self,xyze,Te_c,Te_n,Ke):
        Te_c = Te_c.reshape(-1,1)
        Te_n = Te_n.reshape(-1,1)
        Ke = Ke.reshape(-1,1)

        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            T_at_gauss_n = jnp.dot(N_vec.reshape(1,-1), Te_n)
            T_at_gauss_c = jnp.dot(N_vec.reshape(1,-1), Te_c)
            K_at_gauss = jnp.dot(N_vec, Ke.squeeze()) * (1 + 
                                    self.material_settings["beta"]*(T_at_gauss_n)**self.thermal_loss_settings["c"])
            DN_DX = self.fe_element.ShapeFunctionsLocalGradients(gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B_mat = jnp.dot(invJ,DN_DX.T)
            gp_stiffness = B_mat.T @ B_mat*K_at_gauss * detJ * gp_weight
            gp_mass = self.material_settings["rho"] * self.material_settings["cp"]* jnp.outer(N_vec, N_vec) * detJ * gp_weight 
            gp_t = self.material_settings["rho"] * self.material_settings["cp"] * 0.5/(self.time_integration_settings["time_step"])*gp_weight  * detJ *(T_at_gauss_n-T_at_gauss_c)**2
            dk_dT = jnp.dot(N_vec, Ke.squeeze()) * self.material_settings["beta"] * self.thermal_loss_settings["c"] * T_at_gauss_n ** (self.thermal_loss_settings["c"] - 1)
            gp_dR = (dk_dT * jnp.outer(N_vec, (B_mat@Te_n).T@B_mat) + K_at_gauss *B_mat.T @ B_mat)* detJ * gp_weight 
            return gp_stiffness,gp_mass, gp_t, gp_dR

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,m_gps,t_gps, dR_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Se_dR = jnp.sum(dR_gps, axis=0)
        Te = jnp.sum(t_gps) 
        element_residuals = jax.lax.stop_gradient((Me+self.time_integration_settings["time_step"]*Se)@Te_n - Me@Te_c)
        # element_weighted_residual_loss  = ((Te_n.T @ element_residuals)[0,0])
        return  0.5*Te_n.T@Se@Te_n + Te, (Me+self.time_integration_settings["time_step"]*Se)@Te_n - Me@Te_c, (Me+self.time_integration_settings["time_step"]*Se_dR)

    def ComputeElementEnergy(self,
                             elem_xyz:jnp.array,
                             elem_current_temps:jnp.array,
                             elem_next_temps:jnp.array,
                             elem_heterogeneity:jnp.array) -> float:
        return self.ComputeElement(elem_xyz,elem_current_temps,elem_next_temps,elem_heterogeneity)[0]
    
    def ComputeElementEnergyVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           nodal_current_temps:jnp.array,
                                           nodal_next_temps:jnp.array,
                                           nodal_heterogeneity:jnp.array):
        return self.ComputeElementEnergy(xyz[elements_nodes[element_id],:],
                                         nodal_current_temps[elements_nodes[element_id]],
                                         nodal_next_temps[elements_nodes[element_id]],
                                         nodal_heterogeneity[elements_nodes[element_id]])

    def ComputeElementsEnergies(self,nodal_current_temps:jnp.array,nodal_next_temps:jnp.array):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None)) \
                        (self.fe_mesh.GetElementsIds(self.element_type),
                        self.fe_mesh.GetElementsNodes(self.element_type),
                        self.fe_mesh.GetNodesCoordinates(),
                        nodal_current_temps,
                        nodal_next_temps,
                        self.material_settings["k0"])
    
    def ComputeElementResidualAndJacobian(self,
                                          elem_xyz:jnp.array,
                                          elem_current_temps:jnp.array,
                                          elem_next_temps:jnp.array,
                                          elem_heterogeneity:jnp.array,
                                          elem_BC:jnp.array,
                                          elem_mask_BC:jnp.array,
                                          transpose_jac:bool):
        _,re,ke = self.ComputeElement(elem_xyz,elem_current_temps,elem_next_temps,elem_heterogeneity)

       # Convert transpose_jac (bool) to an integer index (0 = False, 1 = True)
        index = jnp.asarray(transpose_jac, dtype=jnp.int32)

        # Define the two branches for switch
        branches = [
            lambda _: ke,                  # Case 0: No transpose
            lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
        ]

        # Apply the switch operation
        ke = jax.lax.switch(index, branches, None)

        return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)

    def ComputeElementResidualAndJacobianVmapCompatible(self,element_id:jnp.integer,
                                                        elements_nodes:jnp.array,
                                                        xyz:jnp.array,
                                                        nodal_current_temps:jnp.array,
                                                        nodal_next_temps:jnp.array,
                                                        full_dirichlet_BC_vec:jnp.array,
                                                        full_mask_dirichlet_BC_vec:jnp.array,
                                                        transpose_jac:bool):
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      nodal_current_temps[elements_nodes[element_id]],
                                                      nodal_next_temps[elements_nodes[element_id]],
                                                      self.material_settings["k0"][elements_nodes[element_id]],
                                                      full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      transpose_jac)
class TransientThermalLoss3DTetra(TransientThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"tetra"},fe_mesh)
        
class TransientThermalLoss3DHexa(TransientThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"hexahedron"},fe_mesh)

class TransientThermalLoss2DQuad(TransientThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"quad"},fe_mesh)
        
class TransientThermalLoss2DTri(TransientThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"triangle"},fe_mesh)