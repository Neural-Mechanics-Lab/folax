"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *

class ThermalLoss(FiniteElementLoss):

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 
        self.thermal_loss_settings = {"beta":0,"c":1}
        if "beta" in self.loss_settings.keys():
            self.thermal_loss_settings["beta"] = self.loss_settings["beta"]
        if "c" in self.loss_settings.keys():
            self.thermal_loss_settings["c"] = self.loss_settings["c"]

    def ComputeElement(self,xyze,de,te,body_force=0):

        def compute_at_gauss_point(gp_point,gp_weight,te):
            te = jax.lax.stop_gradient(te)
            N = self.fe_element.ShapeFunctionsValues(gp_point)
            conductivity_at_gauss = jnp.dot(N, de.squeeze()) * (1 + 
                                    self.thermal_loss_settings["beta"]*(jnp.dot(N,te.squeeze()))**self.thermal_loss_settings["c"])
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            gp_stiffness = conductivity_at_gauss * (DN_DX @ DN_DX.T) * detJ * gp_weight
            gp_f = gp_weight * detJ * body_force *  N.reshape(-1,1) 
            return gp_stiffness,gp_f
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0,None))(gp_points,gp_weights,te)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        def compute_elem_res(Se,te ,Fe):
            te = jax.lax.stop_gradient(te)
            return (Se @ te - Fe)
        element_residuals = compute_elem_res(Se,te ,Fe)
        return  ((te.T @ element_residuals)[0,0]), (Se @ te - Fe), Se

class ThermalLoss3DTetra(ThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"tetra"},fe_mesh)
        
class ThermalLoss3DHexa(ThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"hexahedron"},fe_mesh)

class ThermalLoss2DQuad(ThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"quad"},fe_mesh)
        
class ThermalLoss2DTri(ThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"triangle"},fe_mesh)