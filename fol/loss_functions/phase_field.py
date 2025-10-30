"""
 Authors:   Yusuke Yamazaki 
            Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class AllenCahnLoss(FiniteElementLoss):

    def Initialize(self) -> None:  
        super().Initialize() 
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        if self.dim == 2:
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.epsilon =  self.loss_settings["material_dict"]["epsilon"]
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
        else:
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.epsilon =  self.loss_settings["material_dict"]["epsilon"]
            self.body_force = jnp.zeros((3,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
    
    def ComputeElement(self,xyze,phi_e_c,phi_e_n,body_force=0):
        phi_e_c = phi_e_c.reshape(-1,1)
        phi_e_n = phi_e_n.reshape(-1,1)

        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            DN_DX = self.fe_element.ShapeFunctionsLocalGradients(gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B_mat = jnp.dot(invJ,DN_DX.T)
            phi_at_gauss_n = jnp.dot(N_vec.reshape(1,-1), phi_e_n)
            phi_at_gauss_c = jnp.dot(N_vec.reshape(1,-1), phi_e_c)
            source_term = 0.25*(phi_at_gauss_n*phi_at_gauss_n - 1)**2
            Dsource_term = (phi_at_gauss_n*phi_at_gauss_n - 1)*phi_at_gauss_n
            gp_stiffness =  B_mat.T@B_mat * detJ * gp_weight
            gp_mass = jnp.outer(N_vec, N_vec) * detJ * gp_weight 
            gp_f_res = N_vec.reshape(-1,1)*Dsource_term * detJ * gp_weight
            gp_f = source_term * detJ * gp_weight
            gp_t = 0.5/(self.dt)*gp_weight  * detJ *(phi_at_gauss_n-phi_at_gauss_c)**2
            gp_Df = jnp.outer(N_vec, N_vec) * (3 * phi_at_gauss_n**2 - 1) *  detJ * gp_weight
            return gp_stiffness,gp_mass, gp_f, gp_f_res, gp_t, gp_Df

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,m_gps,f_gps,f_res_gps,t_gps, df_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps)
        Fe_res = jnp.sum(f_res_gps,axis=0)
        Te = jnp.sum(t_gps)
        dFe = jnp.sum(df_gps, axis=0)

        return 0.5*phi_e_n.T@Se@phi_e_n + 1/(self.epsilon**2)*Fe + Te, ((Me+self.dt*Se)@phi_e_n - (Me@phi_e_c- 1/(self.epsilon**2)*self.dt*Fe_res)), (Me+self.dt*Se - self.dt/(self.epsilon**2)*dFe)

class AllenCahnLoss2DQuad(AllenCahnLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Phi"],  
                               "element_type":"quad"},fe_mesh)
        
class AllenCahnLoss2DTri(AllenCahnLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Phi"],  
                               "element_type":"triangle"},fe_mesh)
class AllenCahnLoss3DHexa(AllenCahnLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Phi"],   
                               "element_type":"hexahedron"},fe_mesh)

