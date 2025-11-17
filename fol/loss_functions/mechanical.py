"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class MechanicalLoss(FiniteElementLoss):

    def Initialize(self) -> None:  
        super().Initialize() 
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        if self.dim == 2:
            self.CalculateNMatrix = self.CalculateNMatrix2D
            self.CalculateBMatrix = self.CalculateBMatrix2D
            self.D = self.CalculateDMatrix2D(self.loss_settings["material_dict"]["young_modulus"],
                                            self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
        else:
            self.CalculateNMatrix = self.CalculateNMatrix3D
            self.CalculateBMatrix = self.CalculateBMatrix3D
            self.D = self.CalculateDMatrix3D(self.loss_settings["material_dict"]["young_modulus"],
                                            self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((3,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

    def CalculateBMatrix2D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((3, 2 * DN_DX.shape[0]))
        indices = jnp.arange(DN_DX.shape[0])
        B = B.at[0, 2 * indices].set(DN_DX[indices,0])
        B = B.at[1, 2 * indices + 1].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices + 1].set(DN_DX[indices,0])  
        return B

    def CalculateBMatrix3D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((6,3*DN_DX.shape[0]))
        index = jnp.arange(DN_DX.shape[0]) * 3
        B = B.at[0, index + 0].set(DN_DX[:,0])
        B = B.at[1, index + 1].set(DN_DX[:,1])
        B = B.at[2, index + 2].set(DN_DX[:,2])
        B = B.at[3, index + 0].set(DN_DX[:,1])
        B = B.at[3, index + 1].set(DN_DX[:,0])
        B = B.at[4, index + 1].set(DN_DX[:,2])
        B = B.at[4, index + 2].set(DN_DX[:,1])
        B = B.at[5, index + 0].set(DN_DX[:,2])
        B = B.at[5, index + 2].set(DN_DX[:,0])
        return B

    def CalculateDMatrix2D(self,young_modulus:float,poisson_ratio:float) -> jnp.array:
        return jnp.array([[1,poisson_ratio,0],[poisson_ratio,1,0],[0,0,(1-poisson_ratio)/2]]) * (young_modulus/(1-poisson_ratio**2))

    def CalculateDMatrix3D(self,young_modulus:float,poisson_ratio:float) -> jnp.array:
            # construction of the constitutive matrix
            c1 = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            c2 = c1 * (1.0 - poisson_ratio)
            c3 = c1 * poisson_ratio
            c4 = c1 * 0.5 * (1.0 - 2.0 * poisson_ratio)
            D = jnp.zeros((6,6))
            D = D.at[0,0].set(c2)
            D = D.at[0,1].set(c3)
            D = D.at[0,2].set(c3)
            D = D.at[1,0].set(c3)
            D = D.at[1,1].set(c2)
            D = D.at[1,2].set(c3)
            D = D.at[2,0].set(c3)
            D = D.at[2,1].set(c3)
            D = D.at[2,2].set(c2)
            D = D.at[3,3].set(c4)
            D = D.at[4,4].set(c4)
            D = D.at[5,5].set(c4)
            return D
    
    def CalculateNMatrix2D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((2, 2 * N_vec.size))
        indices = jnp.arange(N_vec.size)   
        N_mat = N_mat.at[0, 2 * indices].set(N_vec)
        N_mat = N_mat.at[1, 2 * indices + 1].set(N_vec)    
        return N_mat
    
    def CalculateNMatrix3D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((3,3*N_vec.size))
        N_mat = N_mat.at[0,0::3].set(N_vec)
        N_mat = N_mat.at[1,1::3].set(N_vec)
        N_mat = N_mat.at[2,2::3].set(N_vec)
        return N_mat

    def ComputeElement(self,xyze,de,uvwe):

        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            e_at_gauss = jnp.dot(N_vec, de.squeeze())
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            gp_stiffness = gp_weight * detJ * e_at_gauss * (B_mat.T @ (self.D @ B_mat))
            gp_f = gp_weight * detJ * (N_mat.T @ self.body_force)
            return gp_stiffness,gp_f

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ uvwe - Fe)
        return  ((uvwe.T @ element_residuals)[0,0]), (Se @ uvwe - Fe), Se

class MechanicalLoss3DTetra(MechanicalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],  
                               "element_type":"tetra"},fe_mesh)

class MechanicalLoss3DHexa(MechanicalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],  
                               "element_type":"hexahedron"},fe_mesh)

class MechanicalLoss2DTri(MechanicalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],  
                               "element_type":"triangle"},fe_mesh)

class MechanicalLoss2DQuad(MechanicalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],  
                               "element_type":"quad"},fe_mesh)
