#
# Authors: Reza Najian Asl, https://github.com/RezaNajian
# Date: July, 2024
# License: FOL/LICENSE
#

"""
fem_utilities.py
A module for performing basic finite element operations.
"""
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax

class ConstantsMeta(type):
    def __setattr__(self, key, value):
        raise AttributeError("Cannot modify a constant value")

class GaussQuadrature(metaclass=ConstantsMeta):

    """
    Gauss Quadrature class for integration.

    """

    @property
    def one_point_GQ(self):
        # 1-point Gauss quadrature
        points = jnp.array([0.0])
        weights = jnp.array([2.0])
        return points,weights

    @property
    def two_point_GQ(self):
        # 2-point Gauss quadrature
        points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
        weights = jnp.array([1.0, 1.0])
        return points,weights
    
    @property
    def three_point_GQ(self):
        # 3-point Gauss quadrature
        points = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)])
        weights = jnp.array([5/9, 8/9, 5/9])
        return points,weights
    
    @property
    def four_point_GQ(self):
        # 4-point Gauss quadrature
        points = jnp.array([-jnp.sqrt((3+2*jnp.sqrt(6/5))/7), -jnp.sqrt((3-2*jnp.sqrt(6/5))/7), jnp.sqrt((3-2*jnp.sqrt(6/5))/7), jnp.sqrt((3+2*jnp.sqrt(6/5))/7)])
        weights = jnp.array([(18-jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18-jnp.sqrt(30))/36])
        return points,weights


class ShapeFunction:
    """
    Base class for shape functions of finite elements.
    """

    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float, optional): Local coordinate in the zeta direction for 3D elements.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float, optional): Local coordinate in the zeta direction for 3D elements.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class QuadShapeFunction(ShapeFunction):
    """
    Shape functions for a quadrilateral element.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function for a quadrilateral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            (1 - xi) * (1 - eta) / 4,
            (1 + xi) * (1 - eta) / 4,
            (1 + xi) * (1 + eta) / 4,
            (1 - xi) * (1 + eta) / 4
        ])
        return N
    
    @partial(jit, static_argnums=(0,))
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions for a quadrilateral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-(1 - eta) / 4, -(1 - xi) / 4],
            [(1 - eta) / 4, -(1 + xi) / 4],
            [(1 + eta) / 4, (1 + xi) / 4],
            [-(1 + eta) / 4, (1 - xi) / 4]
        ])
        return dN_dxi

class TriangleShapeFunction(ShapeFunction):
    """
    Shape functions for a triangular element.
    """

    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function for a triangular element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            1 - xi - eta,
            xi,
            eta
        ])
        return N
    
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions for a triangular element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-1, -1],
            [1, 0],
            [0, 1]
        ])
        return dN_dxi

class TetrahedralShapeFunction(ShapeFunction):
    """
    Shape functions for a tetrahedral element.
    """

    def evaluate(self, xi, eta, zeta):
        """
        Evaluate the shape function for a tetrahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            1 - xi - eta - zeta,
            xi,
            eta,
            zeta
        ])
        return N
    
    def derivatives(self, xi, eta, zeta):
        """
        Evaluate the derivatives of the shape functions for a tetrahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        return dN_dxi

class HexahedralShapeFunction(ShapeFunction):
    """
    Shape functions for a hexahedral element.
    """

    def evaluate(self, xi, eta, zeta):
        """
        Evaluate the shape function for a hexahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            (1 - xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 + eta) * (1 + zeta) / 8,
            (1 - xi) * (1 + eta) * (1 + zeta) / 8
        ])
        return N
    
    def derivatives(self, xi, eta, zeta):
        """
        Evaluate the derivatives of the shape functions for a hexahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-(1 - eta) * (1 - zeta) / 8, -(1 - xi) * (1 - zeta) / 8, -(1 - xi) * (1 - eta) / 8],
            [(1 - eta) * (1 - zeta) / 8, -(1 + xi) * (1 - zeta) / 8, -(1 + xi) * (1 - eta) / 8],
            [(1 + eta) * (1 - zeta) / 8, (1 + xi) * (1 - zeta) / 8, -(1 + xi) * (1 + eta) / 8],
            [-(1 + eta) * (1 - zeta) / 8, (1 - xi) * (1 - zeta) / 8, -(1 - xi) * (1 + eta) / 8],
            [-(1 - eta) * (1 + zeta) / 8, -(1 - xi) * (1 + zeta) / 8, (1 - xi) * (1 - eta) / 8],
            [(1 - eta) * (1 + zeta) / 8, -(1 + xi) * (1 + zeta) / 8, (1 + xi) * (1 - eta) / 8],
            [(1 + eta) * (1 + zeta) / 8, (1 + xi) * (1 + zeta) / 8, (1 + xi) * (1 + eta) / 8],
            [-(1 + eta) * (1 + zeta) / 8, (1 - xi) * (1 + zeta) / 8, (1 - xi) * (1 + eta) / 8]
        ])
        return dN_dxi

class MaterialModel:
    """
    Base class for Material models.
    """

    def evaluate(self, F, *args, **kwargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def fourth_order_identity_tensor(self, dim=3):
        """
        Calculate fourth identity matrix
        """
        eye = jnp.eye(dim)
        I4 = jnp.einsum('ik,jl->ijkl', eye, eye)
        return I4
    
    def diad_special(self, A, B, dim):
        """
        Calculate a specific tensor diad: Cijkl = (1/2)*(A[i,k] * B[j,l] + A[i,l] * B[j,k])
        """
        P = 0.5* (jnp.einsum('ik,jl->ijkl',A,B) + jnp.einsum('il,jk->ijkl',A,B))
        return P
    
    def TensorToVoigt(self, tensor):
        """
        Convert a tensor to a vector
        """
        if tensor.size == 4:
            voigt = jnp.zeros((3,1))
            voigt = voigt.at[0,0].set(tensor[0,0])
            voigt = voigt.at[1,0].set(tensor[1,1])
            voigt = voigt.at[2,0].set(tensor[0,1])
            return voigt
        elif tensor.size == 9:
            voigt = jnp.zeros((6,1))
            voigt = voigt.at[0,0].set(tensor[0,0])
            voigt = voigt.at[1,0].set(tensor[1,1])
            voigt = voigt.at[2,0].set(tensor[2,2])
            voigt = voigt.at[3,0].set(tensor[1,2])
            voigt = voigt.at[4,0].set(tensor[0,2])
            voigt = voigt.at[5,0].set(tensor[0,1])
            return voigt
        
    def VoigtToTensor(self, voigt):
        """
        Convert a tensor to a vector
        """
        if voigt.size == 3:
            tensor = jnp.zeros((2,2))
            tensor = tensor.at[0,0].set(voigt[0,0])
            tensor = tensor.at[1,1].set(voigt[1,0])
            tensor = tensor.at[0,1].set(voigt[2,0])
            tensor = tensor.at[1,0].set(voigt[2,0])
            return tensor
        elif voigt.size == 6:
            tensor = jnp.zeros((3,3))
            tensor = tensor.at[0,0].set(voigt[0,0])
            tensor = tensor.at[1,1].set(voigt[1,0])
            tensor = tensor.at[2,2].set(voigt[2,0])
            tensor = tensor.at[1,2].set(voigt[3,0])
            tensor = tensor.at[2,1].set(voigt[3,0])
            tensor = tensor.at[0,2].set(voigt[4,0])
            tensor = tensor.at[2,0].set(voigt[4,0])
            tensor = tensor.at[0,1].set(voigt[5,0])
            tensor = tensor.at[1,0].set(voigt[5,0])
            return tensor

    def FourthTensorToVoigt(self,Cf):
        """
        Convert a fouth-order tensor to a second-order tensor
        """
        if Cf.size == 16:
            C = jnp.zeros((3,3))
            C = C.at[0,0].set(Cf[0,0,0,0])
            C = C.at[0,1].set(Cf[0,0,1,1])
            C = C.at[0,2].set(Cf[0,0,0,1])
            C = C.at[1,0].set(C[0,1])
            C = C.at[1,1].set(Cf[1,1,1,1])
            C = C.at[1,2].set(Cf[1,1,0,1])
            C = C.at[2,0].set(C[0,2])
            C = C.at[2,1].set(C[1,2])
            C = C.at[2,2].set(Cf[0,1,0,1])
            return C
        elif Cf.size == 81: 
            C = jnp.zeros((6, 6))
            indices = [
                (0, 0), (1, 1), (2, 2), 
                (1, 2), (0, 2), (0, 1)
                ]
            
            for I, (i, j) in enumerate(indices):
                for J, (k, l) in enumerate(indices):
                    C = C.at[I, J].set(Cf[i, j, k, l])
            
            return C


class NeoHookianModel(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu, *args, **keyargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        ph = 0.5*k*(J-(1/J))
        #dp_dJ = (k/4)*(2 + 2*J**(-2))
        dp_dJ = 0.5*k*(1 + J**(-2))

        # Strain Energy
        xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-2/3))*jnp.trace(C)
        xsie_iso = 0.5*mu*(I1_bar - 3)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*ph*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/3))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*ph + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*ph*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/3)*(J**(-2/3))*jnp.vdot(S_bar,C)*P_bar - \
                (2/3)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
        
class NeoHookianModel2D(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu, *args, **keyargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        p = 0.5*k*(J-(1/J))
        dp_dJ = 0.5*k*(1 + J**(-2))

        # Strain Energy
        xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-2/2))*jnp.trace(C)
        xsie_iso = 0.5*mu*(I1_bar - 2)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*p*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/2)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/2))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/2)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*p + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*p*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/2)*(J**(-2/2))*jnp.vdot(S_bar,C)*P_bar - \
                (2/2)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
class SaintVenant(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, lambda_, mu, *args, **keyargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.
        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants
        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:

        E = 0.5*(F.T @ F - jnp.eye(F.shape[0]))
        xsie = 0.5*lambda_*(jnp.linalg.trace(E) ** 2) + mu*jnp.linalg.trace(E @ E)

        I_fourth = self.fourth_order_identity_tensor(F.shape[0])
        C_tangent_fourth = lambda_ * jnp.einsum('ij,kl->jikl',jnp.eye(F.shape[0]),jnp.eye(F.shape[0])) +\
                            2 * mu * I_fourth
        Se = jnp.einsum('ijkl,kl->ij',C_tangent_fourth,E)
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
class NeoHookianModelAD(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, C_mat, k, mu, lambda_, *args, **keyargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """

        def strain_energy(C_voigt):
            C = self.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
            I1_bar = (J**(-2/3))*jnp.trace(C)
            xsie_iso = 0.5*mu*(I1_bar - 3)
            return 0.5*mu*(I1_bar - 3) - mu*jnp.log(J) + (lambda_/2)*(jnp.log(J))**2
        
        def strain_energy_paper(C_voigt):
            C = self.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
            I1_bar = (J**(-2/3))*jnp.trace(C)
            xsie_iso = 0.5*mu*(I1_bar - 3)
            return xsie_vol + xsie_iso
        
        def second_piola(C_voigt):
            return 2*jax.grad(strain_energy_paper)(C_voigt)
        
        def tangent(C_voigt):
            return 2*jax.jacfwd(second_piola)(C_voigt)
        
        C_voigt = self.TensorToVoigt(C_mat)

        xsie = strain_energy_paper(C_voigt)
        Se_voigt = second_piola(C_voigt)
        C_tangent = tangent(C_voigt)

        return xsie, Se_voigt, C_tangent.squeeze()
    
class NeoHookianModel2DAD(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, C_mat, k, mu, lambda_, *args, **keyargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:
        # Strain Energy

        def strain_energy(C_voigt):
            C = self.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            return 0.5*mu*(jnp.linalg.trace(C) - 2) - mu*jnp.log(J) + 0.5*lambda_*(jnp.log(J)**2)
            # return 0.5*mu*((J**(-1))*jnp.linalg.trace(C) - 2) - mu*jnp.log(J) + 0.5*lambda_*(jnp.log(J)**2)
        
        def strain_energy_paper(C_voigt):
            C = self.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            return (k/4)*(J**2 - 2*jnp.log(J) -1) + 0.5*mu*((J**(-2/2))*jnp.trace(C) - 2)
        
        def second_piola(C_voigt):
            return 2*jax.grad(strain_energy)(C_voigt)
        
        def tangent(C_voigt):
            return 2*jax.jacfwd(second_piola)(C_voigt)
        
        # C_mat = jnp.dot(F.T,F)
        C_voigt = self.TensorToVoigt(C_mat)

        xsie = strain_energy(C_voigt)
        Se_voigt = second_piola(C_voigt)
        C_tangent = tangent(C_voigt)
        return xsie, Se_voigt, C_tangent.squeeze()