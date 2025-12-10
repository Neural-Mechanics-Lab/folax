"""
Utility functions for constitutive models:
- Tensor operations (deviatoric, invariants)
"""
import jax.numpy as jnp
from jax import Array
from typing import Tuple,Optional

class TensorVoigtArray:

    
    @staticmethod 
    def TensorToVoigt(tensor):
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
    @staticmethod
    def VoigtToTensor(voigt):
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
    @staticmethod
    def TensorToArray(tensor):
        """
        Convert a tensor to a vector
        """
        if tensor.size == 4:
            voigt = jnp.zeros(3)
            voigt = voigt.at[0].set(tensor[0,0])
            voigt = voigt.at[1].set(tensor[1,1])
            voigt = voigt.at[2].set(tensor[0,1])
            return voigt
        elif tensor.size == 9:
            voigt = jnp.zeros(6)
            voigt = voigt.at[0].set(tensor[0,0])
            voigt = voigt.at[1].set(tensor[1,1])
            voigt = voigt.at[2].set(tensor[2,2])
            voigt = voigt.at[3].set(tensor[0,1])
            voigt = voigt.at[4].set(tensor[1,2])
            voigt = voigt.at[5].set(tensor[0,2])
            return voigt
    @staticmethod
    def ArrayToTensor(voigt):
        """
        Convert a tensor to a vector
        """
        if voigt.size == 3:
            tensor = jnp.zeros((2,2))
            tensor = tensor.at[0,0].set(voigt[0])
            tensor = tensor.at[1,1].set(voigt[1])
            tensor = tensor.at[0,1].set(voigt[2])
            tensor = tensor.at[1,0].set(voigt[2])
            return tensor
        elif voigt.size == 6:
            tensor = jnp.zeros((3,3))
            tensor = tensor.at[0,0].set(voigt[0])
            tensor = tensor.at[1,1].set(voigt[1])
            tensor = tensor.at[2,2].set(voigt[2])
            tensor = tensor.at[1,2].set(voigt[4])
            tensor = tensor.at[2,1].set(voigt[4])
            tensor = tensor.at[0,2].set(voigt[5])
            tensor = tensor.at[2,0].set(voigt[5])
            tensor = tensor.at[0,1].set(voigt[3])
            tensor = tensor.at[1,0].set(voigt[3])
            return tensor
    
    @staticmethod
    def FourthTensorToVoigt(Cf):
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

class TensorOperations:
    """Common tensor operations for constitutive models"""
    
    @staticmethod
    def trace(tensor: Array) -> float:
        """Compute trace of tensor"""
        return jnp.trace(tensor)
    
    @staticmethod
    def deviatoric(tensor: Array) -> Array:
        """
        Compute deviatoric part of tensor.
        
        Args:
            tensor: Symmetric tensor (any dimension)
            
        Returns:
            Deviatoric tensor: dev(A) = A - (1/3)tr(A)I
        """
        dim = tensor.shape[0]
        I = jnp.eye(dim, dtype=tensor.dtype)
        return tensor - I * (TensorOperations.trace(tensor) / dim)
    
    @staticmethod
    def frobenius_norm(tensor: Array) -> float:
        """
        Compute Frobenius norm: ||A|| = sqrt(A:A)
        """
        return jnp.sqrt(jnp.tensordot(tensor, tensor, axes=2))
    
    @staticmethod
    def von_mises_stress(stress_tensor: Array) -> float:
        """
        Compute von Mises equivalent stress.
        
        Args:
            stress_tensor: Cauchy stress tensor
            
        Returns:
            σ_eq = sqrt(3/2 * s:s) where s is deviatoric stress
        """
        s = TensorOperations.deviatoric(stress_tensor)
        return jnp.sqrt(1.5) * TensorOperations.frobenius_norm(s)
    
    @staticmethod
    def hydrostatic_pressure(stress_tensor: Array) -> float:
        """Compute hydrostatic pressure: p = (1/3)tr(σ)"""
        return TensorOperations.trace(stress_tensor) / 3.0
    
    @staticmethod
    def fourth_order_identity_tensor(dim=3):
        """
        Calculate fourth identity matrix
        """
        eye = jnp.eye(dim)
        I4 = jnp.einsum('ik,jl->ijkl', eye, eye)
        return I4
    
    @staticmethod
    def diad_special(A, B, dim):
        """
        Calculate a specific tensor diad: Cijkl = (1/2)*(A[i,k] * B[j,l] + A[i,l] * B[j,k])
        """
        P = 0.5* (jnp.einsum('ik,jl->ijkl',A,B) + jnp.einsum('il,jk->ijkl',A,B))
        return P