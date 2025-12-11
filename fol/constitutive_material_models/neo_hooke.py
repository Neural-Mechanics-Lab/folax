import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .base import BaseConstitutiveModel
from .utils import TensorOperations as TO
from .utils import TensorVoigtArray as TVA



# -----------------------------------------
class NeoHookianModel2D(BaseConstitutiveModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu):
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
        I_fourth = TO.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/2)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/2))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = TO.diad_special(invC,invC,invC.shape[0]) - (1/2)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*p + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*p*TO.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/2)*(J**(-2/2))*jnp.vdot(S_bar,C)*P_bar - \
                (2/2)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = TVA.TensorToVoigt(Se)
        C_tangent = TVA.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
class NeoHookianModel(BaseConstitutiveModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu):
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
        I_fourth = TO.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/3))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = TO.diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*ph + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*ph*TO.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/3)*(J**(-2/3))*jnp.vdot(S_bar,C)*P_bar - \
                (2/3)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = TVA.TensorToVoigt(Se)
        C_tangent = TVA.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
        
    
class NeoHookianModelAD(BaseConstitutiveModel):
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
            C = TVA.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
            I1_bar = (J**(-2/3))*jnp.trace(C)
            xsie_iso = 0.5*mu*(I1_bar - 3)
            return 0.5*mu*(I1_bar - 3) - mu*jnp.log(J) + (lambda_/2)*(jnp.log(J))**2
        
        def strain_energy_paper(C_voigt):
            C = TVA.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
            I1_bar = (J**(-2/3))*jnp.trace(C)
            xsie_iso = 0.5*mu*(I1_bar - 3)
            return xsie_vol + xsie_iso
        
        def second_piola(C_voigt):
            return 2*jax.grad(strain_energy)(C_voigt)
        
        def tangent(C_voigt):
            return 2*jax.jacfwd(second_piola)(C_voigt)
        
        C_voigt = TVA.TensorToVoigt(C_mat)

        xsie = strain_energy(C_voigt)
        Se_voigt = second_piola(C_voigt)
        C_tangent = tangent(C_voigt)

        return xsie, Se_voigt, C_tangent.squeeze()
    
class NeoHookianModel2DAD(BaseConstitutiveModel):
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
            C = TVA.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            return 0.5*mu*(jnp.linalg.trace(C) - 2) - mu*jnp.log(J) + 0.5*lambda_*(jnp.log(J)**2)

        
        def strain_energy_paper(C_voigt):
            C = TVA.VoigtToTensor(C_voigt)
            J = jnp.sqrt(jnp.linalg.det(C))
            return (k/4)*(J**2 - 2*jnp.log(J) -1) + 0.5*mu*((J**(-2/2))*jnp.trace(C) - 2)
        
        def second_piola(C_voigt):
            return 2*jax.grad(strain_energy)(C_voigt)
        
        def tangent(C_voigt):
            return 2*jax.jacfwd(second_piola)(C_voigt)
        
        # C_mat = jnp.dot(F.T,F)
        C_voigt = TVA.TensorToVoigt(C_mat)

        xsie = strain_energy(C_voigt)
        Se_voigt = second_piola(C_voigt)
        C_tangent = tangent(C_voigt)
        return xsie, Se_voigt, C_tangent.squeeze()
