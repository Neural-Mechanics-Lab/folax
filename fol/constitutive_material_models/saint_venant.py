import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
from .base import BaseConstitutiveModel
from .utils import TensorVoigtArray as TVA
from .utils import TensorOperations as TO

class SaintVenant(BaseConstitutiveModel):
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

        I_fourth = TO.fourth_order_identity_tensor(F.shape[0])
        C_tangent_fourth = lambda_ * jnp.einsum('ij,kl->jikl',jnp.eye(F.shape[0]),jnp.eye(F.shape[0])) +\
                            2 * mu * I_fourth
        Se = jnp.einsum('ijkl,kl->ij',C_tangent_fourth,E)
        Se_voigt = TVA.TensorToVoigt(Se)
        C_tangent = TVA.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent

class SaintVenantAD(BaseConstitutiveModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, E_mat, lambda_, mu, *args, **keyargs):
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
        E_voigt = TVA.TensorToVoigt(E_mat)

        def strain_energy(E_voigt):
            E = TVA.VoigtToTensor(E_voigt)
            return 0.5*lambda_*(jnp.linalg.trace(E) ** 2) + mu*jnp.linalg.trace(E @ E)
        xsie = strain_energy(E_voigt)

        
        def second_piola(E_voigt):
            return jax.grad(strain_energy)(E_voigt)
        Se_voigt = second_piola(E_voigt)

        def tangent(E_voigt):
            return jax.jacfwd(second_piola)(E_voigt)
        C_tangent = tangent(E_voigt)
        
        return xsie, Se_voigt, C_tangent.squeeze()
