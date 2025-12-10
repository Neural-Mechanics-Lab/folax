"""
J2 (von Mises) plasticity with isotropic hardening.

Implements return mapping algorithm for small-strain plasticity.
"""
import jax
import jax.numpy as jnp
from jax import Array
from functools import partial
from typing import Tuple

from .base import PlasticityModel
from .utils import TensorVoigtArray as TVA
from .utils import TensorOperations as TO
from .solvers import NewtonSolver


# -----------------------------------------
def plane_strain_embedding(tensor_2d: Array, zz_component: float = 0.0) -> Array:
        """
        Embed 2D tensor into 3D assuming plane strain.
        
        Args:
            tensor_2d: 2x2 tensor
            zz_component: Value for the zz component
            
        Returns:
            3x3 tensor with plane strain assumption
        """
        tensor_3d = jnp.zeros((3, 3), dtype=tensor_2d.dtype)
        tensor_3d = tensor_3d.at[:2, :2].set(tensor_2d)
        tensor_3d = tensor_3d.at[2, 2].set(zz_component)
        return tensor_3d

def isotropic_3d(E: float, nu: float) -> Tuple[float, float, callable]:
        """
        Construct 3D isotropic elasticity tensor.
        
        Args:
            E: Young's modulus
            nu: Poisson's ratio
            
        Returns:
            lam: Lamé's first parameter
            G: Shear modulus
            C_dot: Function that applies elasticity tensor to strain
        """
        lam = E * nu / ((1 + nu) * (1 - 2*nu))
        G = E / (2 * (1 + nu))
        I3 = jnp.eye(3)
        
        def C_dot(strain_tensor: Array) -> Array:
            """Apply elasticity tensor: σ = λ*tr(ε)*I + 2G*ε"""
            tr_eps = jnp.trace(strain_tensor)
            return lam * tr_eps * I3 + 2.0 * G * strain_tensor
        
        return lam, G, C_dot

class J2Plasticity(PlasticityModel):
    """
    J2 (von Mises) plasticity with nonlinear isotropic hardening.
    
    Yield function: f = σ_eq - (σ_y0 + H(ξ))
    Hardening law: H(ξ) = h1 * (1 - exp(-h2 * ξ))
    
    where ξ is cumulative plastic strain.
    """
    
    def __init__(self,
                 E: float,
                 nu: float,
                 yield_stress: float,
                 hardening_modulus: float,
                 hardening_exponent: float = 0.0,
                 tolerance: float = 1e-6,
                 max_iter: int = 50):
        """
        Args:
            E: Young's modulus
            nu: Poisson's ratio
            yield_stress: Initial yield stress σ_y0
            hardening_modulus: Saturation hardening h1
            hardening_exponent: Hardening rate h2
            tolerance: Newton solver tolerance
            max_iter: Maximum Newton iterations
        """
        self.E = E
        self.nu = nu
        self.y0 = yield_stress
        self.h1 = hardening_modulus
        self.h2 = hardening_exponent
        
        # Setup elastic tensor
        self.lam, self.G, self.C_elastic = isotropic_3d(E, nu)
        
        # Newton solver for return mapping
        self.newton = NewtonSolver(max_iter=max_iter, tolerance=tolerance)
    
    
    def initial_state(self, dim: int = 3) -> Array:
        """
        Initialize state with zero plastic strain.
        
        Returns:
            Flat state vector [ε_p (6 components for 3D), ξ]
        """
        n_components = 3 if dim == 2 else 6
        return jnp.zeros(n_components + 1)
    
    def _hardening(self, xi: float) -> float:
        """Evaluate hardening law"""
        return self.y0 + self.h1 * (1.0 - jnp.exp(-self.h2 * xi))
    
    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self,
                 strain: Array,
                 state: Array) -> Tuple[Array, Array]:
        """
        Evaluate stress and tangent via return mapping.
        
        Args:
            strain: Total strain tensor (2x2 or 3x3)
            state: Plastic state from previous step
            
        Returns:
            stress: Cauchy stress tensor
            new_state: Updated plastic state
        """
        # Handle dimension
        dim = strain.shape[0]
        
        # Embed 2D to 3D if needed (plane strain)
        if dim == 2:
            ps_zz = -(state[0] + state[1])
            ps_2d= TVA.ArrayToTensor(state[:3])
            ps_3d = plane_strain_embedding(ps_2d, ps_zz)
            strain_3d = plane_strain_embedding(strain, 0.0)
        else:
            ps_3d = TVA.ArrayToTensor(state[:-1])
            strain_3d = strain
        
        # Return mapping algorithm
        stress_3d, ps_3d_new, xi_new = self._return_mapping(
            strain_3d, 
            ps_3d,
            state[-1]
        )
        
        # Extract 2D if needed
        if dim == 2:
            stress = stress_3d[:2, :2]
            stress_out = TVA.TensorToArray(stress)
            ps_new = ps_3d_new[:2, :2]
            ps_new_flat= TVA.TensorToArray(ps_new)
        else:
            stress = stress_3d
            stress_out= TVA.TensorToArray(stress)
            ps_new = ps_3d_new
            ps_new_flat= TVA.TensorToArray(ps_new)
        
        new_state_flat = jnp.concatenate([ps_new_flat,jnp.array([xi_new])])
        
        #Use when want to compute at gauss point level 
        '''strain_array= TVA.TensorToArray(strain)
        def stress_voigt_autodiff(strain_array):
            strain_tensor=TVA.ArrayToTensor(strain_array)
            if dim==2:
                strain_tensor=plane_strain_embedding(strain_tensor,0.0)  
            stress_1,_,_= self._return_mapping(strain_tensor,ps_3d,state[-1])
            stress_2=stress_1
            if dim==2:
                stress_2 = stress_1[:2, :2]
            
            return TVA.TensorToArray(stress_2)

        C_voigt = jax.jacfwd(stress_voigt_autodiff)(strain_array)'''
        
        return stress_out, new_state_flat
    
    def _return_mapping(self,
                        total_strain: Array,
                        plastic_strain: Array,
                        xi: float) -> Tuple[Array, Array, float]:
        """
        Return mapping algorithm (radial return).
        
        Args:
            total_strain: Total strain (3x3)
            plastic_strain: Plastic strain from previous step (3x3)
            xi: Cumulative plastic strain from previous step
            
        Returns:
            stress: Updated stress (3x3)
            plastic_strain_new: Updated plastic strain (3x3)
            xi_new: Updated cumulative plastic strain
        """
        # Trial elastic step
        elastic_strain_trial = total_strain - plastic_strain
        stress_trial = self.C_elastic(elastic_strain_trial)
        
        s_trial = TO.deviatoric(stress_trial)
        sigma_eq_trial = TO.von_mises_stress(stress_trial)
        
        # Check yield condition
        yield_stress = self._hardening(xi)
        f_trial = sigma_eq_trial - yield_stress
        
        # Elastic return
        def elastic_return():
            return stress_trial, plastic_strain, xi
        
        # Plastic return
        def plastic_return():
            return self._plastic_corrector(
                total_strain, plastic_strain, xi,
                s_trial, sigma_eq_trial
            )
        
        # Conditional return
        return jax.lax.cond(
            f_trial <0.0,
            elastic_return,
            plastic_return
        )
    
    def _plastic_corrector(self,
                           total_strain: Array,
                           plastic_strain_old: Array,
                           xi_old: float,
                           s_trial: Array,
                           sigma_eq_trial: float) -> Tuple[Array, Array, float]:
        """
        Plastic corrector step - solve for plastic multiplier.
        
        Returns:
            stress: Corrected stress
            plastic_strain_new: Updated plastic strain
            xi_new: Updated cumulative plastic strain
        """
        # Flow direction
        n_flow = s_trial / (sigma_eq_trial + 1e-12)
        
        # Setup residual for Newton solve
        def make_residual():
            def residual(dx):
                """
                Solve for incremental plastic strain and plastic multiplier.
                
                Unknowns: [Δε_p (6 components), Δλ (1 scalar)]
                """
                deps_p_voigt = dx[:-1]
                dlambda = dx[-1]
                
                # Update plastic strain
                deps_p = TVA.ArrayToTensor(deps_p_voigt)
                eps_p_new = plastic_strain_old + deps_p
                
                # Compute stress
                eps_e = total_strain - eps_p_new
                sigma = self.C_elastic(eps_e)
                
                # Deviatoric stress and equivalent stress
                s = TO.deviatoric(sigma)
                sigma_eq = TO.von_mises_stress(sigma)
                
                # Flow direction
                n = s / (sigma_eq + 1e-12)
                n_voigt = TVA.TensorToArray(n)
                
                # Cumulative plastic strain update
                xi_new = xi_old + dlambda
                
                # Residuals
                # Flow rule: Δε_p = Δλ * n
                r_flow = deps_p_voigt - dlambda * n_voigt
                
                # Yield condition: σ_eq = σ_y(ξ)
                r_yield = sigma_eq - self._hardening(xi_new)
                
                return jnp.concatenate([r_flow, jnp.array([r_yield])])
            
            return residual
        
        # Initial guess (elastic predictor)
        x0 = jnp.zeros(7)
        
        # Solve nonlinear system
        residual_fn = make_residual()
        solution = self.newton.solve(residual_fn, x0)
        
        # Extract solution
        deps_p_voigt = solution[:6]
        dlambda = solution[6]
        
        # Update state
        deps_p = TVA.ArrayToTensor(deps_p_voigt)
        plastic_strain_new = plastic_strain_old + deps_p
        xi_new = xi_old + dlambda
        
        # Compute final stress
        elastic_strain = total_strain - plastic_strain_new
        stress = self.C_elastic(elastic_strain)
        
        return stress, plastic_strain_new, xi_new