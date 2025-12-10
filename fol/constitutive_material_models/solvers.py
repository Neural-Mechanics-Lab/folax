"""
Nonlinear solvers for constitutive models.
Implements Newton-Raphson and variants compatible with JAX.
"""
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, Tuple, Optional
from functools import partial


class NewtonSolver:
    """
    JAX-compatible Newton-Raphson solver.
    
    Differentiable through using jax.lax.while_loop for fixed iterations
    or jax.lax.cond for conditional logic.
    """
    
    def __init__(self, max_iter: int = 50, tolerance: float = 1e-6):
        """
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance on residual norm
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    @partial(jax.jit, static_argnums=(0, 1))
    def solve(self, 
              residual_fn: Callable[[Array], Array],
              x0: Array) -> Array:
        """
        Solve R(x) = 0 using Newton-Raphson.
        
        Args:
            residual_fn: Function computing residual R(x)
            x0: Initial guess
            
        Returns:
            Solution x such that ||R(x)|| < tolerance
        """
        
        def cond_fn(state):
            x, k = state
            r = residual_fn(x)
            norm_r = jnp.linalg.norm(r)
            return jnp.logical_and(norm_r > self.tolerance, k < self.max_iter)
        
        def body_fn(state):
            x, k = state
            r = residual_fn(x)
            J = jax.jacfwd(residual_fn)(x)
            
            # Solve J * dx = -r
            dx = jnp.linalg.solve(J, -r)
            x_new = x + dx
            
            return (x_new, k + 1)
        
        # Run Newton iterations
        x_final, _ = jax.lax.while_loop(cond_fn, body_fn, (x0, 0))
        
        return x_final
    
    @partial(jax.jit, static_argnums=(0, 1))
    def solve_with_info(self,
                        residual_fn: Callable[[Array], Array],
                        x0: Array) -> Tuple[Array, dict]:
        """
        Solve with diagnostic information.
        
        Returns:
            x: Solution
            info: Dictionary with convergence info
        """
        
        def body_fn(state):
            x, k, converged = state
            r = residual_fn(x)
            norm_r = jnp.linalg.norm(r)
            
            # Check convergence
            conv = norm_r < self.tolerance
            
            # Only update if not converged
            def update_fn(x):
                J = jax.jacfwd(residual_fn)(x)
                dx = jnp.linalg.solve(J, -r)
                return x + dx
            
            def no_update_fn(x):
                return x
            
            x_new = jax.lax.cond(conv, no_update_fn, update_fn, x)
            
            return (x_new, k + 1, jnp.logical_or(converged, conv))
        
        def cond_fn(state):
            _, k, converged = state
            return jnp.logical_and(jnp.logical_not(converged), k < self.max_iter)
        
        # Solve
        x_final, n_iter, converged = jax.lax.while_loop(
            cond_fn, body_fn, (x0, 0, False)
        )
        
        r_final = residual_fn(x_final)
        
        info = {
            'converged': converged,
            'iterations': n_iter,
            'residual_norm': jnp.linalg.norm(r_final)
        }
        
        return x_final, info


