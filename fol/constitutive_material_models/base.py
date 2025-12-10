from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
import jax.numpy as jnp
from jax import Array

class BaseConstitutiveModel(ABC):
    """
    Abstract base class for constitutive models.
    Minimal interface - subclasses define their own evaluate signature.
    """
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Tuple:
        """
        Evaluate constitutive relation.
        
        Returns vary by material type:
        - Hyperelastic: (energy, stress, tangent)
        - Small-strain plastic: (stress, new_state)
        - Large-strain plastic: (stress, new_state)
        """
        pass
    

class HyperelasticModel(BaseConstitutiveModel):
    """Base for hyperelastic materials (no history)"""
    
    @abstractmethod
    def evaluate(self, F: Array, *params) -> Tuple[float, Array, Array]:
        """
        Returns:
            energy: Strain energy density
            stress: Stress tensor (PK2 or Cauchy)
            tangent: Material tangent
        """
        pass


class PlasticityModel(BaseConstitutiveModel):
    """Base for plasticity models (with history)"""
    
    @abstractmethod
    def evaluate(self, strain: Array, state: Array) -> Tuple[Array, Array, Array]:
        """
        Returns:
            stress: Cauchy stress
            new_state: Updated state vector
        """
        pass
    
    @abstractmethod
    def initial_state(self, dim: int = 3) -> Array:
        """Must return initial state vector"""
        pass