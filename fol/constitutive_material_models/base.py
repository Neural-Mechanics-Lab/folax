"""
 Authors: Rishabh Arora, https://github.com/rishabharora236-cell
 Date: Dec, 2025
 License: FOL/LICENSE
"""
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
