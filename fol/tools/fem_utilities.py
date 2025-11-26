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
from jax import jit,lax
import jax
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
    


# ---------- Public API inside your model: get C_alg via AD ----------
class ElastoplasticityModel2D(MaterialModel):

    @staticmethod    # ---------- helpers to pack/unpack 2D Voigt ----------
    def voigt2_to_tensor(v):
        # v = [xx, yy, xy_tensor] (xy is tensor shear, not engineering)
        return jnp.array([[v[0], v[2]],
                        [v[2], v[1]]], dtype=v.dtype)
    @staticmethod
    def tensor2_to_voigt(T):
        # inverse of above (xy is tensor shear component)
        return jnp.array([T[0,0], T[1,1], T[0,1]], dtype=T.dtype)
    @staticmethod
    def to3D_from_2D_plane_strain(A2, Azz=0.0):
        A3 = jnp.zeros((3,3), dtype=A2.dtype)
        A3 = A3.at[0:2, 0:2].set(A2)
        A3 = A3.at[2,2].set(Azz)
        return A3
    @staticmethod
    # (optional) Map 3x3 Voigt -> 4th-order tensor in 2D (xx,yy,xy_tensor ordering)
    def voigt33_to_C4_2D(Cv):
        # Cv order: [xx,yy,xy]x[xx,yy,xy] with tensor-shear in xy
        # Build 4th-order C_ijkl (i,j,k,l in {x,y}) consistent with our Voigt convention
        C = jnp.zeros((2,2,2,2), dtype=Cv.dtype)
        # helper: map (xx->0, yy->1, xy->2)
        def idx_pair(a):
            return (0,0) if a==0 else (1,1) if a==1 else (0,1)
        def sym_set(C, i,j,k,l, val):
            # enforce minor symmetries: ij and kl
            C = C.at[i,j,k,l].set(val)
            C = C.at[j,i,k,l].set(val)
            C = C.at[i,j,l,k].set(val)
            C = C.at[j,i,l,k].set(val)
            return C
        for a in range(3):
            for b in range(3):
                i,j = idx_pair(a)
                k,l = idx_pair(b)
                C = sym_set(C, i,j,k,l, Cv[a,b])
        return C

    # ---------- Elasticity in 3D ----------
    @staticmethod
    def C_elastic_3D(E, nu):
        I3 = jnp.eye(3)
        lam = E*nu / ((1+nu)*(1-2*nu))
        G   = E/(2*(1+nu))
        def C_dot(e3):
            tr = jnp.trace(e3)
            return lam*tr*I3 + 2.0*G*e3
        return lam, G, C_dot

    # ---------- Newton solver made JAX-differentiable ----------
    @staticmethod
    def newton_solve(R, x0, maxit=60, tol=1e-3):

        # run fixed number of iterations but allow early stop with no-op steps
        def while_cond(state):
            x, k = state
            r = R(x)
            return jnp.logical_and(jnp.linalg.norm(r) > tol, k < maxit)

        def while_body(state):
            x, k = state
            r = R(x)
            J = jax.jacfwd(R)(x)
            dx = jnp.linalg.solve(J, -r)
            return (x + dx, k + 1)

        x_final, _ = jax.lax.while_loop(while_cond, while_body, (x0, 0))
        return x_final

    # ---------- Plasticity helpers ----------
    @staticmethod
    def tr3(A): return jnp.trace(A)
    @staticmethod
    def frob(A): return jnp.sqrt(jnp.tensordot(A, A))
    @staticmethod
    def dev3(A):
        I3 = jnp.eye(3, dtype=A.dtype)
        return A - I3 * (ElastoplasticityModel2D.tr3(A) / 3.0)
    @staticmethod
    def flow_normal(sig3):
        s   = ElastoplasticityModel2D.dev3(sig3)
        seq = jnp.sqrt(1.5) * ElastoplasticityModel2D.frob(s)
        inv = jnp.where(seq > 0.0, 1.0/seq, 0.0)
        return 1.5 * s * inv

    # ---------- Core: one local integration step as a pure function ----------
    @staticmethod
    def local_update(ts2, ps2_n, xi_n, E, nu, h1, h2, y0):
        """
        Inputs:
        ts2   : total strain (2x2 tensor) at current step
        ps2_n : plastic strain at previous step (2x2)
        xi_n  : cum. plastic strain at previous step (scalar)
        Returns:
        sig2_v  : Cauchy stress (2D Voigt, length 3)
        ps2_new : updated plastic strain (2x2)
        xi_n1   : updated cum. plastic strain (scalar)
        """
        _, _, C_dot = ElastoplasticityModel2D.C_elastic_3D(E, nu)

        # plane strain embeddings
        ps_zz  = -(ps2_n[0,0] + ps2_n[1,1])
        ps3_n  = ElastoplasticityModel2D.to3D_from_2D_plane_strain(ps2_n, ps_zz)
        ts3    = ElastoplasticityModel2D.to3D_from_2D_plane_strain(ts2, 0.0)

        # trial state
        es_trial  = ts3 - ps3_n
        sig_trial = C_dot(es_trial)
        s_trial   = ElastoplasticityModel2D.dev3(sig_trial)
        sig_eq    = jnp.sqrt(1.5) * ElastoplasticityModel2D.frob(s_trial)
        yl        = y0 + h1*(1.0 - jnp.exp(-h2*xi_n))
        f_yield   = sig_eq - yl

        # Elastic return: no plastic flow
        def elastic_return():
            sig2 = sig_trial[0:2, 0:2]
            sig2_v = ElastoplasticityModel2D.tensor2_to_voigt(sig2)
            return sig2_v, ps2_n, xi_n
        
        # Plastic return: solve for plastic multiplier
        def plastic_return():
            def make_residual():
                def R(dx):
                    depsp_v = dx[:-1]
                    dp      = dx[-1]
                    
                    # Update plastic strain
                    depsp2 = ElastoplasticityModel2D.voigt2_to_tensor(depsp_v)
                    epsp2  = ps2_n + depsp2
                    epsp_zz = -(epsp2[0,0] + epsp2[1,1])
                    epsp3   = ElastoplasticityModel2D.to3D_from_2D_plane_strain(epsp2, epsp_zz)
                    
                    # Compute stress
                    eps3 = ts3
                    ee3  = eps3 - epsp3
                    sig3 = C_dot(ee3)
                    
                    # Deviatoric stress and equivalent stress
                    s3  = ElastoplasticityModel2D.dev3(sig3)
                    seq = jnp.sqrt(1.5) * ElastoplasticityModel2D.frob(s3)
                    
                    # Flow normal
                    n3 = s3 / (seq + 1e-12)  # Avoid division by zero
                    n2 = n3[0:2, 0:2]
                    n_v = jnp.array([n2[0,0], n2[1,1], n2[0,1]], dtype=n2.dtype)
                    
                    # Residuals
                    r_flow = depsp_v - n_v * dp
                    r_yield = seq - (y0 + h1*(1.0 - jnp.exp(-h2*(xi_n + dp))))
                    
                    return jnp.concatenate([r_flow, jnp.array([r_yield])], axis=0)
                
                return R
            
            R = make_residual()
            x0 = jnp.zeros((4,))
            x = ElastoplasticityModel2D.newton_solve(R, x0, maxit=60, tol=1e-3)
            
            dps_v = x[:-1]
            dp    = x[-1]
            
            # Update plastic variables
            dps2 = ElastoplasticityModel2D.voigt2_to_tensor(dps_v)
            ps2_new = ps2_n + dps2
            xi_n1 = xi_n + dp
            
            # Compute final stress
            ps_zz_new = -(ps2_new[0,0] + ps2_new[1,1])
            ps3_new   = ElastoplasticityModel2D.to3D_from_2D_plane_strain(ps2_new, ps_zz_new)
            es    = ts3 - ps3_new
            sig3  = C_dot(es)
            sig2  = sig3[0:2, 0:2]
            sig2_v = ElastoplasticityModel2D.tensor2_to_voigt(sig2)
            
            return sig2_v, ps2_new, xi_n1
        
        # Choose elastic or plastic return
        return jax.lax.cond(f_yield <= 0.0, elastic_return, plastic_return)

    @partial(jit, static_argnums=(0,))
    def evaluate(self, E, nu, h1, h2, y0, ts, state):
        """
        Returns:
          C_alg_4 : 4th-order tangent in 2D (i,j,k,l over x,y)  OR you can return the 3x3 Voigt if you prefer
          sig2_v  : stress in 2D Voigt (length 3)
          ps_new  : updated plastic strain tensor (2x2)
          xi_n1   : updated cum. plastic strain (scalar)
        """
        ps_v  = state[:3]
        xi_n=state[3]
        ps=ElastoplasticityModel2D.voigt2_to_tensor(ps_v)

        # 1) do the usual local update to get stress at the given ts
        sig2_v, ps_new, xi_n1 = ElastoplasticityModel2D.local_update(ts, ps, xi_n, E, nu, h1, h2, y0)

        # 2) define a pure function "stress(ts_v)" so we can differentiate w.r.t. total strain
        def stress_voigt_from_ts_voigt(ts_v_flat):
            ts2 = ElastoplasticityModel2D.voigt2_to_tensor(ts_v_flat)
            sig2_v_loc, _, _ = ElastoplasticityModel2D.local_update(ts2, ps, xi_n, E, nu, h1, h2, y0)
            return sig2_v_loc  # length-3 Voigt vector

        # 3) automatic differentiation: d(sig_voigt)/d(eps_voigt) -> (3x3) plane-strain tangent in Voigt
        ts_v = ElastoplasticityModel2D.tensor2_to_voigt(ts)
        C_voigt = jax.jacfwd(stress_voigt_from_ts_voigt)(ts_v)  # shape (3,3)

        # 4) (optional) lift to a true 4th-order tensor in 2D

        ps_v = ElastoplasticityModel2D.tensor2_to_voigt(ps_new)             # shape (3,)
        state_vec = jnp.concatenate([ps_v, jnp.array([xi_n1])])  # shape (4,)

        return C_voigt, sig2_v, state_vec
    
class JAXNewton:
    def __init__(self, maxit=50, tol=1e-3):
        self.maxit = maxit
        self.tol = tol

    def solve(self, R, x0):
        """
        R: residual function R(x) -> (m,), with m == len(x)
        x0: initial guess
        returns: x, info dict
        """
        x = x0
        for k in range(self.maxit):
            r = R(x)
            nrm = jnp.linalg.norm(r)

            # Use lax.cond() to check the residual norm condition
            def condition_met_fn(x):
                return x

            def continue_iteration_fn(x):
                J = jax.jacfwd(R)(x)  # Compute Jacobian
                dx = jnp.linalg.solve(J, -r)  # Solve for the step
                x_new = x + dx
                return x_new

            # Apply the condition and either exit the loop or continue
            x= lax.cond(nrm < self.tol, condition_met_fn, continue_iteration_fn, x)

        # Final residual after the loop
        r = R(x)
        nrm = jnp.linalg.norm(r)
        return x
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
            return 2*jax.grad(strain_energy)(C_voigt)
        
        def tangent(C_voigt):
            return 2*jax.jacfwd(second_piola)(C_voigt)
        
        C_voigt = self.TensorToVoigt(C_mat)

        xsie = strain_energy(C_voigt)
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
