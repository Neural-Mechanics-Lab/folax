import pytest
import unittest
import os
import numpy as np
from fol.geometries.triangle_2d_3 import Triangle2D3
from fol.geometries.quadrilateral_2d_4 import Quadrilateral2D4
from fol.geometries.tetrahedra_3d_4 import Tetrahedra3D4
from fol.geometries.hexahedra_3d_8 import Hexahedra3D8
from fol.tools.usefull_functions import *
import jax

class TestGeometries(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_tri = Triangle2D3("test_triangle2D3")
        self.tri_points_coordinates = jnp.array([[-0.5,-0.45,0.1],
                                                 [0.7,-0.5,0.2],
                                                 [0.02,-0.01,-0.15]])

        self.test_quad = Quadrilateral2D4("test_quadrilateral2D4")
        self.quad_points_coordinates = jnp.array([[-0.5, - 0.45,  0.1],
                                                  [-0.03, -0.5,   0.0],
                                                  [0.02, -0.01, -0.15],
                                                  [-0.49, -0.0,   0.0]])
        
        self.test_tetra = Tetrahedra3D4("test_tetrahedra3D4")
        self.tet_points_coordinates = jnp.array([[0.1, 0.1, 0.1],
                                                 [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 1.0, 0.1]])
        
        self.test_hexa = Hexahedra3D8("test_hexahedra3D8")
        self.hex_points_coordinates = jnp.array([[0.24900,  0.34200,  0.19200],
                                                 [0.32000,  0.18600,  0.64300],
                                                 [0.16500,  0.74500,  0.70200],
                                                 [0.27300,  0.75000,  0.23000],
                                                 [0.00000,  0.00000,  0.00000],
                                                 [0.00000,  0.00000,  1.00000],
                                                 [0.00000,  1.00000,  1.00000],
                                                 [0.00000,  1.00000,  0.00000]])
    def test_tri2D3(self):
        points,weights = self.test_tri.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.333333 , 0.333333 , 0]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.5]), rtol=1e-5, atol=1e-6)
        shape_function_values = jax.vmap(self.test_tri.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values[0],jnp.array([0.333333,0.333333,0.333333]), rtol=1e-5, atol=1e-6)
        shape_function_grads = jax.vmap(self.test_tri.ShapeFunctionsLocalGradients)(points)
        np.testing.assert_allclose(shape_function_grads[0],jnp.array([[-1,-1],[1,0],[0,1]]), rtol=1e-5, atol=1e-6)
        jacobian = jax.vmap(self.test_tri.Jacobian, in_axes=(None, 0))(self.tri_points_coordinates,points)
        np.testing.assert_allclose(jacobian,jnp.array([[[1.2,0.52],[-0.05,0.44]]]), rtol=1e-5, atol=1e-6)

        self.test_tri.SetGaussIntegrationMethod("GI_GAUSS_2")
        points,weights = self.test_tri.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.166667 , 0.166667 , 0],
                                                     [0.666667 , 0.166667 , 0],
                                                     [0.166667 , 0.666667 , 0]]), 
                                                     rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.166667,0.166667,0.166667]), rtol=1e-5, atol=1e-6)

        self.test_tri.SetGaussIntegrationMethod("GI_GAUSS_3")
        points,weights = self.test_tri.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.2 , 0.2 , 0],
                                                     [0.6 , 0.2 , 0],
                                                     [0.2 , 0.6 , 0],
                                                     [0.333333 , 0.333333 , 0]]), 
                                                     rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.260417,0.260417,0.260417,-0.28125]), rtol=1e-5, atol=1e-6)

    def test_quad2D4(self):
        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0., 0., 0.]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([4]), rtol=1e-5, atol=1e-6)
        shape_function_values = jax.vmap(self.test_quad.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values[0],jnp.array([0.25, 0.25, 0.25, 0.25]), rtol=1e-5, atol=1e-6)
        shape_function_grads = jax.vmap(self.test_quad.ShapeFunctionsLocalGradients)(points)
        np.testing.assert_allclose(shape_function_grads[0],jnp.array([[-0.25,-0.25],[0.25,-0.25],
                                                                      [0.25,0.25], [-0.25,0.25]]), rtol=1e-5, atol=1e-6)
        jacobian = jax.vmap(self.test_quad.Jacobian, in_axes=(None, 0))(self.quad_points_coordinates,points)
        np.testing.assert_allclose(jacobian,jnp.array([[[0.245,0.015],[-0.015,0.235]]]), rtol=1e-5, atol=1e-6)

        shape_function_g_grads = jax.vmap(self.test_quad.ShapeFunctionsGlobalGradients,in_axes=(None, 0))(self.quad_points_coordinates,points)
        np.testing.assert_allclose(shape_function_g_grads,jnp.array([[[-1.08131,-0.99481],
                                                                      [0.951557,-1.12457],
                                                                      [1.08131,0.99481],
                                                                      [-0.951557,1.12457]]]), rtol=1e-5, atol=1e-6) 

        self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_2")
        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.57735026, -0.57735026, 0.00],
                                                    [ 0.57735026, -0.57735026, 0.00],
                                                    [ 0.57735026,  0.57735026, 0.00],
                                                    [-0.57735026,  0.57735026, 0.00]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([1, 1, 1, 1]), rtol=1e-5, atol=1e-6)
        
        self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_3")
        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.774597 , -0.774597 , 0],
                                                    [0 , -0.774597 , 0],
                                                    [0.774597 , -0.774597 , 0],
                                                    [-0.774597 , 0 , 0],
                                                    [0 , 0 , 0],
                                                    [0.774597 , 0 , 0],
                                                    [-0.774597 , 0.774597 , 0],
                                                    [0 , 0.774597 , 0],
                                                    [0.774597 , 0.774597 , 0]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.308642,0.493827,0.308642,
                                                      0.493827,0.790123,0.493827,
                                                      0.308642,0.493827,0.308642]), rtol=1e-5, atol=1e-6)
        
        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_4")
            points,weights = self.test_quad.GetIntegrationData()
        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_5")
            points,weights = self.test_quad.GetIntegrationData()

    def test_tet3D4(self):
        points,weights = self.test_tetra.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.25,0.25,0.25]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.166667]), rtol=1e-5, atol=1e-6)
        shape_function_values = jax.vmap(self.test_tetra.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values,jnp.array([[0.25, 0.25, 0.25, 0.25]]), rtol=1e-5, atol=1e-6)
        shape_function_grads = jax.vmap(self.test_tetra.ShapeFunctionsLocalGradients)(points)
        np.testing.assert_allclose(shape_function_grads,jnp.array([[[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]]), rtol=1e-5, atol=1e-6)
        jacobians = jax.vmap(self.test_tetra.Jacobian, in_axes=(None, 0))(self.tet_points_coordinates,points)
        np.testing.assert_allclose(jacobians,jnp.array([[[0.187394,-0.1,-0.1],
                                                         [0.178085,0.9,0.9],
                                                         [-0.0432702,-0.1,0]]]), rtol=1e-5, atol=1e-6)
        shape_function_g_grads = jax.vmap(self.test_tetra.ShapeFunctionsGlobalGradients, in_axes=(None, 0))(self.tet_points_coordinates,points)
        np.testing.assert_allclose(shape_function_g_grads,jnp.array([[[-3.87163,-1.54129,0],
                                                                      [4.8267,0.5363,0],
                                                                      [-2.08852,-0.232058,-10],
                                                                      [1.13345,1.23705,10]]]), rtol=1e-5, atol=1e-6)        

        self.test_tetra.SetGaussIntegrationMethod("GI_GAUSS_2")
        points,weights = self.test_tetra.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.58541 , 0.138197 , 0.138197],
                                                     [0.138197 , 0.58541 , 0.138197],
                                                     [0.138197 , 0.138197 , 0.58541],
                                                     [0.138197 , 0.138197 , 0.138197]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.0416667,0.0416667,0.0416667,0.0416667]), rtol=1e-5, atol=1e-6)
        shape_function_values = jax.vmap(self.test_tetra.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values,jnp.array([[0.138197,0.58541,0.138197,0.138197],
                                                                    [0.138197,0.138197,0.58541,0.138197],
                                                                    [0.138197,0.138197,0.138197,0.58541],
                                                                    [0.58541,0.138197,0.138197,0.138197]]), rtol=1e-5, atol=1e-6)
        shape_function_grads = jax.vmap(self.test_tetra.ShapeFunctionsLocalGradients)(points)
        np.testing.assert_allclose(shape_function_grads,jnp.array([[[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                                                                   [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                                                                   [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                                                                   [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]]), rtol=1e-5, atol=1e-6)
        jacobians = jax.vmap(self.test_tetra.Jacobian, in_axes=(None, 0))(self.tet_points_coordinates,points)
        np.testing.assert_allclose(jacobians,jnp.array([[[0.187394,-0.1,-0.1],[0.178085,0.9,0.9],[-0.0432702,-0.1,0]],
                                                        [[0.187394,-0.1,-0.1],[0.178085,0.9,0.9],[-0.0432702,-0.1,0]],
                                                        [[0.187394,-0.1,-0.1],[0.178085,0.9,0.9],[-0.0432702,-0.1,0]],
                                                        [[0.187394,-0.1,-0.1],[0.178085,0.9,0.9],[-0.0432702,-0.1,0]]]), rtol=1e-5, atol=1e-6)
        shape_function_g_grads = jax.vmap(self.test_tetra.ShapeFunctionsGlobalGradients, in_axes=(None, 0))(self.tet_points_coordinates,points)
        np.testing.assert_allclose(shape_function_g_grads,jnp.array([[[-3.87163,-1.54129,0],[4.8267,0.5363,0],[-2.08852,-0.232058,-10],[1.13345,1.23705,10]],
                                                                     [[-3.87163,-1.54129,0],[4.8267,0.5363,0],[-2.08852,-0.232058,-10],[1.13345,1.23705,10]],
                                                                     [[-3.87163,-1.54129,0],[4.8267,0.5363,0],[-2.08852,-0.232058,-10],[1.13345,1.23705,10]],
                                                                     [[-3.87163,-1.54129,0],[4.8267,0.5363,0],[-2.08852,-0.232058,-10],[1.13345,1.23705,10]]]), rtol=1e-5, atol=1e-6)   

        self.test_tetra.SetGaussIntegrationMethod("GI_GAUSS_3")
        points,weights = self.test_tetra.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.0158359 , 0.328055 , 0.328055],
                                                     [0.328055 , 0.0158359 , 0.328055],
                                                     [0.328055 , 0.328055 , 0.0158359],
                                                     [0.328055 , 0.328055 , 0.328055],
                                                     [0.679143 , 0.106952 , 0.106952],
                                                     [0.106952 , 0.679143 , 0.106952],
                                                     [0.106952 , 0.106952 , 0.679143],
                                                     [0.106952 , 0.106952 , 0.106952]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.023088,0.023088,0.023088,0.023088,
                                                      0.0185787,0.0185787,0.0185787,0.0185787]), rtol=1e-5, atol=1e-6)

        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_4")
            points,weights = self.test_quad.GetIntegrationData()
        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_5")
            points,weights = self.test_quad.GetIntegrationData()

    def test_hex3D8(self):
        points,weights = self.test_hexa.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0.0,0.0,0.0]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([8.0]), rtol=1e-5, atol=1e-6)
        shape_function_values = jax.vmap(self.test_hexa.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values,jnp.array([[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]]), rtol=1e-5, atol=1e-6)
        shape_function_grads = jax.vmap(self.test_hexa.ShapeFunctionsLocalGradients)(points)
        np.testing.assert_allclose(shape_function_grads,jnp.array([[[-0.125,-0.125,-0.125],
                                                                    [0.125,-0.125,-0.125],
                                                                    [0.125,0.125,-0.125],
                                                                    [-0.125,0.125,-0.125],
                                                                    [-0.125,-0.125,0.125],
                                                                    [0.125,-0.125,0.125],
                                                                    [0.125,0.125,0.125],
                                                                    [-0.125,0.125,0.125]]]), rtol=1e-5, atol=1e-6)
        jacobians = jax.vmap(self.test_hexa.Jacobian, in_axes=(None, 0))(self.hex_points_coordinates,points)
        np.testing.assert_allclose(jacobians,jnp.array([[[-0.004625,-0.016375,-0.125875],
                                                         [-0.020125,0.370875,-0.002875],
                                                         [0.365375,0.012125,0.029125]]]), rtol=1e-5, atol=1e-6)
        shape_function_g_grads = jax.vmap(self.test_hexa.ShapeFunctionsGlobalGradients, in_axes=(None, 0))(self.hex_points_coordinates,points)
        np.testing.assert_allclose(shape_function_g_grads,jnp.array([[[0.919462,-0.285127,-0.34618],
                                                                      [1.0784,-0.300517,0.339212],
                                                                      [1.07159,0.372056,0.376172],
                                                                      [0.912652,0.387446,-0.309221],
                                                                      [-1.07159,-0.372056,-0.376172],
                                                                      [-0.912652,-0.387446,0.309221],
                                                                      [-0.919462,0.285127,0.34618],
                                                                      [-1.0784,0.300517,-0.339212]]]), rtol=1e-5, atol=1e-6)    

        self.test_hexa.SetGaussIntegrationMethod("GI_GAUSS_2")
        points,weights = self.test_hexa.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.57735 , -0.57735 , -0.57735],
                                                     [0.57735 , -0.57735 , -0.57735],
                                                     [0.57735 , 0.57735 , -0.57735],
                                                     [-0.57735 , 0.57735 , -0.57735],
                                                     [-0.57735 , -0.57735 , 0.57735],
                                                     [0.57735 , -0.57735 , 0.57735],
                                                     [0.57735 , 0.57735 , 0.57735],
                                                     [-0.57735 , 0.57735 , 0.57735]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]), rtol=1e-5, atol=1e-6)

        self.test_hexa.SetGaussIntegrationMethod("GI_GAUSS_3")
        points,weights = self.test_hexa.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.774597 , -0.774597 , -0.774597],
                                                     [0 , -0.774597 , -0.774597],
                                                     [0.774597 , -0.774597 , -0.774597],
                                                     [-0.774597 , 0 , -0.774597],
                                                     [0 , 0 , -0.774597],
                                                     [0.774597 , 0 , -0.774597],
                                                     [-0.774597 , 0.774597 , -0.774597],
                                                     [0 , 0.774597 , -0.774597],
                                                     [0.774597 , 0.774597 , -0.774597],
                                                     [-0.774597 , -0.774597 , 0],
                                                     [0 , -0.774597 , 0],
                                                     [0.774597 , -0.774597 , 0],
                                                     [-0.774597 , 0 , 0],
                                                     [0 , 0 , 0],
                                                     [0.774597 , 0 , 0],
                                                     [-0.774597 , 0.774597 , 0],
                                                     [0 , 0.774597 , 0],
                                                     [0.774597 , 0.774597 , 0],
                                                     [-0.774597 , -0.774597 , 0.774597],
                                                     [0 , -0.774597 , 0.774597],
                                                     [0.774597 , -0.774597 , 0.774597],
                                                     [-0.774597 , 0 , 0.774597],
                                                     [0 , 0 , 0.774597],
                                                     [0.774597 , 0 , 0.774597],
                                                     [-0.774597 , 0.774597 , 0.774597],
                                                     [0 , 0.774597 , 0.774597],
                                                     [0.774597 , 0.774597 , 0.774597]]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(weights,jnp.array([0.171468,0.274348,0.171468,0.274348,0.438957,0.274348,0.171468, 
                                                        0.274348,0.171468,0.274348,0.438957,0.274348, 
                                                        0.438957,0.702332,0.438957,0.274348,0.438957, 
                                                        0.274348,0.171468,0.274348,0.171468,0.274348, 
                                                        0.438957,0.274348,0.171468,0.274348,0.171468]), rtol=1e-5, atol=1e-6)

if __name__ == '__main__':
    unittest.main()