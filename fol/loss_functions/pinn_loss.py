"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from  .loss import Loss
import jax
import jax.numpy as jnp
import warnings
from jax import jit,grad
from functools import partial
from abc import abstractmethod
from fol.tools.decoration_functions import *
from jax.experimental import sparse
from fol.mesh_input_output.mesh import Mesh
from fol.tools.fem_utilities import *
from fol.geometries import fe_element_dict

class PinnThermalLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name)
        self.loss_settings = loss_settings
        self.dofs = ['T']
        # self.element_type = self.loss_settings["element_type"]
        self.fe_mesh = fe_mesh
        if "dirichlet_bc_dict" not in self.loss_settings.keys():
            fol_error("dirichlet_bc_dict should provided in the loss settings !")

    def __CreateDofsDict(self, dofs_list:list, dirichlet_bc_dict:dict):
        number_dofs_per_node = len(dofs_list)
        dirichlet_indices = []
        dirichlet_values = []     
        dirichlet_indices_dict = {}
        for dof_index,dof in enumerate(dofs_list):
            dirichlet_indices_dict[dof]={}
            for boundary_name,boundary_value in dirichlet_bc_dict[dof].items():
                boundary_node_ids = jnp.array(self.fe_mesh.GetNodeSet(boundary_name))
                dirichlet_bc_indices = number_dofs_per_node*boundary_node_ids + dof_index
                dirichlet_indices.append(dirichlet_bc_indices)

                dirichlet_bc_values = boundary_value * jnp.ones_like(dirichlet_bc_indices)
                dirichlet_values.append(dirichlet_bc_values)
                dirichlet_indices_dict[dof][boundary_name]=dirichlet_bc_indices
        
        if len(dirichlet_indices) != 0:
            self.dirichlet_indices = jnp.concatenate(dirichlet_indices)
            self.dirichlet_values = jnp.concatenate(dirichlet_values)
        else:
            self.dirichlet_indices = jnp.array([], dtype=jnp.int32)
            self.dirichlet_values = jnp.array([])

        all_indices = jnp.arange(number_dofs_per_node*self.fe_mesh.GetNumberOfNodes())
        self.non_dirichlet_indices = jnp.setdiff1d(all_indices, self.dirichlet_indices)
        self.dirichlet_indices_dict = dirichlet_indices_dict

    def Initialize(self,reinitialize=False) -> None:

        if self.initialized and not reinitialize:
            return

        self.number_dofs_per_node = len(self.dofs)
        self.total_number_of_dofs = len(self.dofs) * self.fe_mesh.GetNumberOfNodes()
        self.__CreateDofsDict(self.dofs,self.loss_settings["dirichlet_bc_dict"])
        self.number_of_unknown_dofs = self.non_dirichlet_indices.size

        # fe element
        # self.fe_element = fe_element_dict[self.element_type]

        # if not "compute_dims" in self.loss_settings.keys():
        #     raise ValueError(f"compute_dims must be provided in the loss settings of {self.GetName()}! ")

        # self.dim = self.loss_settings["compute_dims"]

        neuman_indices_top = jnp.argwhere(self.fe_mesh.GetNodesCoordinates()[:,1]==1.)
        neuman_indices_bottom = jnp.argwhere(self.fe_mesh.GetNodesCoordinates()[:,1]==0.)
        self.neuman_indices = jnp.union1d(neuman_indices_top,neuman_indices_bottom)

        node_num_at_edge = int(self.fe_mesh.GetNumberOfNodes()**0.5)
        dx,dy = 1. / (node_num_at_edge - 1), 1. / (node_num_at_edge - 1)
        coords = self.fe_mesh.GetNodesCoordinates()
        self.area = dx*dy*jnp.ones(coords.shape[0],)
        for i in range(self.fe_mesh.GetNumberOfNodes()):
            if coords[i,0] == 0 or coords[i,0] == 1:
                self.area = self.area.at[i].multiply(0.5)
            if coords[i,1] == 0 or coords[i,1] == 1:
                self.area = self.area.at[i].multiply(0.5)

        def ConstructFullDofVector(known_dofs: jnp.array,all_dofs: jnp.array):
            return all_dofs.at[:,self.dirichlet_indices].set(self.dirichlet_values)

        def ConstructFullDofVectorParametricLearning(known_dofs: jnp.array,all_dofs: jnp.array):
            return all_dofs.at[:,self.dirichlet_indices].set(known_dofs)

        if self.loss_settings.get("parametric_boundary_learning"):
            self.full_dof_vector_function = ConstructFullDofVectorParametricLearning
            self.K_matrix = self.loss_settings.get("K_matrix", jnp.ones(self.fe_mesh.GetNumberOfNodes()))
        else:
            self.full_dof_vector_function = ConstructFullDofVector

        # set scalar-valued loss function exponent
        if "loss_function_exponent" in self.loss_settings:
            self.loss_function_exponent = self.loss_settings["loss_function_exponent"]
        else:
            self.loss_function_exponent = 1.0

        self.initialized = True

    def GetFullNeuman(self,gradient_values):
        # solution_vector = jnp.zeros((self.fe_mesh.GetNumberOfNodes(),3))
        # solution_vector = solution_vector.at[:].set(gradient_values)
        gradient_values = gradient_values.at[:,self.neuman_indices,1].set(0.)
        return gradient_values
    
    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        return self.full_dof_vector_function(known_dofs,unknown_dofs)

    def Finalize(self) -> None:
        pass

    def GetNumberOfUnknowns(self):
        return self.number_of_unknown_dofs
    
    def GetTotalNumberOfDOFs(self):
        return self.total_number_of_dofs
    
    def GetDOFs(self):
        return self.dofs

    def ComputeNodalEnergy(self,
                             nn_derivative_i:jnp.array,
                             de:jnp.array,
                             te:jnp.array,
                             area:jnp.array) -> float:
        grad_T = nn_derivative_i.flatten()
        te_sg =  jax.lax.stop_gradient(te)
        return te_sg * 0.5*de* (grad_T @ grad_T)*area
    
    def ComputeNodalEnergyVmapCompatible(self,
                                           node_id:jnp.integer,
                                           nn_derivative_batch:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        return self.ComputeNodalEnergy(nn_derivative_batch[node_id],
                                         full_control_vector[node_id],
                                         full_dof_vector[node_id].reshape(-1,1),
                                         area=self.area[node_id])
    

    def ComputeNodessEnergies(self,nn_derivative_batch:jnp.array,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        # parallel calculation of energies
        return jax.vmap(self.ComputeNodalEnergyVmapCompatible,(0,None,None,None)) \
                        (jnp.arange(self.fe_mesh.GetNumberOfNodes()),
                        nn_derivative_batch,
                        total_control_vars,
                        total_primal_vars)

    def ComputeTotalEnergy(self,nn_derivative_batch:jnp.array,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        return jnp.sum(self.ComputeNodessEnergies(nn_derivative_batch,total_control_vars,total_primal_vars)) 


    def ComputeBatchLoss(self,nn_derivative_batch:jnp.array, batch_params:jnp.array,batch_dofs:jnp.array):
        batch_params = jnp.atleast_2d(batch_params)
        batch_params = batch_params.reshape(batch_params.shape[0], -1)
        batch_dofs = jnp.atleast_2d(batch_dofs)
        batch_dofs = batch_dofs.reshape(batch_dofs.shape[0], -1)
        BC_applied_batch_dofs = self.GetFullDofVector(batch_params,batch_dofs)
        NBC_applied_batch_dofs = self.GetFullNeuman(nn_derivative_batch)

        def ComputeSingleLoss(params,dofs,nn_derivative_batch):
            return jnp.sum(self.ComputeNodessEnergies(nn_derivative_batch,params,dofs))**self.loss_function_exponent

        batch_energies = jax.vmap(ComputeSingleLoss)(batch_params,BC_applied_batch_dofs,NBC_applied_batch_dofs)

        def ComputeSingleBCLoss(dofs,nn_derivative):
            dirichlet_err = abs(dofs[self.dirichlet_indices] - self.dirichlet_values)
            neumann_err = abs(nn_derivative[:,1] - 0)
            return jnp.mean(dirichlet_err**self.loss_function_exponent) + jnp.mean(neumann_err**self.loss_function_exponent)
        
        batch_bc_loss = jax.vmap(ComputeSingleBCLoss, in_axes=(0,0))(batch_dofs,nn_derivative_batch)

        total_loss = jnp.mean(batch_energies) + jnp.mean(batch_bc_loss)

        return total_loss,(jnp.min(total_loss),jnp.max(total_loss),jnp.mean(total_loss))

   