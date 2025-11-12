"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/LICENSE
"""
from abc import ABC
import jax.numpy as jnp
import jax
import numpy as np
import os
import meshio
from fol.tools.decoration_functions import *
from fol.geometries import fe_element_dict

_mdpa_to_meshio_type = {
    "Line2D2": "line",
    "Line3D2": "line",
    "Triangle2D3": "triangle",
    "Triangle3D3": "triangle",
    "Quadrilateral2D4": "quad",
    "Quadrilateral3D4": "quad",
    "Tetrahedra3D4": "tetra",
    "Hexahedra3D8": "hexahedron",
    "Prism3D6": "wedge",
    "Line2D3": "line3",
    "Triangle2D6": "triangle6",
    "Triangle3D6": "triangle6",
    "Quadrilateral2D9": "quad9",
    "Quadrilateral3D9": "quad9",
    "Tetrahedra3D10": "tetra10",
    "Hexahedra3D27": "hexahedron27",
    "Point2D": "vertex",
    "Point3D": "vertex",
    "Quadrilateral2D8": "quad8",
    "Quadrilateral3D8": "quad8",
    "Hexahedra3D20": "hexahedron20",
}

class Mesh(ABC):
    """Base abstract input-output class.

    The base abstract InputOutput class has the following responsibilities.
        1. Initalizes and finalizes the IO.

    """
    def __init__(self, io_name: str, file_name:str, case_dir:str=".", scale_factor:float=1) -> None:
        self.__name = io_name
        self.file_name = file_name
        self.mesh_format = file_name.split(sep=".")[1]
        self.case_dir = case_dir
        self.scale_factor = scale_factor
        self.node_ids = jnp.array([])
        self.nodes_coordinates = jnp.array([])
        self.elements_nodes = {}
        self.node_sets = {}
        self.element_sets = {}
        self.mesh_io = None
        self.is_initialized = False

    def GetName(self) -> str:
        return self.__name

    def Initialize(self) -> None:
        """Initializes the io.

        This method initializes the io.

        """

        if self.is_initialized:
            return

        if self.mesh_format != "mdpa":
            self.mesh_io = meshio.read(os.path.join(self.case_dir, self.file_name))
            self.mesh_io.point_data_to_sets('point_tags')
            self.mesh_io.cell_data_to_sets('cell_tags')
            self.node_ids = jnp.arange(len(self.mesh_io.points))
            self.nodes_coordinates = self.scale_factor * jnp.array(self.mesh_io.points)
            #create elemnt nodes dict based on element types
            self.elements_nodes = {}
            for elements_info in self.mesh_io.cells:
                self.elements_nodes[elements_info.type] = jnp.array(elements_info.data)
            # create node sets
            self.node_sets = {}
            for tag,tag_info_list in self.mesh_io.point_tags.items():
                filtered_tag_info_list = [item for item in tag_info_list if 'Group_Of_All_Nodes' not in item]
                if len(filtered_tag_info_list)>1:
                    fol_error(f" the input mesh is not valid ! point set {filtered_tag_info_list} is not unique !")
                elif len(filtered_tag_info_list)==1:
                    point_set_name = filtered_tag_info_list[0]
                    self.node_sets[point_set_name] = jnp.array(self.mesh_io.point_sets[f"set-key-{tag}"])
                    
            # TODO: create element sets 
            self.element_sets = {}

        else:
            with open(os.path.join(self.case_dir, self.file_name), "rb") as f:
                    # Read mesh
                    while True:
                        line = f.readline().decode()
                        if not line:
                            break
                        environ = line.strip()
                        if environ.startswith("Begin Nodes"):
                            self.__ReadKratosNodes(f)
                        elif environ.startswith("Begin Elements"):
                            self.__ReadKratosElements(f, environ)
                        elif environ.startswith("Begin SubModelPart "):
                            self.__ReadKratosSubModelPart(f, environ)

            self.mesh_io = meshio.Mesh(self.nodes_coordinates,self.elements_nodes)
        
        fol_info(f"{len(self.node_ids)} points read ")
        for element_type,element_nodes in self.elements_nodes.items():
            fol_info(f"{len(element_nodes)} {element_type} elements read ")
        for node_set_name,node_ids in self.node_sets.items():
            fol_info(f"({node_set_name},{len(node_ids)} nodes) read ")
        
        self.CheckAndOrientElements()

        self.is_initialized = True
    
    def CheckAndOrientElements(self):
        jax_nodes_coords = jnp.array(self.nodes_coordinates)
        for element_type,elements_nodes in self.elements_nodes.items():
            if element_type in fe_element_dict.keys():
                fol_element = fe_element_dict[element_type]
                gp_point,_= fol_element.GetIntegrationData()
                @jax.jit
                def negative_det(elem_nodes):
                    elem_nodes_coordinates = jax_nodes_coords[elem_nodes]
                    det = jnp.linalg.det(fol_element.Jacobian(elem_nodes_coordinates,gp_point))
                    return jnp.where(det >= 0, 0, 1),jnp.where(det >= 0, elem_nodes, elem_nodes.at[0].set(elem_nodes[1]).at[1].set(elem_nodes[0]))
                elem_state,swap_elems_nodes = jax.vmap(negative_det)(elements_nodes)
                num_neg_jac_elems = jnp.sum(elem_state)
                if num_neg_jac_elems>0:
                    fol_warning(f"nodes of {num_neg_jac_elems} {element_type} elements with negative jacobian are swapped !")
                    self.elements_nodes[element_type] = swap_elems_nodes
                    new_elem_state,_ = jax.vmap(negative_det)(self.elements_nodes[element_type])
                    num_neg_jac_elems = jnp.sum(new_elem_state)
                    if num_neg_jac_elems>0:
                        fol_warning(f"although nodes are swapped, {num_neg_jac_elems} {element_type} elements still have negative jacobians or inverted !")

    def GetNodesIds(self) -> jnp.array:
        return self.node_ids
    
    def GetNumberOfNodes(self) -> int:
        return len(self.node_ids)

    def GetNodesCoordinates(self) -> jnp.array:
        return self.nodes_coordinates
    
    def GetNodesX(self) -> jnp.array:
        return self.nodes_coordinates[:,0]
    
    def GetNodesY(self) -> jnp.array:
        return self.nodes_coordinates[:,1]
    
    def GetNodesZ(self) -> jnp.array:
        return self.nodes_coordinates[:,2]
    
    def GetElementsIds(self,element_type) -> jnp.array:
        return jnp.arange(len(self.elements_nodes[element_type]))
    
    def GetNumberOfElements(self,element_type) -> jnp.array:
        return len(self.elements_nodes[element_type])

    def GetElementsNodes(self,element_type) -> jnp.array:
        return self.elements_nodes[element_type]
    
    def GetNodeSet(self,set_name) -> jnp.array:
        return self.node_sets[set_name]
    
    def __getitem__(self, key):
        return self.mesh_io.point_data[key]
    
    def __setitem__(self, key, value):
        self.mesh_io.point_data[key] = np.array(value)

    @print_with_timestamp_and_execution_time
    def Finalize(self,export_dir:str=".",export_format:str="vtk") -> None:
        file_name=self.file_name.split('.')[0]+"."+export_format
        self.mesh_io.write(os.path.join(export_dir, file_name),file_format=export_format)

    def __ReadKratosNodes(self, f):
        pos = f.tell()
        num_nodes = 0
        while True:
            line = f.readline().decode()
            if "End Nodes" in line:
                break
            num_nodes += 1
        f.seek(pos)

        nodes_data = np.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
        self.nodes_coordinates = jnp.array(nodes_data[:, 1:] * self.scale_factor)
        self.node_ids = jnp.arange(len(self.nodes_coordinates))

    def __ReadKratosElements(self, f, environ=None):
        mesh_io_element_type = None
        if environ is not None:
            if environ.startswith("Begin Elements "):
                entity_name = environ[15:]
                for key in _mdpa_to_meshio_type:
                    if key in entity_name:
                        mesh_io_element_type = _mdpa_to_meshio_type[key]
                        break
        kr_element_nodes = []          
        while True:
            line = f.readline().decode()
            if line.startswith("End Elements"):
                break
            data = [int(k) for k in filter(None, line.split())]
            num_nodes_per_elem = len(data) - 2

            # Subtract one to account for the fact that python indices are 0-based.
            kr_element_nodes.append(np.array(data[-num_nodes_per_elem:]) - 1)

        if mesh_io_element_type not in self.elements_nodes.keys():
            self.elements_nodes[mesh_io_element_type] = jnp.array(kr_element_nodes)
        else:
            self.elements_nodes[mesh_io_element_type] = jnp.vstack((self.elements_nodes[mesh_io_element_type],
                                                                    jnp.array(kr_element_nodes)))

    def __ReadKratosSubModelPart(self, f, environ=None):
        if environ is not None:
            model_part_name = environ[19:]
        else:
            return 
        node_ids = []
        line = f.readline().decode()
        if line.strip().startswith("Begin SubModelPartNodes"):
            while True:
                line = f.readline().decode()
                if line.strip().startswith("End SubModelPartNodes"):
                    break
                node_ids.append(int(line.strip())-1)

            self.node_sets[model_part_name] = jnp.array(node_ids)
