/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <memory>
#include <tuple>
#include <omp.h>
#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "containers/model.h"
#include "structural_mechanics_application.h"
#include "structural_mechanics_application_variables.h"
#include "geometries/tetrahedra_3d_4.h"
#include "utilities/variable_utils.h"
#include "utilities/dof_utilities/dof_array_utilities.h"
#include "utilities/parallel_utilities.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

struct State {
  static xla::ffi::TypeId id;
  // explicit State(int32_t value) : value(value) {}
  int32_t value=42;
  bool initialized = false;
  Kratos::Model kr_cpu_model;
  int num_dofs_per_node = 3;
  int num_nodes_per_elem = 4;
};
ffi::TypeId State::id = {};

static ffi::ErrorOr<std::unique_ptr<State>> StateInstantiate() {
  return std::make_unique<State>();
}

#define ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                      \
  switch (element_type) {                                                 \
    case ffi::F32:                                                        \
      return fn<float>(__VA_ARGS__);                                      \
    case ffi::F64:                                                        \
      return fn<double>(__VA_ARGS__);                                     \
    default:                                                              \
      return ffi::Error::InvalidArgument("Unsupported input data type."); \
}

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
//  this function returns the total number of elements in the buffer, and the size of
// the first and last dimension.
std::tuple<int64_t, int64_t, int64_t> GetDims(const ffi::AnyBuffer buffer) {
  const ffi::AnyBuffer::Dimensions dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::tuple<int64_t, int64_t, int64_t>(0, 0, 0);
  }
  return std::tuple<int64_t, int64_t, int64_t> (buffer.element_count(),dims.front(), dims.back());
}

template <typename T>
ffi::Error ComputeSampleElements(cudaStream_t stream, 
                                 State* state,
                                 int sample_start_id,
                                 int sample_size,
                                 std::vector<T>& batch_res_host) {
  auto& r_model_part = state->kr_cpu_model.GetModelPart("ModelPart");

  Kratos::Matrix lhs;
  Kratos::Vector rhs;
  Kratos::Element::EquationIdVectorType  elem_eqs_ids;
  #pragma omp parallel for
  for(int elem_id=0;elem_id<r_model_part.NumberOfElements();elem_id++){

    auto& this_elem =  r_model_part.GetElement(elem_id+1);
    this_elem.CalculateLocalSystem(lhs, rhs, r_model_part.GetProcessInfo());
    //get global dof ids
    this_elem.EquationIdVector(elem_eqs_ids,r_model_part.GetProcessInfo());
    // atomic add
    for (std::size_t i=0; i<elem_eqs_ids.size(); i++) {
        int gid = elem_eqs_ids[i];
        #pragma omp atomic
        batch_res_host[sample_start_id+gid] -= rhs[i]; //here we need to multiply with -1 because kratos multiplies with -1 internally
    }
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error CreateKratosModel(cudaStream_t stream, 
                               State* state, 
                               ffi::AnyBuffer coordinates,
                               ffi::Buffer<ffi::S32> elems_nodes,
                               ffi::AnyBuffer element_properties) {
 if (!state->initialized){
    using namespace Kratos;
    T *coordinates_data = coordinates.typed_data<T>();
    auto [total_num_coords, total_num_nodes, domain_dim] = GetDims(coordinates);
    std::vector<T> coord_data_cp_host(total_num_coords);
    cudaMemcpy(coord_data_cp_host.data(), coordinates_data, total_num_coords * sizeof(T), cudaMemcpyDeviceToHost);

    //get elements nodes
    int number_elements = elems_nodes.dimensions().front();
    int num_nodes_per_elem = elems_nodes.dimensions().back();
    std::vector<int32_t> elems_nodes_cp_host(elems_nodes.element_count());
    cudaMemcpy(elems_nodes_cp_host.data(), elems_nodes.typed_data(), elems_nodes.element_count() * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // now get properties
    int num_properties = element_properties.element_count();
    std::vector<T> element_properties_cp_host(num_properties);
    cudaMemcpy(element_properties_cp_host.data(), element_properties.typed_data<T>(), num_properties * sizeof(T), cudaMemcpyDeviceToHost);

    state->kr_cpu_model.CreateModelPart("ModelPart", 1);
    auto& r_model_part = state->kr_cpu_model.GetModelPart("ModelPart");
    r_model_part.GetProcessInfo().SetValue(DOMAIN_SIZE,int(domain_dim));

    r_model_part.AddNodalSolutionStepVariable(DISPLACEMENT);
    r_model_part.AddNodalSolutionStepVariable(REACTION);
    r_model_part.AddNodalSolutionStepVariable(VOLUME_ACCELERATION);

    // // Set the element properties
    auto p_elem_prop = r_model_part.CreateNewProperties(0);
    p_elem_prop->SetValue(YOUNG_MODULUS,element_properties_cp_host[1]);
    p_elem_prop->SetValue(POISSON_RATIO,element_properties_cp_host[0]);
    p_elem_prop->SetValue(THICKNESS,1.0);
    p_elem_prop->SetValue(DENSITY,1.0);

    static KratosStructuralMechanicsApplication sm_app;
    sm_app.Register();
    auto p_constitutive_law = KratosComponents<ConstitutiveLaw>().Get("LinearElastic3DLaw").Clone();

    p_elem_prop->SetValue(CONSTITUTIVE_LAW,p_constitutive_law);

    // create nodes
    for (int n = 0; n < total_num_nodes; n += 1)
      r_model_part.CreateNewNode(n, coord_data_cp_host[n*domain_dim+0], coord_data_cp_host[n*domain_dim+1], coord_data_cp_host[n*domain_dim+2]);

    // add Dofs
    VariableUtils().AddDofWithReaction(DISPLACEMENT_X, REACTION_X,r_model_part);
    VariableUtils().AddDofWithReaction(DISPLACEMENT_Y, REACTION_Y,r_model_part);
    VariableUtils().AddDofWithReaction(DISPLACEMENT_Z, REACTION_Z,r_model_part);

    // create elements
    for (int elem_id=0;elem_id<number_elements;elem_id++){
      std::vector<Kratos::ModelPart::IndexType> element_nodes;
      for (int e_node_idx=0;e_node_idx<num_nodes_per_elem;e_node_idx++)
        element_nodes.push_back(elems_nodes_cp_host[elem_id*num_nodes_per_elem+e_node_idx]);

      auto p_element = r_model_part.CreateNewElement(
          "SmallDisplacementElement3D4N", elem_id+1, element_nodes, p_elem_prop);
      
      p_element->Initialize(r_model_part.GetProcessInfo());
      p_element->Check(r_model_part.GetProcessInfo());
    }

    //setup dof list and equation ids
    Kratos::DofArrayUtilities::DofsArrayType dofs_array;
    Kratos::DofArrayUtilities::SetUpDofArray(r_model_part, dofs_array);
    Kratos::IndexPartition<std::size_t>(dofs_array.size()).for_each([&](std::size_t Index){
        typename Kratos::DofArrayUtilities::DofsArrayType::iterator dof_iterator = dofs_array.begin() + Index;
        dof_iterator->SetEquationId(Index);
    });
    
    state->initialized = true;
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error ComputeNodalResidualsImpl(cudaStream_t stream, 
                               State* state, 
                               ffi::AnyBuffer coordinates,
                               ffi::Buffer<ffi::S32> elems_nodes,
                               ffi::AnyBuffer element_properties,
                               ffi::AnyBuffer batch_nodal_solution,
                               ffi::Result<ffi::AnyBuffer> batch_nodal_res) {

  CreateKratosModel<T>(stream,state,coordinates,elems_nodes,element_properties);
 
  using namespace Kratos;
  auto [total_sol_size, sol_first_dim, sol_last_dim] = GetDims(batch_nodal_solution);
  T *batch_nodal_solution_data = batch_nodal_solution.typed_data<T>();
  std::vector<T> batch_nodal_solution_data_cp_host(total_sol_size);
  cudaMemcpyAsync(batch_nodal_solution_data_cp_host.data(), batch_nodal_solution_data, total_sol_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::vector<T> batch_res_host(total_sol_size,0.0);
  auto& r_model_part = state->kr_cpu_model.GetModelPart("ModelPart");
  auto num_dofs_per_node = state->num_dofs_per_node;
  //batch loop starts here
  for (int64_t n = 0, batch_idx = 0; n < total_sol_size; n += sol_last_dim, ++batch_idx) {
    int64_t sample_sol_start_id = n;
    // set the sample solution
    #pragma omp parallel for
    for(int node_id=0;node_id<r_model_part.NumberOfNodes();node_id++){
      auto& v = r_model_part.GetNode(node_id).FastGetSolutionStepValue(DISPLACEMENT);
      for(int dof_index=0;dof_index<num_dofs_per_node;dof_index++)
        v[dof_index] = batch_nodal_solution_data_cp_host[sample_sol_start_id+(node_id*num_dofs_per_node)+dof_index];
    }
    // compute elements for this sample
    ComputeSampleElements<T>(stream, state, sample_sol_start_id, sol_last_dim, batch_res_host);
  }  

  //load the results to gpu
  cudaMemcpyAsync(batch_nodal_res->typed_data<T>(), batch_res_host.data(), total_sol_size * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error ComputeElementsImpl(cudaStream_t stream, 
                               State* state, 
                               ffi::AnyBuffer coordinates,
                               ffi::Buffer<ffi::S32> elems_nodes,
                               ffi::AnyBuffer element_properties,
                               ffi::AnyBuffer nodal_solution,
                               ffi::Result<ffi::AnyBuffer> elements_lhs,
                               ffi::Result<ffi::AnyBuffer> elements_rhs) {

  CreateKratosModel<T>(stream,state,coordinates,elems_nodes,element_properties);
 
  using namespace Kratos;
  auto [total_sol_size, sol_first_dim, sol_last_dim] = GetDims(nodal_solution);
  if (total_sol_size != sol_first_dim ||  sol_first_dim!=sol_last_dim) {
    return ffi::Error::InvalidArgument("nodal_solution input must be an array");
  }

  T *nodal_solution_data = nodal_solution.typed_data<T>();
  std::vector<T> nodal_solution_data_cp_host(total_sol_size);
  cudaMemcpyAsync(nodal_solution_data_cp_host.data(), nodal_solution_data, total_sol_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  auto& r_model_part = state->kr_cpu_model.GetModelPart("ModelPart");
  auto num_dofs_per_node = state->num_dofs_per_node;
  //set the nodal solution
  #pragma omp parallel for
  for(int node_id=0;node_id<r_model_part.NumberOfNodes();node_id++){
    auto& v = r_model_part.GetNode(node_id).FastGetSolutionStepValue(DISPLACEMENT);
    for(int dof_index=0;dof_index<num_dofs_per_node;dof_index++)
      v[dof_index] = nodal_solution_data_cp_host[(node_id*num_dofs_per_node)+dof_index];
  } 

  Kratos::Matrix lhs;
  Kratos::Vector rhs;
  int elem_num_dofs = state->num_dofs_per_node * state->num_nodes_per_elem;
  auto num_elements = r_model_part.NumberOfElements();
  std::vector<T> elems_rhs_host(elem_num_dofs*num_elements);
  std::vector<T> elems_lhs_host(elem_num_dofs*elem_num_dofs*num_elements);
  #pragma omp parallel for
  for(int elem_id=0;elem_id<num_elements;elem_id++){
    auto& this_elem =  r_model_part.GetElement(elem_id+1);
    this_elem.CalculateLocalSystem(lhs, rhs, r_model_part.GetProcessInfo());
    for (std::size_t i=0; i<elem_num_dofs; i++) {
      int rhs_elem_dof_id = elem_id * elem_num_dofs + i;
      elems_rhs_host[rhs_elem_dof_id] = -rhs[i]; //here we need to multiply with -1 because kratos multiplies with -1 internally
      for (std::size_t j=0; j<elem_num_dofs; j++){
        int lhs_elem_dof_id = elem_id * elem_num_dofs * elem_num_dofs + i * elem_num_dofs +j;
        elems_lhs_host[lhs_elem_dof_id] = lhs(i,j);
      }
    }
  }

  //load the results to gpu
  cudaMemcpyAsync(elements_rhs->typed_data<T>(), elems_rhs_host.data(), elems_rhs_host.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(elements_lhs->typed_data<T>(), elems_lhs_host.data(), elems_lhs_host.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

static ffi::Error ComputeNodalResidualsExecute(cudaStream_t stream, 
                               State* state, 
                               ffi::AnyBuffer node_coordinates,
                               ffi::Buffer<ffi::S32> elements_nodes,
                               ffi::AnyBuffer element_properties,
                               ffi::AnyBuffer nodal_solution,
                               ffi::Result<ffi::AnyBuffer> nodal_res) {
  ELEMENT_TYPE_DISPATCH(node_coordinates.element_type(), ComputeNodalResidualsImpl, stream, state, node_coordinates, elements_nodes, element_properties, nodal_solution, nodal_res);
}

static ffi::Error ComputeElementsExecute(cudaStream_t stream, 
                               State* state, 
                               ffi::AnyBuffer node_coordinates,
                               ffi::Buffer<ffi::S32> elements_nodes,
                               ffi::AnyBuffer element_properties,
                               ffi::AnyBuffer nodal_solution,
                               ffi::Result<ffi::AnyBuffer> elements_lhs,
                               ffi::Result<ffi::AnyBuffer> elements_rhs) {
  ELEMENT_TYPE_DISPATCH(node_coordinates.element_type(), ComputeElementsImpl, stream, state, node_coordinates, elements_nodes, element_properties, nodal_solution, elements_lhs, elements_rhs);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kComputeNodalResidualsExecute, ComputeNodalResidualsExecute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::State<State>>()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::Buffer<ffi::S32>>()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kComputeElementsExecute, ComputeElementsExecute,
                       ffi::Ffi::Bind()
                      .Ctx<ffi::PlatformStream<cudaStream_t>>()
                      .Ctx<ffi::State<State>>()
                      .Arg<ffi::AnyBuffer>()
                      .Arg<ffi::Buffer<ffi::S32>>()
                      .Arg<ffi::AnyBuffer>()
                      .Arg<ffi::AnyBuffer>()
                      .Ret<ffi::AnyBuffer>()
                      .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kStateInstantiate, StateInstantiate,
                       ffi::Ffi::BindInstantiate());

NB_MODULE(kr_small_displacement_element, m) {

  m.def("type_id",
        []() { return nb::capsule(reinterpret_cast<void*>(&State::id)); });
  m.def("compute_nodal_residuals", []() {
    nb::dict d;
    d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kStateInstantiate));
    d["execute"] = nb::capsule(reinterpret_cast<void*>(kComputeNodalResidualsExecute));
    return d;
  });
  m.def("compute_elements", []() {
    nb::dict d;
    d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kStateInstantiate));
    d["execute"] = nb::capsule(reinterpret_cast<void*>(kComputeElementsExecute));
    return d;
  });
}
