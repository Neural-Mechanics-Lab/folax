cmake_minimum_required(VERSION 3.15...3.30)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SKBUILD_PROJECT_NAME "fol_ffi_functions")

message(STATUS "SKBUILD_PROJECT_NAME: ${SKBUILD_PROJECT_NAME}")

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

##### find KRATOS_ROOT #####
# Allow user to provide KRATOS_ROOT explicitly with -DKRATOS_ROOT=...
set(KRATOS_ROOT "" CACHE PATH "Path to the Kratos installation")

# If not provided, check environment variable
if(NOT KRATOS_ROOT AND DEFINED ENV{KRATOS_ROOT})
    set(KRATOS_ROOT $ENV{KRATOS_ROOT})
endif()

# Validate
if(KRATOS_ROOT)
    if(EXISTS ${KRATOS_ROOT})
        message(STATUS "Using KRATOS_ROOT: ${KRATOS_ROOT}")
    else()
        message(FATAL_ERROR "KRATOS_ROOT was provided, but the path does not exist: ${KRATOS_ROOT}")
    endif()
else()
    message(FATAL_ERROR "KRATOS_ROOT must be set either as an environment variable or with -DKRATOS_ROOT=/path/to/kratos")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
nanobind_add_module(kr_small_displacement_element NB_STATIC "kr_small_displacement_element.cc")
target_include_directories(kr_small_displacement_element PUBLIC ${XLA_DIR})
target_include_directories(kr_small_displacement_element PRIVATE ${KRATOS_ROOT}/kratos)
target_include_directories(kr_small_displacement_element PRIVATE ${KRATOS_ROOT}/applications/StructuralMechanicsApplication)
target_include_directories(kr_small_displacement_element PRIVATE ${KRATOS_ROOT}/external_libraries)
target_link_libraries(kr_small_displacement_element PRIVATE ${KRATOS_ROOT}/build/Release/kratos/libKratosCore.so)
target_link_libraries(kr_small_displacement_element PRIVATE ${KRATOS_ROOT}/build/Release/applications/StructuralMechanicsApplication/libKratosStructuralMechanicsCore.so)
target_link_libraries(kr_small_displacement_element PRIVATE CUDA::cudart)

set_target_properties(kr_small_displacement_element PROPERTIES
    BUILD_RPATH "${KRATOS_ROOT}/build/Release/kratos;${KRATOS_ROOT}/build/Release/applications/StructuralMechanicsApplication"
    INSTALL_RPATH "${KRATOS_ROOT}/build/Release/kratos;${KRATOS_ROOT}/build/Release/applications/StructuralMechanicsApplication"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
)

install(TARGETS kr_small_displacement_element LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

