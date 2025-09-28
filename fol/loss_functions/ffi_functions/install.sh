export nanobind_DIR=/home/reza/jax-ai-stack/jax_ffi/jax_ffi_py_env/lib/python3.10/site-packages/nanobind &&
export KRATOS_ROOT=/home/reza/Kratos &&
rm -rf Build &&
cmake -B Build -DCMAKE_CXX_COMPILER=g++-10  -DCMAKE_CUDA_FLAGS="-ccbin g++-10" &&
cmake --build Build  &&
pip install .
