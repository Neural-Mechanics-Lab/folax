<p align=center><img height="40.0%" width="40.0%" src="https://github.com/user-attachments/assets/c1759ba8-be9a-48d8-b733-4264e34f411f"></p>

[![License][license-image]][license] 
[![CI](https://github.com/RezaNajian/folax/actions/workflows/CI.yml/badge.svg)](https://github.com/RezaNajian/folax/actions/workflows/CI.yml)
[![PyPI version](https://img.shields.io/pypi/v/folax.svg)](https://pypi.org/project/folax/)


[license-image]: https://img.shields.io/badge/license-BSD-green.svg?style=flat
[license]: https://github.com/RezaNajian/FOL/LICENSE

# Folax: Solution and Optimization of parameterized PDEs
**F**inite **O**perator **L**earning (FOL) with [**JAX**](https://github.com/jax-ml/jax) constitutes a unified numerical framework that seamlessly integrates established numerical methods with advanced scientific machine learning techniques for solving and optimizing parametrized partial differential equations (PDEs).  In constructing a physics-informed operator learning approach, FOL formulates a purely physics-based loss function derived from the Method of Weighted Residuals, allowing discrete residuals—computed using classical PDE solution techniques—to be directly incorporated into backpropagation during network training. This approach ensures that the learned operators rigorously satisfy the underlying governing equations while maintaining consistency with established numerical discretizations. Importantly, this loss formulation is agnostic to the network architecture and has been successfully applied to architectures such as Conditional Neural Fields, Fourier Neural Operators (FNO), and DeepONets. 

FOL has been applied in the following scientific studies:
- A Physics-Informed Meta-Learning Framework for the Continuous Solution of Parametric PDEs on Arbitrary Geometries [[arXiv](https://arxiv.org/abs/2504.02459)].
- Finite Operator Learning: Bridging Neural Operators and Numerical Methods for Efficient Parametric Solution and Optimization of PDEs [[arXiv](https://arxiv.org/abs/2407.04157)].
- Digitalizing metallic materials from image segmentation to multiscale solutions via physics informed operator learning [[npj Computational Materials](https://www.nature.com/articles/s41524-025-01718-y)].
- A Finite Operator Learning Technique for Mapping the Elastic Properties of Microstructures to Their Mechanical Deformations [[Numerical Methods in Eng.](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7637)].
- SPiFOL: A Spectral-based physics-informed finite operator learning for prediction of mechanical behavior of microstructures [[J. Mechanics and Physics of Solids](https://www.sciencedirect.com/science/article/pii/S0022509625001954)].

We built upon several widely adopted Python packages, including [JAX](https://github.com/jax-ml/jax) for high-performance array computations on CPUs and GPUs, [PETSc](https://petsc.org/release/) for the efficient solution of large-scale linear systems, [Metis](https://github.com/KarypisLab/METIS) for mesh partitioning (integration forthcoming), [Flax](https://github.com/google/flax?tab=readme-ov-file) for constructing modular and flexible neural networks, [Optax](https://github.com/google-deepmind/optax) for applying state-of-the-art gradient-based optimization algorithms, and [Orbax](https://github.com/google/orbax) for efficient checkpointing and serialization. This foundation ensures scalability, computational efficiency, and ease of use in large-scale training and simulation workflows.

## Installation
### CPU installation 
To install folax using pip (recommended) for CPU usage you can type the following command

``pip install folax[cpu]``

### GPU installation
To install folax using pip (recommended) for GPU usage you can type the following command

``pip install folax[cuda]``

### Developer installation
If you would like to do development in folax, please first clone the repo and in the folax folder, run the following command

``pip install -e .[cuda,dev]``

### Docker installation
To use folax in a Docker container, you can build and run the container as follows:

1. **Build the Docker image:**
   ```bash
   docker build -t folax-runner .
   ```

2. **Run the container with the project mounted:**
   ```bash
   docker run -d --name folax-setup -v $(pwd):/workspace/folax folax-runner tail -f /dev/null
   ```

3. **Create virtual environment and install dependencies:**
   ```bash
   docker exec folax-setup bash -c "cd /workspace/folax && uv python install 3.12 && uv venv --python 3.12 --clear .venv && uv sync"
   ```

4. **Run your scripts inside the container:**
   ```bash
   docker exec folax-setup bash -c "cd /workspace/folax && source .venv/bin/activate && python your_script.py"
   ```

   Or get an interactive shell:
   ```bash
   docker exec -it folax-setup bash
   ```

5. **Stop and remove the container when done:**
   ```bash
   docker stop folax-setup
   docker rm folax-setup
   ```

## Contributing
If you would like to contribute to the project, please open a pull request with small changes. If you would like to see big changes in the source code, please open an issue or discussion so we can start a conversation.

