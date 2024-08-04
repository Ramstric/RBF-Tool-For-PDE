# RBF-Tool-For-PDE: Approximate Partial Differential Equations using Radial Basis Functions in Python

> “From the old elements of earth, air, fire, and water to the latest in electrons, quarks, black holes, and superstrings, every inanimate thing in the universe bends to the rule of **differential equations...**” _- Strogatz, S. (2019). Infinite Powers: How Calculus Reveals the Secrets of the Universe. HarperCollins._

This repository contains a Python implementation of the **Radial Basis Function (RBF) Collocation method** for solving **Partial Differential Equations (PDEs)**, as well as a variety of examples.

By definition, the RBF method is a mesh-free method that uses a linear combination of radial basis functions to approximate the solution of a PDE. As such it provides multiple advantages like:

- **Mesh-free:** Unlike other iterative methods like Finite Differences, this method can adapt to different geometries since it does not depend on the construction of a mesh and the computational complexity that entails.
- **Accurate:** It provides high accuracy approximations of the solution of a PDE.
- **Efficient:** The method eliminates the need for an iterative algorithm, and together with the right tools like PyTorch and a high-performance GPU, you can work with millions of data in a matter of milliseconds.
 
This implementation includes the most commonly used radial basis functions, such as Gaussian, Multiquadric and Inverse Multiquadric functions.

The code is written in Python and is designed to be easy to use and modify for different application, easy to understand and to be used as a learning tool.

Please refer to the paper [Aproximando Ecuaciones Diferenciales Parciales con Funciones de Base Radial](https://immediate-family-b23.notion.site/Aproximando-Ecuaciones-Diferenciales-Parciales-con-Funciones-de-Base-Radial-afdd59c15eba4e5c98fd420e8c2ef365?pvs=4) where I explain the theory behind the method in a simple and practical way.

## Table of Contents

- [Usage](#usage)
- [Documentation](#documentation)
  - [Interpolator](#interpolator)
  - [DifferentialEquationSolver](#differentialequationsolver)
- [How to use](#how-to-use)


## Usage

The tools provided are in the scripts:
- `RadialBasisFunctions.py`: Contains multiple radial basis functions and derivatives used in the examples. Please feel free to add more functions.
- `Interpolator.py`: Implements an interpolation method using RBFs, which serves as the basis for the PDE solver. This interpolator scales to $n$ dimensions.
- `DifferentialEquationSolver.py`: Implements the Collocation method for solving ODEs and PDEs.
- `WeightsInterpolator.py`: If needed, an interpolator which uses data points and weights previously calculated. It's meant to ease the computational cost in case of multiple evaluations without needing to recalculate the weights.

## Documentation

### Interpolator

```python
class Interpolator(*r: torch.Tensor, f: torch.Tensor, radius: float, rbf_name: str):
```

A simple interpolator using Radial Basis Functions.

#### Parameters:
- `*r` (torch.Tensor): Tensors containing the data points. Each tensor must be of the same size. The amount of tensors must be equal to the amount of dimensions of the data points i.e. if the interpolation is in 3D, there must be 2 pairs of tensors, sice the third dimension is the function value given in `f`.
- `f` (torch.Tensor): The values of the function at the data points. Same size as the data points.
- `radius` (float): The radius or smoothing parameter of the RBF.
- `rbf_name` (str): The name of the RBF to use. Must be the name of a function in `RadialBasisFunctions.py`.

#### Methods:

| Method                          | Description                                     |
|---------------------------------|-------------------------------------------------|
| `interpolate(*r: torch.Tensor)` | Evaluates the interpolator at the given points. |

### DifferentialEquationSolver

```python
class DifferentialInterpolator(boundary: list[torch.Tensor], inner: list[torch.Tensor], f: torch.Tensor, radius: float, rbf_name: str, derivative_operator)
```

An implementation of the RBF Collocation method for solving PDEs.

#### Parameters:
- `boundary` (list[torch.Tensor]): List of tensors containing the boundary data points. Each tensor must be of the same size.
- `inner` (list[torch.Tensor]): List of tensors containing the inner data points. Each tensor must be of the same size.
- `f` (torch.Tensor): The values of the function at the data points. Same size as the data points.
- `radius` (float): The radius or smoothing parameter of the RBF.
- `rbf_name` (str): The name of the RBF to use. Must be the name of a function in `RadialBasisFunctions.py`.
- `derivative_operator` (function): A function that returns the linear combination of the derivatives of the RBFs according to the differential equation. Remember to derive the same RBF passed in `rbf_name`.

#### Methods:

| Method                          | Description                               |
|---------------------------------|-------------------------------------------|
| `interpolate(*r: torch.Tensor)` | Evaluates the solver at the given points. |

## How to use

The steps of how to use the tools are as follows:

### Interpolator

1. Take note of the data points desired to interpolate and the function values at those points.
2. Create an instance of the `Interpolator` class with its parameters.

```python
from RBF.Interpolator import Interpolator

interpolator = Interpolator(x, y, f=z, radius=0.8, rbf_name="gaussian")
```

3. Prepare the points where you want to evaluate the interpolator.
4. Call the `interpolate` method on another variable.

```python
z_interpolated = interpolator.interpolate(x_interpolated, y_interpolated)
```

Please refer to the examples in the `interpolation` folder.

### DifferentialEquationSolver

1. Generate the data points for the boundary and inner points of the domain. The boundary points are given by the boundary operator dictated by the conditions of the PDE and the inner points are given by the equation itself. 
2. Establish the order of the variables passed to the `DifferentialEquationSolver` class i.e. $[x, t]$ then the boundary points will be $[x_{boundary}, t_{boundary}]$ and the inner points will be $[x_{inner}, t_{inner}]$.
3. Define the derivative operator according to the differential equation

```python
def derivative_operator(r, radius, variables=None):
    # For this code variables are [x, t] in that order
    return rbf.multiquadric_t(r, variables[1], radius) - rbf.multiquadric_xx(r, variables[1], radius)
```

Note why the order of the variables is important.

4. Create an instance of the `DifferentialEquationSolver` class with its parameters.

```python
from RBF import DifferentialEquationSolver

DEInterpolator = DifferentialEquationSolver.DifferentialInterpolator(boundary=[boundary_x, boundary_time], inner=[inner_x, inner_time],
                                                   f=u, radius=0.2, rbf_name="multiquadric",
                                                   derivative_operator=derivative_operator)
```

5. Prepare the points where you want to evaluate the interpolator. Make sure to generate a meshgrid or similar if needed.
6. Call the `interpolate` method on another variable.

```python
u_interpolated = DEInterpolator.interpolate(x_interpolated, time_interpolated)
```