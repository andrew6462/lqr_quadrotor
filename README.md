# Quadrotor Dynamics with LQR Control

This repository contains the Python code that simulates quadrotor dynamics controlled by a Linear Quadratic Regulator (LQR) controller.  

## Repository Structure

- **mat/**
  - Contains the precomputed LQR controllers (`K` and `Kc`).

- **src/**
  - `linear_qua_ode.py`  
    Linearized quadrotor dynamics controlled by LQR, simulated using ODE45.
  - `linear_qua_ode_int.py`  
    Linearized quadrotor dynamics with LQR + integrator, simulated using ODE45.
  - `LQR_design.py`  
    Computes the matrices `K` and `Kc` for the LQR controller.
  - `nonlinear_quad_euler.py`  
    Nonlinear quadrotor model controlled by LQR, simulated with Euler’s method.
  - `nonlinear_quad_euler_int.py`  
    Nonlinear quadrotor model with LQR + integrator, simulated with Euler’s method.
  - `nonlinear_quad_ode.py`  
    Nonlinear quadrotor model with LQR, simulated using ODE45.
  - `nonlinear_quad_ode_rot.py`  
    Nonlinear quadrotor model with LQR, simulated using ODE45 and considering full rotation matrices.
  - `reference_governor.py`  
    Reference governor implementation on the nonlinear system.
  - `reference_governor_lin.py`  
    Reference governor implementation on the linearized system.
  - `reference_governor_rot.py`  
    Reference governor implementation on the nonlinear system with rotation matrices.

## Reference

For more details about the controller design, see:  

> R. Romagnoli, B. H. Krogh, D. de Niz, A. D. Hristozov and B. Sinopoli,  
> *Software Rejuvenation for Safe Operation of Cyber–Physical Systems in the Presence of Run-Time Cyberattacks*,  
> IEEE Transactions on Control Systems Technology, 2023.  
> doi: [10.1109/TCST.2023.3236470](https://doi.org/10.1109/TCST.2023.3236470)  

---
