This repository contains the Python code that simulates the Quadrotor dynamics controlled by a LQR controller. 
mat/ -contains the LQR controllers (K and Kc) 
src/ -linear_qua_ode.py - linearized quadrotor dynamics controlled by LQR simulated using ODE45 
     -linear_qua_ode_int.py -linearized quadrotor dynamics controlled by LQR + integrator simulated using ODE45 
     -LQR_design.py - Computes the matrices K and Kc 
     -nonlinear_quad_euler.py - simulates the nonlinear model using the Euler's Method controlled by the LQR 
     -nonlinear_quad_euler_int.py - simulates the nonlinear model using the Euler's Method controlled by the LQR + integrator 
     -nonlinear_quad_ode.py - similar to above but, it uses the ODE45 solver 
     -nonlinear_quad_ode_rot.py - Similar to above but, it considers the rotation matrices in the nonlinear model of the quadrotor 
     -reference_governor.py - reference governor implementation on the nonlinear system 
     -reference_governor_lin.py - reference governor implementation on the linearize system 
     -reference_governor_rot.py - reference governor implementation on the nonlinear system and rotation matrices

You can find more details about the controller design in

R. Romagnoli, B. H. Krogh, D. de Niz, A. D. Hristozov and B. Sinopoli, Software Rejuvenation
for Safe Operation of Cyber–Physical Systems in the Presence of Run-Time Cyberattacks, IEEE
Transactions on Control Systems Technology, doi: 10.1109/TCST.2023.3236470.
