import numpy as np
import control as ctrl
import scipy.io as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, cont2discrete
from scipy.integrate import odeint

def quadrotor_init():
    # Quadrotor Parameters
    M = 0.6  # mass (Kg)
    L = 0.2159 / 2  # arm length (m)
    g = 9.81  # acceleration due to gravity (m/s^2)
    m = 0.410  # Sphere mass (Kg)
    R = 0.0503513  # Sphere radius (m)
    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor + propeller mass (Kg)

    # Inertia
    Jx = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jy = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jz = (2 * m * R) / 5 + 4 * L ** 2 * m_m

    # Linearized Model in Hovering Mode
    # Define the A matrix
    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    # Define the B matrix
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [-1 / M, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1 / Jx, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1 / Jy, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1 / Jz],
                  [0, 0, 0, 0]])

    # Define the C matrix
    C = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    n = A.shape[0]
    p = B.shape[1]
    q = C.shape[0]

    D = np.zeros((q, p))

    # LQR Controller
    Ac = np.zeros((q, q))
    Bc = np.eye(q)

    hat_A = np.block([[A, np.zeros((n, q))],
                      [Bc @ C, Ac]])
    hat_B = np.vstack((B, np.zeros((q, p))))

    hat_C = np.hstack((C, np.zeros((q, q))))
    hat_D = np.zeros((q, p))

    folder = '../mat/K.mat'
    folder1 = '../mat/Kc.mat'
    K = load_matrix_K(folder)
    Kc = load_matrix_Kc(folder1)

    # Closed-loop system
    Acl = np.block([[A - B @ K, B @ Kc],
                    [-Bc @ C, Ac]])
    Bcl = np.vstack([np.zeros((n, p)), Bc])

    Ccl = hat_C
    Dcl = np.zeros((q, p))

    # Continuous to discrete conversion
    ts1 = 0.01  # Sampling time
    sys = ctrl.ss(Acl, Bcl, Ccl, Dcl)
    sysd = ctrl.c2d(sys, ts1)

    Ad = sysd.A
    Bd = sysd.B

    return Acl, Bcl, Ad, Bd, K, Kc, ts1

def load_matrix_K(folder):
    # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
    return sci.loadmat(folder)['K']

def load_matrix_Kc(folder):
     # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
     return sci.loadmat(folder)['Kc']

if __name__ == "__main__":
    #Quadrotor Parameters
    M = 0.6  # mass (Kg)
    L = 0.2159 / 2  # arm length (m)
    g = 9.81  # acceleration due to gravity (m/s^2)
    m = 0.410  # Sphere mass (Kg)
    R = 0.0503513  # Sphere radius (m)
    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor + propeller mass (Kg)

    # Inertia
    Jx = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jy = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jz = (2 * m * R) / 5 + 4 * L ** 2 * m_m

    # Linearized Model in Hovering Mode
    # Define the A matrix
    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, g, 0,  0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 0]])



    # Define the B matrix
    B = np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [-1/M, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1/Jx, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1/Jy, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1/Jz],
              [0, 0, 0, 0]])

    # Define the C matrix
    C = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    n = A.shape[0]
    p = B.shape[1]
    q = C.shape[0]

    D = np.zeros((q, p))

    # LQR Controller
    Ac = np.zeros((q, q))
    Bc = np.eye(q)

    hat_A = np.block([[A, np.zeros((n, q))],
                  [Bc @ C, Ac]])
    hat_B = np.vstack((B, np.zeros((q, p))))

    hat_C = np.hstack((C, np.zeros((q, q))))
    hat_D = np.zeros((q, p))

    # LQR Matrices
    Q = 1 * np.eye(hat_A.shape[0])
    R = 1 * np.eye(p)
    K_hat, S, E = ctrl.lqr(hat_A, hat_B, Q, R)
    #K_hat, S, E = np.linalg.lstsq(hat_A, hat_B, rcond=None)

    K = K_hat[:, :n]
    Kc = K_hat[:, n:]


    # Closed-loop system
    Acl = np.block([[A - B @ K, B @ Kc],
                [-Bc @ C, Ac]])
    Bcl = np.vstack([np.zeros((n, p)), Bc])

    # Save both K and Kc into the .mat file
    folder = '../mat/K.mat'
    sci.savemat(folder, {'K': K})
    folder1 = '../mat/Kc.mat'
    sci.savemat(folder1, {'Kc': Kc})