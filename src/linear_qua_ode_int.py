import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
from scipy.integrate import odeint

def quadrotor_dx(x, t, x_sp, K):

    # Quadrotor's parameters
    M = 0.6  # mass (Kg)
    L = 0.2159 / 2  # arm length (m)

    g = 9.81  # acceleration due to gravity m/s^2

    m = 0.410  # Sphere Mass (Kg)
    R = 0.0503513  # Radius Sphere (m)

    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor +  propeller mass (Kg)
    Jx = (2 * m * R) / 5 + 2 * (L ** 2) * m_m
    Jy = (2 * m * R) / 5 + 2 * (L ** 2) * m_m
    Jz = (2 * m * R) / 5 + 4 * (L ** 2) * m_m

    #Define the A matrix
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


    # Controls
    cu = -K @ (x - x_sp)  # translate the coords to your new estimate. This is not error it is coordinate translation
    dx = A@x+B@cu

    return dx


def load_matrix_K(folder):
    # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
    return sci.loadmat(folder)['K']


if __name__ == "__main__":
    folder = '../mat/control.mat'
    K  = load_matrix_K(folder)
    # Simulation parameters
    Ts = 0.01  # sampling time
    T = 20  # time interval
    tt = np.arange(0, T + Ts, Ts)  # vector of time
    Ns = tt.size  # total number of samples
    n = 12  # dimension of the state space
    # Initial Conditions
    x0 = np.zeros(12)
    # Setpoint definition
    x_sp = np.zeros(12)
    x_sp[1] = 1
    x_sp[3] = 1
    x_sp[5] = 1
    xx = odeint(quadrotor_dx, x0, tt, args=(x_sp, K))

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(xx[:, 1])
    plt.title('x')

    plt.subplot(3, 1, 2)
    plt.plot(xx[:, 3])
    plt.title('y')

    plt.subplot(3, 1, 3)
    plt.plot(xx[:, 5])
    plt.title('z')

    plt.tight_layout()
    plt.show()