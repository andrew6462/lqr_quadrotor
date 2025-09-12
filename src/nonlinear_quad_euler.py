import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci



def quadrotor_u(x, cu):

    pi = np.pi
    sin = np.sin
    cos = np.cos
    tan = np.tan

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


    # Controls

    ff = M*g
    cu[0] = cu[0] + ff

    # Nonlinear Model


    phi: float = x[7]
    theta: float = x[9]
    psi: float = x[11]

    p = x[6]
    q = x[8]
    r = x[10]

    u = x[0]
    v = x[2]
    w = x[4]

    du = r * v - q * w - g * sin(theta)
    dv = p * w - r * u + g * cos(theta) * sin(phi)
    dw = q * u - p * v + g * cos(theta) * cos(phi) - (1 / M) * cu[0]
    dp = ((Jy - Jz) / Jx) * q * r + (1 / Jx) * cu[1]
    dq = ((Jz - Jx) / Jy) * p * r + (1 / Jy) * cu[2]
    dr = ((Jx - Jy) / Jz) * p * q + (1 / Jz) * cu[3]

    dx = np.zeros(12)
    dx[0] = du
    dx[1] = x[0]
    dx[2] = dv
    dx[4] = dw

    dx[3] = x[2]
    dx[5] = x[4]
    dx[6] = dp
    dx[7] = x[6]
    dx[8] = dq
    dx[9] = x[8]
    dx[10] = dr
    dx[11] = x[10]

    return dx

def load_matrix_K(folder):
    # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
    return sci.loadmat(folder)['K']

if __name__ == "__main__":
    #folder = '../mat/control.mat'
    folder = '../mat/K.mat'
    K = load_matrix_K(folder)
    # Simulation parameters
    Ts = 0.01  # sampling time
    T = 10  # time interval
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

    x_tot = np.zeros((Ns, n))
    for j in range(1, Ns):
        cu = - K@(x0 - x_sp)
        x0 = x0 + quadrotor_u(x0, cu)*Ts
        x_tot[j, :] = x0
    #plots
    # plots

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_tot[:, 1])
    plt.title('x')

    plt.subplot(3, 1, 2)
    plt.plot(x_tot[:, 3])
    plt.title('y')

    plt.subplot(3, 1, 3)
    plt.plot(x_tot[:, 5])
    plt.title('z')

    plt.tight_layout()
    plt.show()