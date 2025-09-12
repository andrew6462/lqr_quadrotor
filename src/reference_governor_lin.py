import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import control as ctrl
from LQR_design import quadrotor_init



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

    #ff = M*g
    #cu[0] = cu[0] + ff

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

def quadrotor_lin_dx(x, cu):

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


    # dx

    dx = A@x+B@cu
    return dx

def load_matrix_K(folder):
    # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
    return sci.loadmat(folder)['K']
def load_matrix_Kc(folder):
    # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
    return sci.loadmat(folder)['Kc']

# RG function
def rg(Hx, Hv, h, r, vk, xx):
    M = Hx.shape[0]
    k = np.zeros(M)
    for j in range(M):
        alpha = Hv[j, :] @ (r - vk)
        beta = h[j] - Hx[j, :] @ xx - Hv[j, :] @ vk
        if alpha > 0:
            k[j] = beta / alpha
        else:
            k[j] = 1
       # if beta < 0:
       #     k[j] = 0
       #     print('negative')
    kappa = min(k)
    return k, kappa

if __name__ == "__main__":
    #folder = '../mat/control.mat'
    #folder = '../mat/K.mat'
    #folder1 = '../mat/Kc.mat'
    #K = load_matrix_K(folder)
    #Kc = load_matrix_Kc(folder1)
    # Simulation parameters
    Ts = 0.001  # sampling time
    T = 5  # time interval
    tt = np.arange(0, T + Ts, Ts)  # vector of time
    Ns = tt.size  # total number of samples
    n = 12  # dimension of the state space
    # Initial Conditions
    x0 = np.zeros(n)
    xc = np.zeros(4)
    # Setpoint definition
    x_sp = np.zeros(12)
    ref = np.zeros(4)
    ref[0] = 5
    ref[1] = 5
    ref[2] = 5
    ref[3] = 0
    Acl, Bcl, Ad, Bd, K, Kc, ts1= quadrotor_init()
    # Reference Governor Initialization
    lstar = 1000
    I = np.eye(16)
    Hx = []
    Hv = []
    h = []
    s = np.array([6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005])
    S = np.block([[-K, Kc], [K, -Kc]] )

    # Build Hx, Hv, and h matrices
    for l in range(1, lstar + 1):
        Hx.append(S @ np.linalg.matrix_power(Ad, l))
        Hv.append(S @ np.linalg.inv(I - Ad) @ (I - np.linalg.matrix_power(Ad, l)) @ Bd)
        h.append(s)

    Hx = np.vstack(Hx + [np.zeros((8, 16))])
    Hv = np.vstack(Hv + [S @ np.linalg.inv(I - Ad) @ Bd])
    h = np.hstack(h + [s * 0.99])

    k_v = []
    vk = 0.01*ref
    xx = np.zeros((16, Ns))
    uu = np.zeros((Ns, 4))
    vv = np.zeros((Ns, 4))
    vv[0, :] = vk
    x_tot = np.zeros((Ns, n))

    for j in range(1, Ns):
        if (j*Ts) % ts1:
            x_old=np.block([x_tot[j-1, :], xc])
            #print(x_old.shape)
            k, kappa = rg(Hx, Hv, h, ref, vk,x_old )
            vk = vk + kappa * (ref-vk)
        cu = - K @ x0 + Kc @ xc
        x0 = x0 + quadrotor_lin_dx(x0, cu)*Ts
        x_tot[j, :] = x0
        uu[j, :] = cu
        vv[j, :] = vk
        e = np.array([vk[0] - x0[1], vk[1] - x0[3], vk[2] - x0[5], vk[3] - x0[11]])
        xc = xc + e * Ts

    #plots

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

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.plot(uu[:, 0])
    plt.title('F')

    plt.subplot(4, 1, 2)
    plt.plot(uu[:, 1])
    plt.title('tau phi')

    plt.subplot(4, 1, 3)
    plt.plot(uu[:, 2])
    plt.title('tau theta')

    plt.subplot(4, 1, 4)
    plt.plot(uu[:, 3])
    plt.title('tau theta')

    plt.tight_layout()
    plt.show()

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.plot(vv[:, 0])
    plt.title('x')

    plt.subplot(4, 1, 2)
    plt.plot(vv[:, 1])
    plt.title('y')

    plt.subplot(4, 1, 3)
    plt.plot(vv[:, 2])
    plt.title('z')

    plt.subplot(4, 1, 4)
    plt.plot(vv[:, 3])
    plt.title('psi')

    plt.tight_layout()
    plt.show()