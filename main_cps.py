import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import math

# =====================================================
# ==============   LOAD CONTROLLER MATRIX   ===========
# =====================================================
def load_matrix_K(path):
    return sci.loadmat(path)['K']

# =====================================================
# ==============   NONLINEAR DRONE MODEL   ============
# =====================================================
def quadrotor_u(x, cu):
    M = 0.6
    L = 0.2159 / 2
    g = 9.81
    m = 0.410
    R = 0.0503513
    m_prop = 0.00311
    m_m = 0.036 + m_prop
    Jx = (2*m*R)/5 + 2*(L**2)*m_m
    Jy = (2*m*R)/5 + 2*(L**2)*m_m
    Jz = (2*m*R)/5 + 4*(L**2)*m_m

    ff = M * g
    cu[0] = cu[0] + ff   # add gravity feed-forward

    phi, theta, psi = x[7], x[9], x[11]
    p, q, r = x[6], x[8], x[10]
    u, v, w = x[0], x[2], x[4]

    du = r*v - q*w - g*math.sin(theta)
    dv = p*w - r*u + g*math.cos(theta)*math.sin(phi)
    dw = q*u - p*v + g*math.cos(theta)*math.cos(phi) - (1/M)*cu[0]
    dp = ((Jy - Jz)/Jx)*q*r + (1/Jx)*cu[1]
    dq = ((Jz - Jx)/Jy)*p*r + (1/Jy)*cu[2]
    dr = ((Jx - Jy)/Jz)*p*q + (1/Jz)*cu[3]

    dx = np.zeros(12)
    dx[0], dx[1] = du, u
    dx[2], dx[3] = dv, v
    dx[4], dx[5] = dw, w
    dx[6], dx[7] = dp, p
    dx[8], dx[9] = dq, q
    dx[10], dx[11] = dr, r
    return dx

# =====================================================
# ==============   FIGURE-8 TRAJECTORY   ==============
# =====================================================
def figure8(t):
    """
    Improved figure-8 trajectory (matches main_multithreaded.py)
    amplitude = 1.0m, period = 6.0s, altitude = 0.5m
    """
    amp = 1.0
    period = 6.0
    z0 = 0.5
    w = 2.0 * math.pi / period

    x = amp * math.sin(w * t)
    y = amp * math.sin(2 * w * t)
    z = z0
    return x, y, z

# =====================================================
# ==============   MAIN SIMULATION   ==================
# =====================================================
if __name__ == "__main__":
    K = load_matrix_K("mat/K.mat")

    Ts = 0.01      # timestep
    T  = 20        # total time (seconds)
    tt = np.arange(0, T+Ts, Ts)
    Ns = tt.size
    n  = 12

    x = np.zeros(n)
    states = np.zeros((Ns, n))

    for j, t in enumerate(tt):
        x_ref = np.zeros(n)
        xr, yr, zr = figure8(t)
        x_ref[1], x_ref[3], x_ref[5] = xr, yr, zr

        cu = -K @ (x - x_ref)
        x = x + quadrotor_u(x, cu)*Ts
        states[j] = x

    # =================================================
    # ==============   PLOTTING RESULTS   =============
    # =================================================
    plt.figure(figsize=(6,6))
    plt.plot(states[:,1], states[:,3], label='Drone path')
    plt.title('UAV Figure-8 Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(True)
    plt.show()
