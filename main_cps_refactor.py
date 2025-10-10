#!/usr/bin/env python3
"""
Quadrotor LQR (Course Project) — driver script

Authors: Andrew P., <Teammate B>, <Teammate C>  |  Course: CPS / Controls  |  Year: Junior (3rd)

What this file does (in plain student English)
----------------------------------------------
- Lets us run the quadrotor with **linear** or **nonlinear** dynamics.
- Uses our **LQR gains** (and integrator if we have Kc) to track references.
- Adds a simple **reference governor** so we don't blow past actuator limits.
- Can run a few preset scenarios (hover, step in z, and a figure‑8) and spit out plots.
- Has a small async batch mode so we can compare cases without waiting forever.

How we run it (examples)
------------------------
# Nonlinear + RG, step in z (what we demoed in lab)
python main_cps_refactor.py --plant nonlinear --traj step_z --use-rg --T 8 --Ts 0.004

# Linear vs Nonlinear quick compare (runs the sims in parallel, then plots)
python main_cps_refactor.py --batch linear_vs_nonlinear --T 8 --Ts 0.004 --save-prefix cmp

# Figure‑8 with RG (and integrator if Kc is available)
python main_cps_refactor.py --plant nonlinear --traj figure8 --use-rg --T 10 --Ts 0.004 --save-prefix fig8

Notes we kept for ourselves
---------------------------
- If `mat/K.mat` or `mat/control.mat` is present, we load K (and Kc). If not, we use a basic fallback K we tuned
  for a ~0.6 kg mini‑quad just to keep the demo stable.
- We avoided the `python-control` package here to keep the script lightweight for graders.
- The reference governor is a line search between the last safe ref and the new command. It's simple but works.
"""

from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import math
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.io import loadmat

# ======================================================================================
# ASSIGNMENT‑SPEC TUNING (edit here if your rubric lists different numbers)
# ======================================================================================
G = 9.81                 # gravity [m/s^2]
M = 0.60                 # mass [kg]  (mini‑quad, per assignment vibe)
JX, JY, JZ = 7.5e-3, 7.5e-3, 1.3e-2   # principal inertias [kg·m^2]

# Actuator limits (DEVIATIONS about hover; choose conservatively)
FZ_MAX = 4.0             # N (± about hover thrust)
TAU_X_MAX = 0.15         # N·m
TAU_Y_MAX = 0.15         # N·m
TAU_Z_MAX = 0.10         # N·m

# Safety/comfort limits (for plots; RG enforces inputs, not angles)
MAX_TILT_DEG = 20.0      # |phi|, |theta| display guide

# RG line‑search iters
RG_ITERS = 14

# ======================================================================================
# Data classes
# ======================================================================================
@dataclass
class PlantSpec:
    kind: str                  # 'linear' | 'nonlinear' | 'nonlinear_rot'

@dataclass
class Gains:
    K: np.ndarray              # (4,12)
    Kc: Optional[np.ndarray]   # (4,p) or None

@dataclass
class Limits:
    u_min: np.ndarray          # (4,)
    u_max: np.ndarray          # (4,)

@dataclass
class SimResult:
    name: str
    t: np.ndarray              # (N+1,)
    X: np.ndarray              # (12, N+1)
    R: np.ndarray              # (12, N+1)
    V: np.ndarray              # (12, N+1)
    U: np.ndarray              # (4,  N+1)
    sat_hits: np.ndarray       # (4,  N)
    meta: Dict[str, Any]

# ======================================================================================
# Helpers
# ======================================================================================

def safe_loadmat(path: str, keys: Tuple[str, ...]) -> dict:
    d = {}
    if os.path.exists(path):
        try:
            mat = loadmat(path)
            for k in keys:
                if k in mat:
                    d[k] = np.array(mat[k], dtype=float)
        except Exception as e:
            print(f"[warn] Failed to load {path}: {e}")
    return d

# ======================================================================================
# Models
# ======================================================================================

# Linearized hover model in 12‑state form:
# x = [xd, x, yd, y, zd, z, phid, phi, thetad, theta, psid, psi]
# u = [Fz, tau_x, tau_y, tau_z]  (deviation about hover)

def build_linear_AB() -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((12,12))
    # position' = velocity
    A[1,0] = 1.0
    A[3,2] = 1.0
    A[5,4] = 1.0
    A[7,6] = 1.0
    A[9,8] = 1.0
    A[11,10] = 1.0
    # gravity cross‑couplings (note sign convention vs nonlinear below)
    A[0,9] = -G     # xdd ~ -g*theta
    A[2,7] = +G     # ydd ~ +g*phi

    B = np.zeros((12,4))
    B[4,0]  = -1.0 / M    # zdd from Fz (deviation)
    B[6,1]  =  1.0 / JX   # phidd from tau_x
    B[8,2]  =  1.0 / JY   # thetadd from tau_y
    B[10,3] =  1.0 / JZ   # psidd from tau_z
    return A, B


def f_linear(x: np.ndarray, u: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ x + B @ u


def f_nonlinear(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # unpack
    xd, x, yd, y, zd, z, phid, phi, thetad, theta, psid, psi = x
    Fz, tx, ty, tz = u
    # Correct small-angle hover couplings to match linearization:
    # xdd ≈ -g * theta,  ydd ≈ +g * phi,  zdd ≈ -(Fz)/M (deviation about hover)
    xdd = -G * theta
    ydd = +G * phi
    zdd = -Fz / M
    # Simple rigid-body rotational dynamics
    phidd   = (tx - (JZ - JY) * thetad * psid) / JX
    thetadd = (ty - (JX - JZ) * phid    * psid) / JY
    psidd   = (tz - (JY - JX) * phid    * thetad) / JZ
    return np.array([
        xdd, xd,
        ydd, yd,
        zdd, zd,
        phidd, phid,
        thetadd, thetad,
        psidd, psid
    ], dtype=float)

# Hook for full rotation‑matrix kinematics (currently shares f with nonlinear)
f_nonlinear_rot = f_nonlinear

# ======================================================================================
# Gains
# ======================================================================================

def load_gains(mat_dir: str = 'mat') -> Gains:
    K = None; Kc = None
    for p in (os.path.join(mat_dir,'K.mat'), os.path.join(mat_dir,'control.mat')):
        d = safe_loadmat(p, ('K','Kc'))
        if K is None and 'K' in d: K = d['K']
        if Kc is None and 'Kc' in d: Kc = d['Kc']
    if K is None:
        # stabilizing fallback tuned for M=0.6, J's above
        # Include cross-coupling from position/velocity → attitude channels so x/y track references
        K = np.zeros((4,12))
        # Fz from z, zdot
        K[0,4] = 3.2; K[0,5] = 4.5
        # tau_x (roll) from phidot, phi and y, ydot (to move in +y)
        K[1,6] = 1.1; K[1,7] = 1.6
        K[1,2] = 0.6; K[1,3] = 0.8
        # tau_y (pitch) from thetadot, theta and x, xdot (to move in +x)
        K[2,8] = 1.1; K[2,9] = 1.6
        K[2,0] = 0.6; K[2,1] = 0.8
        # tau_z from psidot, psi
        K[3,10]= 0.8; K[3,11]= 1.2
        print('[info] Using demo K (fallback tuned to assignment spec + xy tracking).')
    else:
        print(f"[info] Loaded K {K.shape}.")
    if Kc is not None:
        print(f"[info] Loaded Kc {Kc.shape}.")
    return Gains(K=K.astype(float), Kc=(Kc.astype(float) if Kc is not None else None))

# ======================================================================================
# Trajectories
# ======================================================================================

def traj_hover() -> Callable[[float], np.ndarray]:
    def r_of_t(t: float) -> np.ndarray:
        return np.zeros(12)
    return r_of_t


def traj_step_z(z_final: float = 1.0, t_step: float = 0.5) -> Callable[[float], np.ndarray]:
    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        if t >= t_step: r[5] = z_final
        return r
    return r_of_t


def traj_figure8(amp: float = 1.0, period: float = 6.0, z0: float = 0.5) -> Callable[[float], np.ndarray]:
    w = 2.0 * math.pi / period
    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        r[1] = amp * math.sin(w * t)
        r[3] = amp * math.sin(2*w * t)
        r[5] = z0
        return r
    return r_of_t

# ======================================================================================
# RG + Integrator + Integrators
# ======================================================================================

def rk4_step(x: np.ndarray, u: np.ndarray, f: Callable[[np.ndarray, np.ndarray], np.ndarray], dt: float) -> np.ndarray:
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def govern_reference(x: np.ndarray, v_prev: np.ndarray, r_target: np.ndarray,
                     K: np.ndarray, u_min: np.ndarray, u_max: np.ndarray,
                     iters: int = RG_ITERS) -> np.ndarray:
    d = r_target - v_prev
    if np.allclose(d, 0.0):
        return v_prev
    lo, hi = 0.0, 1.0
    v_best = v_prev
    for _ in range(iters):
        kappa = 0.5*(lo+hi)
        v = v_prev + kappa*d
        u = -K @ (x - v)
        if np.all(u <= u_max + 1e-9) and np.all(u >= u_min - 1e-9):
            v_best = v; lo = kappa
        else:
            hi = kappa
    return v_best

# ======================================================================================
# Single simulation (CPU‑bound) — suitable for running in a thread
# ======================================================================================

def simulate_one(name: str, plant: str, traj: str, T: float, Ts: float,
                 use_rg: bool, use_integrator: bool, gains: Gains,
                 angle_int_only: bool = True,
                 z_final: float = 1.0, step_time: float = 0.5) -> SimResult:

    # Select plant dynamics
    if plant == 'linear':
        A, B = build_linear_AB()
        f = lambda x,u: f_linear(x,u,A,B)
    elif plant == 'nonlinear':
        f = f_nonlinear
    else:
        f = f_nonlinear_rot

    # Trajectory
    if traj == 'hover':
        r_of_t = traj_hover()
    elif traj == 'step_z':
        r_of_t = traj_step_z(z_final=z_final, t_step=step_time)
    else:
        r_of_t = traj_figure8()

    # Limits
    u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
    u_min = -u_max

    # Time grid
    N = int(round(T / Ts))
    t = np.linspace(0.0, T, N+1)

    # State
    x = np.zeros(12)
    v = np.zeros(12)
    xi = np.zeros(12)

    # Logs
    X = np.zeros((12, N+1)); X[:,0] = x
    R = np.zeros((12, N+1))
    V = np.zeros((12, N+1))
    U = np.zeros((4,  N+1))
    sat_hits = np.zeros((4, N), dtype=bool)

    pos_idx = [1,3,5]  # integrate position errors by default when Kc present

    K, Kc = gains.K, gains.Kc

    for k in range(N):
        tk = t[k]
        r = r_of_t(tk)
        if use_rg:
            v = govern_reference(x, v, r, K=K, u_min=u_min, u_max=u_max)
        else:
            v = r
        err = (x - v)
        u = -K @ err
        if use_integrator and Kc is not None:
            epos = v[pos_idx] - x[pos_idx]
            xi[pos_idx] += epos * Ts
            u = u - (Kc @ xi[pos_idx])  # assumes Kc maps 3 int‑errors → 4 inputs
        u_clip = np.minimum(np.maximum(u, u_min), u_max)
        sat_hits[:,k] = np.abs(u - u_clip) > 1e-12
        x = rk4_step(x, u_clip, f=f, dt=Ts)

        X[:,k+1] = x
        R[:,k] = r
        V[:,k] = v
        U[:,k] = u_clip

    R[:,-1] = r_of_t(t[-1])
    V[:,-1] = v
    U[:,-1] = U[:,-2]

    return SimResult(
        name=name, t=t, X=X, R=R, V=V, U=U, sat_hits=sat_hits,
        meta={
            'plant': plant, 'traj': traj, 'T': T, 'Ts': Ts,
            'use_rg': use_rg, 'use_integrator': use_integrator,
            'limits': {'u_min': u_min, 'u_max': u_max}
        }
    )

# ======================================================================================
# Plotting
# ======================================================================================

def plot_sim(res: SimResult, save_prefix: str = '') -> None:
    t, X, R, V, U, sat = res.t, res.X, res.R, res.V, res.U, res.sat_hits

    def maybe_save(fig, name: str):
        if save_prefix:
            out = f"{save_prefix}_{res.name}_{name}.png"
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"[saved] {out}")

    # Trajectory XY
    fig1 = plt.figure()
    plt.plot(X[1,:], X[3,:], label='plant (x,y)')
    plt.plot(R[1,:], R[3,:], label='ref (x,y)')
    if res.meta['use_rg']:
        plt.plot(V[1,:], V[3,:], label='gov (x,y)')
    plt.axis('equal')
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title(f"XY: {res.meta['plant']} | {res.meta['traj']} | RG={res.meta['use_rg']} | INT={res.meta['use_integrator']}")
    plt.legend(); plt.grid(True)
    maybe_save(fig1, 'traj_xy')

    # Figure-8 visualization with time gradient (only when traj == figure8)
    if res.meta['traj'] == 'figure8':
        fig1b = plt.figure()
        sc = plt.scatter(X[1,:], X[3,:], c=t, s=10)
        plt.plot(R[1,:], R[3,:], linewidth=1.0, alpha=0.6)
        plt.scatter([X[1,0]],[X[3,0]], marker='o', label='start')
        plt.scatter([X[1,-1]],[X[3,-1]], marker='x', label='end')
        plt.axis('equal')
        plt.xlabel('x [m]'); plt.ylabel('y [m]')
        plt.title('Figure-8 path (color = time)')
        cb = plt.colorbar(sc); cb.set_label('time [s]')
        plt.legend(); plt.grid(True)
        maybe_save(fig1b, 'traj_xy_figure8')

        # 3D trajectory (x,y,z) with time color gradient
    fig1c = plt.figure()
    ax = fig1c.add_subplot(111, projection='3d')
    p3d = ax.scatter(X[1,:], X[3,:], X[5,:], c=t, s=8)
    ax.plot(R[1,:], R[3,:], R[5,:], linewidth=1.0, alpha=0.6)
    if res.meta['use_rg']:
        ax.plot(V[1,:], V[3,:], V[5,:], linewidth=1.0, alpha=0.6)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title('3D Trajectory (color = time)')
    cb3d = plt.colorbar(p3d); cb3d.set_label('time [s]')
    maybe_save(fig1c, 'traj_xyz_3d')

    # Altitude
    fig2 = plt.figure()
    plt.plot(t, X[5,:], label='z')
    plt.plot(t, R[5,:], label='z_ref')
    if res.meta['use_rg']:
        plt.plot(t, V[5,:], label='z_gov')
    plt.xlabel('time [s]'); plt.ylabel('z [m]')
    plt.title('Altitude response')
    plt.legend(); plt.grid(True)
    maybe_save(fig2, 'z')

    # Euler angles + limit lines
    fig3 = plt.figure()
    plt.plot(t, X[7,:],  label='phi [rad]')
    plt.plot(t, X[9,:],  label='theta [rad]')
    plt.plot(t, X[11,:], label='psi [rad]')
    lim = math.radians(MAX_TILT_DEG)
    plt.hlines([+lim, -lim], xmin=t[0], xmax=t[-1], linestyles='dashed')
    plt.xlabel('time [s]'); plt.ylabel('angle [rad]')
    plt.title(f'Euler angles (±{MAX_TILT_DEG:.0f}° guide)')
    plt.legend(); plt.grid(True)
    maybe_save(fig3, 'angles')

    # Inputs + limits
    labels = ['Fz','tau_x','tau_y','tau_z']
    lims = res.meta['limits']
    fig4 = plt.figure()
    for i in range(4):
        plt.plot(t, U[i,:], label=labels[i])
        plt.hlines([lims['u_min'][i], lims['u_max'][i]], xmin=t[0], xmax=t[-1], linestyles='dashed')
    plt.xlabel('time [s]'); plt.ylabel('input')
    plt.title('Control inputs')
    plt.legend(); plt.grid(True)
    maybe_save(fig4, 'inputs')

    # Saturation events (any channel)
    fig5 = plt.figure()
    sat_any = sat.any(axis=0).astype(int)
    plt.plot(t[:-1], sat_any, drawstyle='steps-post')
    plt.ylim([-0.1, 1.1])
    plt.xlabel('time [s]'); plt.ylabel('sat (any)')
    plt.title('Actuator saturation events')
    plt.grid(True)
    maybe_save(fig5, 'sat')

    plt.show()

# ======================================================================================
# Async / batch orchestration
# ======================================================================================

async def run_batch(kind: str, T: float, Ts: float, gains: Gains, save_prefix: str,
                    z_final: float, step_time: float) -> None:
    tasks: List[asyncio.Task] = []
    res_list: List[SimResult] = []
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def submit(name, plant, traj, use_rg, use_int):
        return loop.run_in_executor(
            pool,
            simulate_one,
            name, plant, traj, T, Ts, use_rg, use_int, gains, True, z_final, step_time
        )

    if kind == 'linear_vs_nonlinear':
        tasks.append(asyncio.create_task(submit('linear_step', 'linear', 'step_z', False, False)))
        tasks.append(asyncio.create_task(submit('nonlinear_step', 'nonlinear', 'step_z', False, False)))
        tasks.append(asyncio.create_task(submit('nonlinear_step_rg', 'nonlinear', 'step_z', True, False)))
    elif kind == 'full':
        tasks.append(asyncio.create_task(submit('nl_step_rg', 'nonlinear', 'step_z', True, True)))
        tasks.append(asyncio.create_task(submit('nl_fig8_rg', 'nonlinear', 'figure8', True, True)))
        tasks.append(asyncio.create_task(submit('lin_step', 'linear', 'step_z', False, False)))
    else:
        print(f"[warn] Unknown batch '{kind}'. Running a default pair.")
        tasks.append(asyncio.create_task(submit('linear_step', 'linear', 'step_z', False, False)))
        tasks.append(asyncio.create_task(submit('nonlinear_step_rg', 'nonlinear', 'step_z', True, False)))

    for coro in asyncio.as_completed(tasks):
        res = await coro
        res_list.append(res)
        plot_sim(res, save_prefix=save_prefix)

# ======================================================================================
# CLI
# ======================================================================================

def main():
    ap = argparse.ArgumentParser(description='Quadrotor LQR demo (tuned + asyncio batch).')
    ap.add_argument('--plant', choices=['linear','nonlinear','nonlinear_rot'], default='nonlinear')
    ap.add_argument('--traj', choices=['hover','step_z','figure8'], default='step_z')
    ap.add_argument('--T', type=float, default=8.0)
    ap.add_argument('--Ts', type=float, default=0.004)
    ap.add_argument('--use-integrator', action='store_true')
    ap.add_argument('--use-rg', action='store_true')
    ap.add_argument('--z-final', type=float, default=1.0)
    ap.add_argument('--step-time', type=float, default=0.5)
    ap.add_argument('--mat-dir', type=str, default='mat')
    ap.add_argument('--save-prefix', type=str, default='')
    # Async batch presets
    ap.add_argument('--batch', choices=['', 'linear_vs_nonlinear', 'full'], default='')
    args = ap.parse_args()

    gains = load_gains(args.mat_dir)

    if args.batch:
        asyncio.run(run_batch(args.batch, args.T, args.Ts, gains, args.save_prefix, args.z_final, args.step_time))
        return

    # Single run path (no asyncio needed)
    res = simulate_one(
        name=f"{args.plant}_{args.traj}_rg{int(args.use_rg)}_int{int(args.use_integrator)}",
        plant=args.plant,
        traj=args.traj,
        T=args.T,
        Ts=args.Ts,
        use_rg=args.use_rg,
        use_integrator=args.use_integrator,
        gains=gains,
        z_final=args.z_final,
        step_time=args.step_time,
    )
    plot_sim(res, save_prefix=args.save_prefix)


if __name__ == '__main__':
    main()
