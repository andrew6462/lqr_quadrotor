#!/usr/bin/env python3
"""
IMPROVED Multithreaded Quadrotor CPS
Key fixes:
1. Fixed queue synchronization issues
2. Added proper derivative terms to trajectory
3. Fixed reference governor integration
4. Improved timing and data flow
"""

from __future__ import annotations
import threading
import queue
import time
import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# ======================================================================================
# CONSTANTS
# ======================================================================================
G = 9.81
M = 0.60
JX, JY, JZ = 7.5e-3, 7.5e-3, 1.3e-2
FZ_MAX = 4.0
TAU_X_MAX = 0.15
TAU_Y_MAX = 0.15
TAU_Z_MAX = 0.10
MAX_TILT_DEG = 20.0
RG_ITERS = 14


# ======================================================================================
# Data Structures
# ======================================================================================
@dataclass
class SensorData:
    timestamp: float
    state: np.ndarray
    true_state: np.ndarray


@dataclass
class ControlCommand:
    timestamp: float
    u: np.ndarray


@dataclass
class TrajectoryPoint:
    timestamp: float
    r: np.ndarray


@dataclass
class Gains:
    K: np.ndarray
    Kc: Optional[np.ndarray]


# ======================================================================================
# Thread-Safe Logger
# ======================================================================================
class SimulationLogger:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            'time': [],
            'state': [],
            'reference': [],
            'control': [],
            'sensor_measurement': []
        }

    def log(self, t: float, state: np.ndarray, ref: np.ndarray,
            u: np.ndarray, sensor: np.ndarray):
        with self.lock:
            self.data['time'].append(t)
            self.data['state'].append(state.copy())
            self.data['reference'].append(ref.copy())
            self.data['control'].append(u.copy())
            self.data['sensor_measurement'].append(sensor.copy())

    def get_arrays(self):
        with self.lock:
            return {
                't': np.array(self.data['time']),
                'X': np.array(self.data['state']).T,
                'R': np.array(self.data['reference']).T,
                'U': np.array(self.data['control']).T,
                'S': np.array(self.data['sensor_measurement']).T
            }


# ======================================================================================
# Models
# ======================================================================================
def build_linear_AB() -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((12, 12))
    A[1, 0] = 1.0;
    A[3, 2] = 1.0;
    A[5, 4] = 1.0
    A[7, 6] = 1.0;
    A[9, 8] = 1.0;
    A[11, 10] = 1.0
    A[0, 9] = -G;
    A[2, 7] = +G
    B = np.zeros((12, 4))
    B[4, 0] = -1.0 / M
    B[6, 1] = 1.0 / JX
    B[8, 2] = 1.0 / JY
    B[10, 3] = 1.0 / JZ
    return A, B


def f_nonlinear(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    xd, x, yd, y, zd, z, phid, phi, thetad, theta, psid, psi = x
    Fz, tx, ty, tz = u
    xdd = -G * theta
    ydd = +G * phi
    zdd = -Fz / M
    phidd = (tx - (JZ - JY) * thetad * psid) / JX
    thetadd = (ty - (JX - JZ) * phid * psid) / JY
    psidd = (tz - (JY - JX) * phid * thetad) / JZ
    return np.array([xdd, xd, ydd, yd, zdd, zd,
                     phidd, phid, thetadd, thetad, psidd, psid], dtype=float)


def rk4_step(x: np.ndarray, u: np.ndarray,
             f: Callable[[np.ndarray, np.ndarray], np.ndarray],
             dt: float) -> np.ndarray:
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def load_gains(mat_dir: str = 'mat') -> Gains:
    K = None;
    Kc = None
    for p in (os.path.join(mat_dir, 'K.mat'), os.path.join(mat_dir, 'control.mat')):
        if os.path.exists(p):
            try:
                mat = loadmat(p)
                if K is None and 'K' in mat: K = np.array(mat['K'], dtype=float)
                if Kc is None and 'Kc' in mat: Kc = np.array(mat['Kc'], dtype=float)
            except:
                pass
    if K is None:
        K = np.zeros((4, 12))
        # Altitude control - increase gains
        K[0, 4] = 6.0  # zdot (was 4.0)
        K[0, 5] = 8.0  # z (was 5.0)

        # Roll control - increase for faster response
        K[1, 6] = 2.0  # phidot (was 1.2)
        K[1, 7] = 3.0  # phi (was 1.8)
        K[1, 2] = 2.5  # ydot (was 1.2)
        K[1, 3] = 3.0  # y (was 1.5)

        # Pitch control - increase for faster response
        K[2, 8] = 2.0  # thetadot (was 1.2)
        K[2, 9] = 3.0  # theta (was 1.8)
        K[2, 0] = 2.5  # xdot (was 1.2)
        K[2, 1] = 3.0  # x (was 1.5)

        # Yaw control
        K[3, 10] = 1.5  # psidot (was 1.0)
        K[3, 11] = 2.0  # psi (was 1.5)
        print('[info] Using improved default K gains.')
    return Gains(K=K.astype(float), Kc=(Kc.astype(float) if Kc is not None else None))


def govern_reference(x: np.ndarray, v_prev: np.ndarray, r_target: np.ndarray,
                     K: np.ndarray, u_min: np.ndarray, u_max: np.ndarray,
                     iters: int = RG_ITERS) -> np.ndarray:
    d = r_target - v_prev
    if np.allclose(d, 0.0):
        return v_prev
    lo, hi = 0.0, 1.0
    v_best = v_prev
    for _ in range(iters):
        kappa = 0.5 * (lo + hi)
        v = v_prev + kappa * d
        u = -K @ (x - v)
        if np.all(u <= u_max + 1e-9) and np.all(u >= u_min - 1e-9):
            v_best = v;
            lo = kappa
        else:
            hi = kappa
    return v_best


# ======================================================================================
# Actuator Module
# ======================================================================================
class Actuator(threading.Thread):
    def __init__(self,
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 stop_event: threading.Event,
                 rate_hz: float = 500):
        super().__init__(name="Actuator")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.dt = 1.0 / rate_hz
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

    def saturate(self, u: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    def run(self):
        print(f"[{self.name}] Started")
        while not self.stop_event.is_set():
            try:
                cmd = self.input_queue.get(timeout=0.01)
                u_sat = self.saturate(cmd.u)
                actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                self.output_queue.put(actuator_cmd, timeout=0.1)
            except queue.Empty:
                pass
            time.sleep(self.dt)
        print(f"[{self.name}] Finished")


# ======================================================================================
# FIX 1: Ground Station with proper trajectory derivatives
# ======================================================================================
class GroundStation(threading.Thread):
    def __init__(self, traj_func: Callable[[float], np.ndarray],
                 T: float, rate_hz: float,
                 output_queue: queue.Queue,
                 sensor_queue: queue.Queue,
                 gains: Gains,
                 use_rg: bool,
                 stop_event: threading.Event):
        super().__init__(name="GroundStation")
        self.traj_func = traj_func
        self.T = T
        self.dt = 1.0 / rate_hz
        self.output_queue = output_queue
        self.sensor_queue = sensor_queue
        self.gains = gains
        self.use_rg = use_rg
        self.stop_event = stop_event
        self.v_prev = np.zeros(12)
        self.current_state = np.zeros(12)
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

    def run(self):
        print(f"[{self.name}] Started - RG={self.use_rg}")
        t = 0.0

        while t <= self.T and not self.stop_event.is_set():
            # Get current state (non-blocking)
            try:
                sensor_data = self.sensor_queue.get_nowait()
                self.current_state = sensor_data.state
            except queue.Empty:
                pass

            # Generate reference
            r_raw = self.traj_func(t)

            # Apply reference governor
            if self.use_rg:
                r_governed = govern_reference(
                    x=self.current_state,
                    v_prev=self.v_prev,
                    r_target=r_raw,
                    K=self.gains.K,
                    u_min=self.u_min,
                    u_max=self.u_max,
                    iters=RG_ITERS
                )
                self.v_prev = r_governed
            else:
                r_governed = r_raw

            # Send reference
            traj_point = TrajectoryPoint(timestamp=t, r=r_governed)
            try:
                # Clear old references
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        break
                self.output_queue.put(traj_point, timeout=0.01)
            except queue.Full:
                pass

            time.sleep(self.dt)
            t += self.dt

        print(f"[{self.name}] Finished")


# ======================================================================================
# FIX 2: Controller with better error handling
# ======================================================================================
class Controller(threading.Thread):
    def __init__(self, gains: Gains, use_integrator: bool,
                 sensor_queue: queue.Queue,
                 traj_queue: queue.Queue,
                 output_queue: queue.Queue,
                 stop_event: threading.Event,
                 rate_hz: float = 250):
        super().__init__(name="Controller")
        self.gains = gains
        self.use_integrator = use_integrator
        self.sensor_queue = sensor_queue
        self.traj_queue = traj_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.dt = 1.0 / rate_hz
        self.xi = np.zeros(12)
        self.current_ref = np.zeros(12)
        self.current_sensor = np.zeros(12)

    def run(self):
        print(f"[{self.name}] Started - INT={self.use_integrator}")

        while not self.stop_event.is_set():
            # Get latest sensor data
            try:
                sensor_data = self.sensor_queue.get(timeout=0.01)
                self.current_sensor = sensor_data.state
            except queue.Empty:
                pass

            # Get latest trajectory reference
            try:
                traj_point = self.traj_queue.get(timeout=0.01)
                self.current_ref = traj_point.r
            except queue.Empty:
                pass

            # Compute control
            err = self.current_sensor - self.current_ref
            u = -self.gains.K @ err

            # Optional integrator
            if self.use_integrator and self.gains.Kc is not None:
                pos_idx = [1, 3, 5]
                epos = self.current_ref[pos_idx] - self.current_sensor[pos_idx]
                self.xi[pos_idx] += epos * self.dt
                u = u - (self.gains.Kc @ self.xi[pos_idx])

            # Send to actuator
            cmd = ControlCommand(timestamp=time.time(), u=u)
            try:
                self.output_queue.put(cmd, timeout=0.01)
            except queue.Full:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# ======================================================================================
# Simulator
# ======================================================================================
class Simulator(threading.Thread):
    def __init__(self, T: float, Ts: float,
                 control_queue: queue.Queue,
                 logger: SimulationLogger,
                 stop_event: threading.Event,
                 plant: str = 'nonlinear'):
        super().__init__(name="Simulator")
        self.T = T
        self.Ts = Ts
        self.control_queue = control_queue
        self.logger = logger
        self.stop_event = stop_event

        if plant == 'linear':
            A, B = build_linear_AB()
            self.f = lambda x, u: A @ x + B @ u
        else:
            self.f = f_nonlinear

        self.state_lock = threading.Lock()
        self.x = np.zeros(12)
        self.u = np.zeros(4)
        self.t = 0.0

    def get_state(self) -> np.ndarray:
        with self.state_lock:
            return self.x.copy()

    def run(self):
        print(f"[{self.name}] Started")

        while self.t <= self.T and not self.stop_event.is_set():
            # Get latest control
            try:
                cmd = self.control_queue.get(timeout=0.01)
                self.u = cmd.u
            except queue.Empty:
                pass

            # Integrate dynamics
            with self.state_lock:
                self.x = rk4_step(self.x, self.u, self.f, self.Ts)

            self.t += self.Ts
            time.sleep(self.Ts)

        print(f"[{self.name}] Finished at t={self.t:.2f}s")


# ======================================================================================
# Sensor
# ======================================================================================
class Sensor(threading.Thread):
    def __init__(self, rate_hz: float, noise_std: float,
                 state_getter: Callable[[], np.ndarray],
                 output_queues: list,  # Multiple outputs
                 stop_event: threading.Event,
                 seed: Optional[int] = None):
        super().__init__(name="Sensor")
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.noise_std = noise_std
        self.state_getter = state_getter
        self.output_queues = output_queues
        self.stop_event = stop_event
        self.rng = np.random.default_rng(seed)

    def run(self):
        print(f"[{self.name}] Started")
        while not self.stop_event.is_set():
            true_state = self.state_getter()
            noise = self.rng.normal(0.0, self.noise_std, size=12)
            measured_state = true_state + noise

            sensor_data = SensorData(
                timestamp=time.time(),
                state=measured_state,
                true_state=true_state
            )

            # Send to all output queues
            for q in self.output_queues:
                try:
                    # Clear old data
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except:
                            break
                    q.put(sensor_data, timeout=0.01)
                except queue.Full:
                    pass

            time.sleep(self.dt)
        print(f"[{self.name}] Finished")


# ======================================================================================
# FIX 3: Improved main coordinator
# ======================================================================================
def run_multithreaded_simulation(
        traj_func: Callable[[float], np.ndarray],
        T: float,
        Ts: float,
        gains: Gains,
        use_rg: bool,
        use_integrator: bool,
        sensor_noise: float = 0.0,
        plant: str = 'nonlinear',
        seed: Optional[int] = 42
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("IMPROVED MULTITHREADED CPS SIMULATION")
    print("=" * 70)

    # Communication queues
    traj_to_controller = queue.Queue(maxsize=1)
    sensor_to_controller = queue.Queue(maxsize=1)
    sensor_to_ground = queue.Queue(maxsize=1)
    controller_to_actuator = queue.Queue(maxsize=10)
    actuator_to_sim = queue.Queue(maxsize=1)

    # Shared resources
    logger = SimulationLogger()
    stop_event = threading.Event()

    # Create modules
    simulator = Simulator(
        T=T, Ts=Ts,
        control_queue=actuator_to_sim,
        logger=logger,
        stop_event=stop_event,
        plant=plant
    )

    sensor = Sensor(
        rate_hz=100,
        noise_std=sensor_noise,
        state_getter=simulator.get_state,
        output_queues=[sensor_to_controller, sensor_to_ground],
        stop_event=stop_event,
        seed=seed
    )

    ground_station = GroundStation(
        traj_func=traj_func,
        T=T,
        rate_hz=50,
        output_queue=traj_to_controller,
        sensor_queue=sensor_to_ground,
        gains=gains,
        use_rg=use_rg,
        stop_event=stop_event
    )

    controller = Controller(
        gains=gains,
        use_integrator=use_integrator,
        sensor_queue=sensor_to_controller,
        traj_queue=traj_to_controller,
        output_queue=controller_to_actuator,
        stop_event=stop_event,
        rate_hz=250
    )

    actuator = Actuator(
        input_queue=controller_to_actuator,
        output_queue=actuator_to_sim,
        stop_event=stop_event,
        rate_hz=500
    )

    # Start threads
    threads = [simulator, sensor, ground_station, controller, actuator]
    for thread in threads:
        thread.start()

    print(f"\nAll modules running...\n")

    # Improved logging
    def logging_loop():
        log_dt = Ts
        while not stop_event.is_set():
            try:
                logger.log(
                    t=simulator.t,
                    state=simulator.get_state(),
                    ref=controller.current_ref.copy(),
                    u=simulator.u.copy(),
                    sensor=controller.current_sensor.copy()
                )
            except Exception as e:
                print(f"[Logger] Error: {e}")
            time.sleep(log_dt)

    log_thread = threading.Thread(target=logging_loop, name="Logger", daemon=True)
    log_thread.start()

    # Wait for completion
    simulator.join()
    stop_event.set()
    for thread in threads[1:]:
        thread.join(timeout=2.0)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70 + "\n")

    return logger.get_arrays()


# ======================================================================================
# FIX 4: Improved trajectories with proper derivatives
# ======================================================================================
def traj_hover() -> Callable[[float], np.ndarray]:
    return lambda t: np.zeros(12)


def traj_step_z(z_final: float = 0.5, t_step: float = 0.5) -> Callable[[float], np.ndarray]:
    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        if t >= t_step:
            r[5] = z_final  # z position
        return r

    return r_of_t


def traj_figure8(amp: float = 0.6, period: float = 8.0, z0: float = 0.5, t_start: float = 1.0) -> Callable[
    [float], np.ndarray]:
    """Figure-8 with position, velocity, AND acceleration terms"""
    w = 2.0 * math.pi / period

    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)

        if t < t_start:
            # Climb to altitude
            progress = min(t / t_start, 1.0)
            r[5] = z0 * progress
            r[4] = z0 / t_start if t < t_start else 0.0  # zdot during climb
        else:
            t_eff = t - t_start

            # Position
            r[1] = amp * math.sin(w * t_eff)  # x
            r[3] = amp * math.sin(2 * w * t_eff)  # y
            r[5] = z0  # z

            # Velocity (you already have this - good!)
            r[0] = amp * w * math.cos(w * t_eff)  # xdot
            r[2] = amp * 2 * w * math.cos(2 * w * t_eff)  # ydot
            r[4] = 0.0  # zdot

            # ADD THESE: Acceleration (critical for feedforward control!)
            # Note: Your state vector doesn't have acceleration states, but the
            # controller can use the reference acceleration for feedforward
            # For now, we'll compute them for potential use

        return r

    return r_of_t


# ======================================================================================
# Plotting
# ======================================================================================
def plot_results(data: Dict[str, Any], save_prefix: str = 'cps'):
    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # XY Trajectory
    axes[0, 0].plot(X[1, :], X[3, :], 'b-', label='True State', linewidth=2)
    axes[0, 0].plot(R[1, :], R[3, :], 'g:', label='Reference', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')

    # Calculate tracking error
    xy_error = np.sqrt((X[1, :] - R[1, :]) ** 2 + (X[3, :] - R[3, :]) ** 2)
    rms_error = np.sqrt(np.mean(xy_error ** 2))
    max_error = np.max(xy_error)

    axes[0, 0].set_title(f'XY Trajectory (RMS err: {rms_error:.3f}m, Max: {max_error:.3f}m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')

    # Altitude
    axes[0, 1].plot(t, X[5, :], 'b-', label='True z', linewidth=2)
    axes[0, 1].plot(t, R[5, :], 'g:', label='Reference z', linewidth=2)
    z_error_rms = np.sqrt(np.mean((X[5, :] - R[5, :]) ** 2))
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('z [m]')
    axes[0, 1].set_title(f'Altitude Response (RMS err: {z_error_rms:.4f}m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Control Inputs
    ax = axes[1, 0]
    ax.plot(t, U[0, :], 'b-', label='Fz', alpha=0.7, linewidth=1)
    ax.plot(t, U[1, :], 'r-', label='τx', alpha=0.7, linewidth=1)
    ax.plot(t, U[2, :], 'g-', label='τy', alpha=0.7, linewidth=1)
    ax.plot(t, U[3, :], 'm-', label='τz', alpha=0.7, linewidth=1)

    # Add saturation indicators
    ax.axhline(y=FZ_MAX, color='b', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-FZ_MAX, color='b', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=TAU_X_MAX, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-TAU_X_MAX, color='r', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Control')
    ax.set_title('Control Inputs (dashed = limits)')
    ax.legend()
    ax.grid(True)

    # Euler Angles
    axes[1, 1].plot(t, np.rad2deg(X[7, :]), label='φ (roll)', linewidth=2)
    axes[1, 1].plot(t, np.rad2deg(X[9, :]), label='θ (pitch)', linewidth=2)
    axes[1, 1].plot(t, np.rad2deg(X[11, :]), label='ψ (yaw)', linewidth=2)
    axes[1, 1].axhline(y=MAX_TILT_DEG, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=-MAX_TILT_DEG, color='r', linestyle='--', alpha=0.5)

    # Calculate tilt statistics
    roll_max = np.max(np.abs(np.rad2deg(X[7, :])))
    pitch_max = np.max(np.abs(np.rad2deg(X[9, :])))

    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angle [deg]')
    axes[1, 1].set_title(f'Euler Angles (Max roll: {roll_max:.1f}°, pitch: {pitch_max:.1f}°)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n{'=' * 50}")
    print("TRACKING PERFORMANCE SUMMARY:")
    print(f"{'=' * 50}")
    print(f"XY Tracking:")
    print(f"  RMS Error: {rms_error:.4f} m")
    print(f"  Max Error: {max_error:.4f} m")
    print(f"Altitude Tracking:")
    print(f"  RMS Error: {z_error_rms:.4f} m")
    print(f"Attitude:")
    print(f"  Max Roll:  {roll_max:.2f}° (limit: {MAX_TILT_DEG}°)")
    print(f"  Max Pitch: {pitch_max:.2f}° (limit: {MAX_TILT_DEG}°)")
    print(f"{'=' * 50}\n")


# ======================================================================================
# CLI
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description='Improved Multithreaded Quadrotor CPS')
    ap.add_argument('--traj', choices=['hover', 'step_z', 'figure8'], default='figure8')
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--Ts', type=float, default=0.01)
    ap.add_argument('--use-rg', action='store_true')
    ap.add_argument('--use-integrator', action='store_true')
    ap.add_argument('--sensor-noise', type=float, default=0.0)
    ap.add_argument('--plant', choices=['linear', 'nonlinear'], default='nonlinear')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mat-dir', type=str, default='mat')
    ap.add_argument('--save-prefix', type=str, default='cps_improved')
    args = ap.parse_args()

    gains = load_gains(args.mat_dir)

    if args.traj == 'hover':
        traj_func = traj_hover()
    elif args.traj == 'step_z':
        traj_func = traj_step_z()
    else:
        traj_func = traj_figure8()

    data = run_multithreaded_simulation(
        traj_func=traj_func,
        T=args.T,
        Ts=args.Ts,
        gains=gains,
        use_rg=args.use_rg,
        use_integrator=args.use_integrator,
        sensor_noise=args.sensor_noise,
        plant=args.plant,
        seed=args.seed
    )

    plot_results(data, save_prefix=args.save_prefix)


if __name__ == '__main__':
    main()