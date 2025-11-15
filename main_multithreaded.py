#!/usr/bin/env python3
"""
FIXED Multithreaded Quadrotor CPS - Addressing Professor Feedback
Changes made:
1. Added Actuator module (NEW - lines 162-195)
2. Moved Reference Governor to Ground Station (lines 197-269)
3. Simplified Controller (lines 271-326)
4. Updated Simulator to receive from Actuator (lines 328-376)
5. Fixed data flow: Ground Station → Controller → Actuator → Simulator → Sensor
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
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import os

# ======================================================================================
# CONSTANTS (unchanged)
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
# Data Structures (unchanged)
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
# Thread-Safe Logger (unchanged)
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
# Models (unchanged)
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
        K[0, 4] = 3.2;
        K[0, 5] = 4.5
        K[1, 6] = 1.1;
        K[1, 7] = 1.6;
        K[1, 2] = 0.6;
        K[1, 3] = 0.8
        K[2, 8] = 1.1;
        K[2, 9] = 1.6;
        K[2, 0] = 0.6;
        K[2, 1] = 0.8
        K[3, 10] = 0.8;
        K[3, 11] = 1.2
        print('[info] Using demo K (fallback).')
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
# CHANGE 1: NEW ACTUATOR MODULE (Lines 162-195)
# WHY: Professor said "actuators are missing" - controller was sending directly to sim
# WHAT IT DOES: Receives control from controller, applies limits, sends to simulator
# ======================================================================================
class Actuator(threading.Thread):
    """
    NEW MODULE - Actuator applies saturation limits to control inputs.
    Sits between Controller and Simulator.
    """

    def __init__(self,
                 input_queue: queue.Queue,  # From controller
                 output_queue: queue.Queue,  # To simulator
                 stop_event: threading.Event,
                 rate_hz: float = 500):
        super().__init__(name="Actuator")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.dt = 1.0 / rate_hz

        # Actuator limits
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

    def saturate(self, u: np.ndarray) -> np.ndarray:
        """Apply physical actuator limits"""
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    def run(self):
        print(f"[{self.name}] Started - actuator rate {1 / self.dt:.1f} Hz")

        while not self.stop_event.is_set():
            try:
                # Receive command from controller
                cmd = self.input_queue.get(timeout=0.01)

                # Apply saturation limits (this is actuator's job!)
                u_sat = self.saturate(cmd.u)

                # Send saturated command to simulator
                actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                self.output_queue.put(actuator_cmd, timeout=0.1)

            except queue.Empty:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# ======================================================================================
# CHANGE 2: MOVED REFERENCE GOVERNOR TO GROUND STATION (Lines 197-269)
# WHY: Professor said "RG should be in ground station, not controller"
# WHAT CHANGED:
#   - Ground station now receives sensor data (line 205)
#   - Ground station has gains and RG flag (lines 206-207)
#   - RG logic moved here from controller (lines 239-252)
# ======================================================================================
class GroundStation(threading.Thread):
    """
    MODIFIED - Ground Station now includes Reference Governor.
    Generates trajectory AND applies RG before sending to controller.
    """

    def __init__(self, traj_func: Callable[[float], np.ndarray],
                 T: float, rate_hz: float,
                 output_queue: queue.Queue,
                 sensor_queue: queue.Queue,  # NEW: needs state for RG
                 gains: Gains,  # NEW: needs K for RG
                 use_rg: bool,  # NEW: RG flag moved here
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

        # Reference governor state
        self.v_prev = np.zeros(12)
        self.current_state = np.zeros(12)

        # Limits for RG
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

    def run(self):
        print(f"[{self.name}] Started - trajectory generation + RG at {1 / self.dt:.1f} Hz, RG={self.use_rg}")
        t = 0.0

        while t <= self.T and not self.stop_event.is_set():
            # Get current state from sensor (needed for reference governor)
            try:
                sensor_data = self.sensor_queue.get(timeout=0.001)
                self.current_state = sensor_data.state
                # Put back for controller
                self.sensor_queue.put(sensor_data)
            except queue.Empty:
                pass

            # Step 1: Generate raw trajectory reference
            r_raw = self.traj_func(t)

            # Step 2: Apply reference governor (MOVED HERE FROM CONTROLLER!)
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

            # Step 3: Send governed reference to controller
            traj_point = TrajectoryPoint(timestamp=t, r=r_governed)
            try:
                self.output_queue.put(traj_point, timeout=0.1)
            except queue.Full:
                pass

            time.sleep(self.dt)
            t += self.dt

        print(f"[{self.name}] Finished")


# ======================================================================================
# CHANGE 3: SIMPLIFIED CONTROLLER (Lines 271-326)
# WHY: Professor said controller should only do control law, not RG or saturation
# WHAT CHANGED:
#   - Removed use_rg parameter (line 275)
#   - Removed self.v (RG state) - line 289
#   - Removed RG logic (old lines 280-285)
#   - Removed saturation (old line 301) - actuator does this now
#   - Controller now just does: u = -K @ error (+ integrator)
# ======================================================================================
class Controller(threading.Thread):
    """
    SIMPLIFIED - Controller now ONLY computes control law.
    No reference governor (moved to Ground Station).
    No saturation (moved to Actuator).
    """

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

        # State (REMOVED: self.v and RG-related variables)
        self.xi = np.zeros(12)  # Integrator state only
        self.current_ref = np.zeros(12)
        self.current_sensor = np.zeros(12)

    def run(self):
        print(f"[{self.name}] Started - control rate {1 / self.dt:.1f} Hz, INT={self.use_integrator}")

        while not self.stop_event.is_set():
            # Get latest sensor data
            try:
                sensor_data = self.sensor_queue.get(timeout=0.01)
                self.current_sensor = sensor_data.state
            except queue.Empty:
                pass

            # Get latest trajectory reference (ALREADY GOVERNED by Ground Station!)
            try:
                traj_point = self.traj_queue.get(timeout=0.01)
                self.current_ref = traj_point.r
            except queue.Empty:
                pass

            # Compute control - SIMPLE NOW!
            # Just the LQR control law: u = -K @ error
            err = self.current_sensor - self.current_ref
            u = -self.gains.K @ err

            # Optional integrator for steady-state error
            if self.use_integrator and self.gains.Kc is not None:
                pos_idx = [1, 3, 5]
                epos = self.current_ref[pos_idx] - self.current_sensor[pos_idx]
                self.xi[pos_idx] += epos * self.dt
                u = u - (self.gains.Kc @ self.xi[pos_idx])

            # Send to actuator (NO SATURATION - actuator handles it!)
            cmd = ControlCommand(timestamp=time.time(), u=u)
            try:
                self.output_queue.put(cmd, timeout=0.1)
            except queue.Full:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# ======================================================================================
# CHANGE 4: SIMULATOR NOW RECEIVES FROM ACTUATOR (Lines 328-376)
# WHY: Data flow should be Controller → Actuator → Simulator
# WHAT CHANGED:
#   - control_queue now comes from Actuator (not Controller) - line 332
#   - Comments updated to reflect this (line 370)
# ======================================================================================
class Simulator(threading.Thread):
    """
    MODIFIED - Simulator now receives from Actuator (not Controller directly).
    Just integrates dynamics - no control logic.
    """

    def __init__(self, T: float, Ts: float,
                 control_queue: queue.Queue,  # FROM ACTUATOR NOW!
                 logger: SimulationLogger,
                 stop_event: threading.Event,
                 plant: str = 'nonlinear'):
        super().__init__(name="Simulator")
        self.T = T
        self.Ts = Ts
        self.control_queue = control_queue
        self.logger = logger
        self.stop_event = stop_event

        # Select dynamics
        if plant == 'linear':
            A, B = build_linear_AB()
            self.f = lambda x, u: A @ x + B @ u
        else:
            self.f = f_nonlinear

        # State
        self.state_lock = threading.Lock()
        self.x = np.zeros(12)
        self.u = np.zeros(4)
        self.t = 0.0

    def get_state(self) -> np.ndarray:
        """Thread-safe state access for sensor"""
        with self.state_lock:
            return self.x.copy()

    def run(self):
        print(f"[{self.name}] Started - dynamics integration at {self.Ts:.4f} s timestep")

        while self.t <= self.T and not self.stop_event.is_set():
            # Get control from ACTUATOR (already saturated!)
            try:
                cmd = self.control_queue.get(timeout=0.01)
                self.u = cmd.u
            except queue.Empty:
                pass

            # ONLY JOB: Integrate dx = f(x, u)
            with self.state_lock:
                self.x = rk4_step(self.x, self.u, self.f, self.Ts)

            self.t += self.Ts
            time.sleep(self.Ts)

        print(f"[{self.name}] Finished at t={self.t:.2f}s")


# ======================================================================================
# Sensor (unchanged - lines 378-414)
# ======================================================================================
class Sensor(threading.Thread):
    """Sensor reads true state and adds noise"""

    def __init__(self, rate_hz: float, noise_std: float,
                 state_getter: Callable[[], np.ndarray],
                 output_queue: queue.Queue,
                 stop_event: threading.Event,
                 seed: Optional[int] = None):
        super().__init__(name="Sensor")
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.noise_std = noise_std
        self.state_getter = state_getter
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.rng = np.random.default_rng(seed)

    def run(self):
        print(f"[{self.name}] Started - sampling at {self.rate_hz:.1f} Hz, noise σ={self.noise_std}")
        while not self.stop_event.is_set():
            true_state = self.state_getter()
            noise = self.rng.normal(0.0, self.noise_std, size=12)
            measured_state = true_state + noise

            sensor_data = SensorData(
                timestamp=time.time(),
                state=measured_state,
                true_state=true_state
            )
            try:
                self.output_queue.put(sensor_data, timeout=0.1)
            except queue.Full:
                pass
            time.sleep(self.dt)
        print(f"[{self.name}] Finished")


# ======================================================================================
# CHANGE 5: UPDATED MAIN COORDINATOR (Lines 416-521)
# WHY: Need to wire up the new Actuator and update data flow
# WHAT CHANGED:
#   - Added controller_to_actuator queue (line 434)
#   - Added actuator_to_sim queue (line 435)
#   - Added sensor_to_ground queue (line 436)
#   - Created Actuator instance (lines 470-475)
#   - Updated GroundStation to receive sensor data and do RG (lines 453-460)
#   - Updated Controller to not do RG (lines 468)
#   - Updated Simulator to receive from actuator (line 444)
#   - Added actuator to threads list (line 478)
# ======================================================================================
def run_multithreaded_simulation(
        traj_func: Callable[[float], np.ndarray],
        T: float,
        Ts: float,
        gains: Gains,
        use_rg: bool,
        use_integrator: bool,
        sensor_noise: float = 0.05,
        plant: str = 'nonlinear',
        seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    FIXED Main coordinator with proper CPS architecture.
    Flow: Ground Station → Controller → Actuator → Simulator → Sensor
    """
    print("\n" + "=" * 70)
    print("MULTITHREADED CPS SIMULATION - PROFESSOR FEEDBACK ADDRESSED")
    print("=" * 70)

    # Communication queues (UPDATED WITH NEW CONNECTIONS)
    traj_to_controller = queue.Queue(maxsize=10)
    sensor_to_controller = queue.Queue(maxsize=10)
    sensor_to_ground = queue.Queue(maxsize=10)  # NEW: Ground station needs state for RG
    controller_to_actuator = queue.Queue(maxsize=10)  # NEW: Controller → Actuator
    actuator_to_sim = queue.Queue(maxsize=10)  # NEW: Actuator → Simulator

    # Shared resources
    logger = SimulationLogger()
    stop_event = threading.Event()

    # Create simulator (receives from actuator now!)
    simulator = Simulator(
        T=T, Ts=Ts,
        control_queue=actuator_to_sim,  # CHANGED: from actuator, not controller
        logger=logger,
        stop_event=stop_event,
        plant=plant
    )

    # Create ground station (now does RG!)
    ground_station = GroundStation(
        traj_func=traj_func,
        T=T,
        rate_hz=50,
        output_queue=traj_to_controller,
        sensor_queue=sensor_to_ground,  # NEW: needs sensor data
        gains=gains,  # NEW: needs K for RG
        use_rg=use_rg,  # NEW: RG moved here
        stop_event=stop_event
    )

    # Create sensor (feeds both controller and ground station)
    sensor = Sensor(
        rate_hz=100,
        noise_std=sensor_noise,
        state_getter=simulator.get_state,
        output_queue=sensor_to_controller,
        stop_event=stop_event,
        seed=seed
    )

    # Create controller (simplified - no RG!)
    controller = Controller(
        gains=gains,
        use_integrator=use_integrator,
        sensor_queue=sensor_to_controller,
        traj_queue=traj_to_controller,
        output_queue=controller_to_actuator,  # CHANGED: sends to actuator
        stop_event=stop_event,
        rate_hz=250
    )

    # Create actuator (NEW MODULE!)
    actuator = Actuator(
        input_queue=controller_to_actuator,
        output_queue=actuator_to_sim,
        stop_event=stop_event,
        rate_hz=500
    )

    # Start all threads (including actuator!)
    threads = [simulator, ground_station, sensor, controller, actuator]
    for thread in threads:
        thread.start()

    print(f"\n{'=' * 70}")
    print("CPS MODULES RUNNING:")
    print("  [1] Ground Station (traj gen + RG)")
    print("  [2] Sensor (state measurement)")
    print("  [3] Controller (control law)")
    print("  [4] Actuator (saturation)")
    print("  [5] Simulator (dynamics)")
    print(f"{'=' * 70}\n")

    # Logging thread (updated to handle new queues)
    def logging_loop():
        while not stop_event.is_set():
            try:
                current_ref = np.zeros(12)
                sensor_meas = np.zeros(12)
                current_u = np.zeros(4)

                # Get reference without consuming
                if not traj_to_controller.empty():
                    tp_list = []
                    # Drain queue and get latest
                    while not traj_to_controller.empty():
                        try:
                            tp = traj_to_controller.get_nowait()
                            tp_list.append(tp)
                        except:
                            break
                    if tp_list:
                        current_ref = tp_list[-1].r
                        # Put last one back
                        try:
                            traj_to_controller.put_nowait(tp_list[-1])
                        except:
                            pass

                # Get sensor data
                if not sensor_to_controller.empty():
                    sd_list = []
                    while not sensor_to_controller.empty():
                        try:
                            sd = sensor_to_controller.get_nowait()
                            sd_list.append(sd)
                        except:
                            break
                    if sd_list:
                        sensor_meas = sd_list[-1].state
                        # Put last one back and send to ground
                        try:
                            sensor_to_controller.put_nowait(sd_list[-1])
                            sensor_to_ground.put_nowait(sd_list[-1])
                        except:
                            pass

                # Get control
                if not actuator_to_sim.empty():
                    cmd_list = []
                    while not actuator_to_sim.empty():
                        try:
                            cmd = actuator_to_sim.get_nowait()
                            cmd_list.append(cmd)
                        except:
                            break
                    if cmd_list:
                        current_u = cmd_list[-1].u
                        # Put last one back
                        try:
                            actuator_to_sim.put_nowait(cmd_list[-1])
                        except:
                            pass

                # Log
                logger.log(
                    t=simulator.t,
                    state=simulator.get_state(),
                    ref=current_ref,
                    u=current_u,
                    sensor=sensor_meas
                )
            except Exception as e:
                print(f"[Logger] Error: {e}")

            time.sleep(Ts)

    log_thread = threading.Thread(target=logging_loop, name="Logger", daemon=True)
    log_thread.start()

    # Wait for simulator
    simulator.join()

    # Stop all threads
    stop_event.set()
    for thread in threads[1:]:
        thread.join(timeout=2.0)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE - All modules stopped properly")
    print("=" * 70 + "\n")

    return logger.get_arrays()


# ======================================================================================
# Trajectories (unchanged - lines 523-546)
# ======================================================================================
def traj_hover() -> Callable[[float], np.ndarray]:
    return lambda t: np.zeros(12)


def traj_step_z(z_final: float = 1.0, t_step: float = 0.5) -> Callable[[float], np.ndarray]:
    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        if t >= t_step: r[5] = z_final
        return r

    return r_of_t


def traj_figure8(amp: float = 1.0, period: float = 6.0, z0: float = 0.5) -> Callable[[float], np.ndarray]:
    """Figure-8 trajectory - smaller scale for better tracking"""
    w = 2.0 * math.pi / period

    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        r[1] = amp * math.sin(w * t)  # x position
        r[3] = amp * math.sin(2 * w * t)  # y position
        r[5] = z0  # z position constant
        return r

    return r_of_t


# ======================================================================================
# Plotting (unchanged - lines 548-607)
# ======================================================================================
def plot_results(data: Dict[str, Any], save_prefix: str = 'cps'):
    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']
    S = data['S']

    # XY Trajectory
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(X[1, :], X[3, :], 'b-', label='True State', linewidth=2)
    plt.plot(R[1, :], R[3, :], 'g:', label='Reference', linewidth=2)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('XY Trajectory (CPS Multithreaded)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    if save_prefix:
        plt.savefig(f'{save_prefix}_xy.png', dpi=150, bbox_inches='tight')

    # Altitude
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(t, X[5, :], 'b-', label='True z', linewidth=2)
    plt.plot(t, R[5, :], 'g:', label='Reference z', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('z [m]')
    plt.title('Altitude Response')
    plt.legend()
    plt.grid(True)
    if save_prefix:
        plt.savefig(f'{save_prefix}_altitude.png', dpi=150, bbox_inches='tight')

    # Control Inputs
    fig3 = plt.figure(figsize=(10, 6))
    labels = ['Fz', 'τx', 'τy', 'τz']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, U[i, :], 'b-', linewidth=1.5)
        plt.axhline(y=FZ_MAX if i == 0 else TAU_X_MAX, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=-FZ_MAX if i == 0 else -TAU_X_MAX, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time [s]')
        plt.ylabel(labels[i])
        plt.grid(True)
    plt.suptitle('Control Inputs')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_controls.png', dpi=150, bbox_inches='tight')

    # Euler Angles
    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(t, X[7, :], label='φ (roll)', linewidth=2)
    plt.plot(t, X[9, :], label='θ (pitch)', linewidth=2)
    plt.plot(t, X[11, :], label='ψ (yaw)', linewidth=2)
    lim = math.radians(MAX_TILT_DEG)
    plt.axhline(y=lim, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=-lim, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.title(f'Euler Angles (±{MAX_TILT_DEG}° guide)')
    plt.legend()
    plt.grid(True)
    if save_prefix:
        plt.savefig(f'{save_prefix}_angles.png', dpi=150, bbox_inches='tight')

    plt.show()


# ======================================================================================
# CLI (unchanged - lines 609-640)
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description='Multithreaded Quadrotor CPS (Fixed per Professor Feedback)')
    ap.add_argument('--traj', choices=['hover', 'step_z', 'figure8'], default='figure8',
                    help='Trajectory to follow')
    ap.add_argument('--T', type=float, default=10.0,
                    help='Simulation time (seconds)')
    ap.add_argument('--Ts', type=float, default=0.01,
                    help='Integration timestep (seconds)')
    ap.add_argument('--use-rg', action='store_true',
                    help='Enable reference governor (in Ground Station)')
    ap.add_argument('--use-integrator', action='store_true',
                    help='Enable integral control')
    ap.add_argument('--sensor-noise', type=float, default=0.0,
                    help='Sensor noise standard deviation')
    ap.add_argument('--plant', choices=['linear', 'nonlinear'], default='nonlinear',
                    help='Plant dynamics model')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for sensor noise')
    ap.add_argument('--mat-dir', type=str, default='mat',
                    help='Directory containing K.mat files')
    ap.add_argument('--save-prefix', type=str, default='cps_fixed',
                    help='Prefix for saved plots')
    args = ap.parse_args()

    # Debug: Print parameters
    print(f"\n{'=' * 70}")
    print(f"SIMULATION PARAMETERS:")
    print(f"  Trajectory: {args.traj}")
    print(f"  Total time: {args.T} seconds")
    print(f"  Timestep: {args.Ts} seconds")
    print(f"  Use RG: {args.use_rg}")
    print(f"  Use Integrator: {args.use_integrator}")
    print(f"{'=' * 70}\n")

    # Load gains
    gains = load_gains(args.mat_dir)

    # Select trajectory
    if args.traj == 'hover':
        traj_func = traj_hover()
    elif args.traj == 'step_z':
        traj_func = traj_step_z()
    else:
        traj_func = traj_figure8()

    # Test trajectory generation
    print("Testing trajectory at t=2.0:")
    test_r = traj_func(2.0)
    print(f"  x={test_r[1]:.3f}, y={test_r[3]:.3f}, z={test_r[5]:.3f}\n")

    # Run simulation
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

    # Debug: Check results
    print(f"\n{'=' * 70}")
    print("SIMULATION RESULTS:")
    print(f"  Time range: {data['t'][0]:.2f} to {data['t'][-1]:.2f} seconds")
    print(f"  Samples collected: {len(data['t'])}")
    x_range = data['X'][1, :].max() - data['X'][1, :].min()
    y_range = data['X'][3, :].max() - data['X'][3, :].min()
    z_range = data['X'][5, :].max() - data['X'][5, :].min()
    print(f"  Motion range: x={x_range:.3f}m, y={y_range:.3f}m, z={z_range:.3f}m")
    print(f"{'=' * 70}\n")

    # Plot results
    plot_results(data, save_prefix=args.save_prefix)


if __name__ == '__main__':
    main()