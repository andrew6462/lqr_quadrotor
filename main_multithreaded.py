#!/usr/bin/env python3


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
# CONSTANTS (same as original)
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
    """Data from sensor to controller"""
    timestamp: float
    state: np.ndarray  # 12-dim state with noise
    true_state: np.ndarray  # Ground truth (for logging)

@dataclass
class ControlCommand:
    """Control input from controller to simulator"""
    timestamp: float
    u: np.ndarray  # 4-dim control input [Fz, tau_x, tau_y, tau_z]

@dataclass
class TrajectoryPoint:
    """Reference from ground station to controller"""
    timestamp: float
    r: np.ndarray  # 12-dim reference state

@dataclass
class Gains:
    K: np.ndarray
    Kc: Optional[np.ndarray]

# ======================================================================================
# Thread-Safe Logger
# ======================================================================================
class SimulationLogger:
    """Thread-safe data logger for all simulation data"""
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
# Models (same as original)
# ======================================================================================
def build_linear_AB() -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((12,12))
    A[1,0] = 1.0; A[3,2] = 1.0; A[5,4] = 1.0
    A[7,6] = 1.0; A[9,8] = 1.0; A[11,10] = 1.0
    A[0,9] = -G; A[2,7] = +G
    B = np.zeros((12,4))
    B[4,0] = -1.0 / M
    B[6,1] = 1.0 / JX
    B[8,2] = 1.0 / JY
    B[10,3] = 1.0 / JZ
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
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def load_gains(mat_dir: str = 'mat') -> Gains:
    K = None; Kc = None
    for p in (os.path.join(mat_dir,'K.mat'), os.path.join(mat_dir,'control.mat')):
        if os.path.exists(p):
            try:
                mat = loadmat(p)
                if K is None and 'K' in mat: K = np.array(mat['K'], dtype=float)
                if Kc is None and 'Kc' in mat: Kc = np.array(mat['Kc'], dtype=float)
            except: pass
    if K is None:
        K = np.zeros((4,12))
        K[0,4] = 3.2; K[0,5] = 4.5
        K[1,6] = 1.1; K[1,7] = 1.6; K[1,2] = 0.6; K[1,3] = 0.8
        K[2,8] = 1.1; K[2,9] = 1.6; K[2,0] = 0.6; K[2,1] = 0.8
        K[3,10]= 0.8; K[3,11]= 1.2
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
        kappa = 0.5*(lo+hi)
        v = v_prev + kappa*d
        u = -K @ (x - v)
        if np.all(u <= u_max + 1e-9) and np.all(u >= u_min - 1e-9):
            v_best = v; lo = kappa
        else:
            hi = kappa
    return v_best

# ======================================================================================
# THREAD 1: Ground Station (Trajectory Generator)
# ======================================================================================
class GroundStation(threading.Thread):
    """
    Ground Station generates trajectory references at a fixed rate.
    Sends TrajectoryPoint objects to the controller.
    """
    def __init__(self, traj_func: Callable[[float], np.ndarray], 
                 T: float, rate_hz: float, 
                 output_queue: queue.Queue, 
                 stop_event: threading.Event):
        super().__init__(name="GroundStation")
        self.traj_func = traj_func
        self.T = T
        self.dt = 1.0 / rate_hz
        self.output_queue = output_queue
        self.stop_event = stop_event
        
    def run(self):
        print(f"[{self.name}] Started - generating trajectory at {1/self.dt:.1f} Hz")
        t = 0.0
        while t <= self.T and not self.stop_event.is_set():
            r = self.traj_func(t)
            traj_point = TrajectoryPoint(timestamp=t, r=r)
            try:
                self.output_queue.put(traj_point, timeout=0.1)
            except queue.Full:
                pass  # Skip if queue full
            time.sleep(self.dt)
            t += self.dt
        print(f"[{self.name}] Finished")

# ======================================================================================
# THREAD 2: Sensor (Measurement with Noise)
# ======================================================================================
class Sensor(threading.Thread):
    """
    Sensor reads true state from simulator and adds noise.
    Sends SensorData to controller at a fixed rate.
    """
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
# THREAD 3: Controller (Control Law Computation)
# ======================================================================================
class Controller(threading.Thread):
    """
    Controller receives sensor data and trajectory references.
    Computes control input using LQR + reference governor.
    Sends ControlCommand to simulator.
    """
    def __init__(self, gains: Gains, use_rg: bool, use_integrator: bool,
                 sensor_queue: queue.Queue,
                 traj_queue: queue.Queue,
                 output_queue: queue.Queue,
                 stop_event: threading.Event,
                 rate_hz: float = 250):
        super().__init__(name="Controller")
        self.gains = gains
        self.use_rg = use_rg
        self.use_integrator = use_integrator
        self.sensor_queue = sensor_queue
        self.traj_queue = traj_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.dt = 1.0 / rate_hz

        # State
        self.v = np.zeros(12)  # Governed reference
        self.xi = np.zeros(12)  # Integrator state
        self.current_ref = np.zeros(12)
        self.current_sensor = np.zeros(12)

        # Limits
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

        # RG tracking
        self.rg_intervention_count = 0
        self.last_rg_intervention_time = -1.0
        self.current_time = 0.0
        
    def run(self):
        print(f"[{self.name}] Started - control rate {1/self.dt:.1f} Hz, RG={self.use_rg}, INT={self.use_integrator}")

        while not self.stop_event.is_set():
            # Get latest sensor data (non-blocking)
            try:
                sensor_data = self.sensor_queue.get(timeout=0.01)
                self.current_sensor = sensor_data.state
            except queue.Empty:
                pass

            # Get latest trajectory reference (non-blocking)
            try:
                traj_point = self.traj_queue.get(timeout=0.01)
                self.current_ref = traj_point.r
                self.current_time = traj_point.timestamp
            except queue.Empty:
                pass

            # Compute control with RG tracking
            v_prev = self.v.copy()
            if self.use_rg:
                self.v = govern_reference(
                    self.current_sensor, self.v, self.current_ref,
                    self.gains.K, self.u_min, self.u_max
                )

                # Check if RG intervened (reference was constrained)
                ref_distance = np.linalg.norm(self.current_ref - self.v)
                if ref_distance > 0.01:  # RG constrained the reference
                    if self.current_time - self.last_rg_intervention_time > 0.5:  # Avoid spam
                        self.rg_intervention_count += 1
                        self.last_rg_intervention_time = self.current_time
            else:
                self.v = self.current_ref

            err = self.current_sensor - self.v
            u = -self.gains.K @ err

            # Integrator (position errors only)
            if self.use_integrator and self.gains.Kc is not None:
                pos_idx = [1, 3, 5]
                epos = self.v[pos_idx] - self.current_sensor[pos_idx]
                self.xi[pos_idx] += epos * self.dt
                u = u - (self.gains.Kc @ self.xi[pos_idx])

            # Saturate
            u_clip = np.minimum(np.maximum(u, self.u_min), self.u_max)

            # Send to simulator
            cmd = ControlCommand(timestamp=time.time(), u=u_clip)
            try:
                self.output_queue.put(cmd, timeout=0.1)
            except queue.Full:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")
        if self.use_rg:
            print(f"[{self.name}] Reference Governor interventions: {self.rg_intervention_count}")

# ======================================================================================
# THREAD 4: Simulator (Plant Dynamics Integration)
# ======================================================================================
class Simulator(threading.Thread):
    """
    Simulator integrates nonlinear dynamics.
    Receives control commands from controller.
    Maintains true state accessible by sensor.
    """
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
        
        # Dynamics
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
        print(f"[{self.name}] Started - integration timestep {self.Ts:.4f} s")
        
        while self.t <= self.T and not self.stop_event.is_set():
            # Get latest control (non-blocking)
            try:
                cmd = self.control_queue.get(timeout=0.01)
                self.u = cmd.u
            except queue.Empty:
                pass  # Use previous control
            
            # Integrate
            with self.state_lock:
                self.x = rk4_step(self.x, self.u, self.f, self.Ts)
            
            self.t += self.Ts
            time.sleep(self.Ts)  # Real-time simulation
        
        print(f"[{self.name}] Finished at t={self.t:.2f}s")

# ======================================================================================
# Main Coordinator
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
    Main coordinator that sets up and runs all threads.
    """
    print("\n" + "="*70)
    print("MULTITHREADED FLIGHT CONTROLLER SIMULATION")
    print("="*70)
    
    # Communication queues
    traj_to_controller = queue.Queue(maxsize=10)
    sensor_to_controller = queue.Queue(maxsize=10)
    controller_to_sim = queue.Queue(maxsize=10)
    
    # Shared resources
    logger = SimulationLogger()
    stop_event = threading.Event()
    
    # Create simulator first (so sensor can access state)
    simulator = Simulator(
        T=T, Ts=Ts,
        control_queue=controller_to_sim,
        logger=logger,
        stop_event=stop_event,
        plant=plant
    )
    
    # Create threads
    ground_station = GroundStation(
        traj_func=traj_func,
        T=T,
        rate_hz=50,  # 50 Hz trajectory updates
        output_queue=traj_to_controller,
        stop_event=stop_event
    )
    
    sensor = Sensor(
        rate_hz=100,  # 100 Hz sensor sampling
        noise_std=sensor_noise,
        state_getter=simulator.get_state,
        output_queue=sensor_to_controller,
        stop_event=stop_event,
        seed=seed
    )
    
    controller = Controller(
        gains=gains,
        use_rg=use_rg,
        use_integrator=use_integrator,
        sensor_queue=sensor_to_controller,
        traj_queue=traj_to_controller,
        output_queue=controller_to_sim,
        stop_event=stop_event,
        rate_hz=250  # 250 Hz control loop
    )
    
    # Logging thread
    def logging_loop():
        last_print_time = 0
        print_interval = 1.0  # Print every 1 second

        while not stop_event.is_set():
            # Get current reference from ground station queue (peek)
            try:
                traj_point = traj_to_controller.get(timeout=0.01)
                current_ref = traj_point.r
                traj_to_controller.put(traj_point)  # Put it back
            except:
                current_ref = np.zeros(12)

            # Get sensor measurement
            try:
                sensor_data = sensor_to_controller.get(timeout=0.01)
                sensor_meas = sensor_data.state
                sensor_to_controller.put(sensor_data)
            except:
                sensor_meas = np.zeros(12)

            # Get control
            try:
                cmd = controller_to_sim.get(timeout=0.01)
                current_u = cmd.u
                controller_to_sim.put(cmd)
            except:
                current_u = np.zeros(4)

            # Log
            current_state = simulator.get_state()
            logger.log(
                t=simulator.t,
                state=current_state,
                ref=current_ref,
                u=current_u,
                sensor=sensor_meas
            )

            # Print trajectory coordinates at intervals
            if simulator.t - last_print_time >= print_interval:
                # Calculate tracking error
                pos_error = np.sqrt((current_state[1] - current_ref[1])**2 +
                                   (current_state[3] - current_ref[3])**2 +
                                   (current_state[5] - current_ref[5])**2)

                # Check if RG is active
                rg_status = ""
                if use_rg:
                    ref_distance = np.linalg.norm(controller.current_ref - controller.v)
                    if ref_distance > 0.01:
                        rg_status = " | [RG ACTIVE - Constraining reference]"
                    else:
                        rg_status = " | [RG: OK]"

                print(f"t={simulator.t:5.2f}s | Pos: x={current_state[1]:6.3f}, y={current_state[3]:6.3f}, z={current_state[5]:6.3f}m | Error: {pos_error:.4f}m{rg_status}")
                last_print_time = simulator.t

            time.sleep(Ts)
    
    log_thread = threading.Thread(target=logging_loop, name="Logger")
    
    # Start all threads
    threads = [simulator, ground_station, sensor, controller, log_thread]
    for thread in threads:
        thread.start()
    
    # Wait for simulator to finish
    simulator.join()
    
    # Stop all other threads
    stop_event.set()
    for thread in threads[1:]:
        thread.join(timeout=2.0)

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)

    # Performance summary
    data = logger.get_arrays()
    final_state = data['X'][:, -1]
    final_ref = data['R'][:, -1]
    all_states = data['X'].T
    all_refs = data['R'].T

    # Calculate tracking performance
    pos_errors = np.sqrt((all_states[:, 1] - all_refs[:, 1])**2 +
                         (all_states[:, 3] - all_refs[:, 3])**2 +
                         (all_states[:, 5] - all_refs[:, 5])**2)
    mean_error = np.mean(pos_errors)
    max_error = np.max(pos_errors)

    print(f"\nFinal Position:    x={final_state[1]:.3f}m, y={final_state[3]:.3f}m, z={final_state[5]:.3f}m")
    print(f"Target Reference:  x={final_ref[1]:.3f}m, y={final_ref[3]:.3f}m, z={final_ref[5]:.3f}m")
    print(f"\nTracking Performance:")
    print(f"  Mean Error:  {mean_error:.4f}m")
    print(f"  Max Error:   {max_error:.4f}m")

    if use_rg:
        print(f"\nReference Governor:")
        print(f"  Total Interventions: {controller.rg_intervention_count}")
        print(f"  Status: {'ACTIVE - Safety constraints enforced' if controller.rg_intervention_count > 0 else 'STANDBY - No constraints needed'}")

    print("="*70 + "\n")

    return data

# ======================================================================================
# Trajectories
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
    w = 2.0 * math.pi / period
    def r_of_t(t: float) -> np.ndarray:
        r = np.zeros(12)
        r[1] = amp * math.sin(w * t)
        r[3] = amp * math.sin(2*w * t)
        r[5] = z0
        return r
    return r_of_t

# ======================================================================================
# Plotting
# ======================================================================================
def plot_results(data: Dict[str, Any], save_prefix: str = 'mt'):
    """Plot simulation results"""
    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']
    S = data['S']

    # Figure 1: 2D Figure-8 Trajectory (XY Top View)
    fig1 = plt.figure(figsize=(12, 10))
    plt.plot(X[1,:], X[3,:], 'b-', label='Actual Trajectory', linewidth=3, alpha=0.8)
    plt.plot(R[1,:], R[3,:], 'r--', label='Reference (Figure-8)', linewidth=2, alpha=0.6)
    plt.scatter([X[1,0]], [X[3,0]], c='green', s=200, marker='o',
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    plt.scatter([X[1,-1]], [X[3,-1]], c='red', s=200, marker='X',
                label='End', zorder=5, edgecolors='black', linewidths=2)
    plt.xlabel('X [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Y [m]', fontsize=14, fontweight='bold')
    plt.title('Figure-8 Trajectory - Top View (XY Plane)\nMultithreaded with Reference Governor',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_figure8_2d.png', dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_prefix}_figure8_2d.png")

    # Figure 2: 3D Figure-8 Trajectory
    fig2 = plt.figure(figsize=(14, 10))
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(X[1,:], X[3,:], X[5,:], 'b-', label='Actual Trajectory', linewidth=3, alpha=0.8)
    ax.plot(R[1,:], R[3,:], R[5,:], 'r--', label='Reference (Figure-8)', linewidth=2, alpha=0.6)
    ax.scatter([X[1,0]], [X[3,0]], [X[5,0]], c='green', s=150, marker='o',
               label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([X[1,-1]], [X[3,-1]], [X[5,-1]], c='red', s=150, marker='X',
               label='End', zorder=5, edgecolors='black', linewidths=2)
    ax.set_xlabel('X [m]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=13, fontweight='bold')
    ax.set_zlabel('Z [m]', fontsize=13, fontweight='bold')
    ax.set_title('Figure-8 Trajectory - 3D View\nMultithreaded with Reference Governor',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_figure8_3d.png', dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_prefix}_figure8_3d.png")

    # Figure 3: Position Tracking Over Time
    fig3, axes = plt.subplots(3, 1, figsize=(14, 9))
    positions = ['X', 'Y', 'Z']
    indices = [1, 3, 5]

    for i, (pos, idx) in enumerate(zip(positions, indices)):
        axes[i].plot(t, X[idx,:], 'b-', linewidth=2, label='Actual')
        axes[i].plot(t, R[idx,:], 'r--', linewidth=2, alpha=0.7, label='Reference')
        axes[i].set_ylabel(f'{pos} [m]', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=10, loc='upper right')
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title('Position Tracking Over Time - Multithreaded with Reference Governor',
                      fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_positions.png', dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_prefix}_positions.png")

    # Figure 4: Control Inputs
    fig4 = plt.figure(figsize=(14, 8))
    labels = ['Thrust (Fz)', 'Roll Torque (τx)', 'Pitch Torque (τy)', 'Yaw Torque (τz)']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(t, U[i,:], 'b-', linewidth=1.5)
        plt.axhline(y=FZ_MAX if i==0 else TAU_X_MAX, color='r', linestyle='--', alpha=0.5, label='Limits')
        plt.axhline(y=-FZ_MAX if i==0 else -TAU_X_MAX, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel(labels[i], fontsize=11)
        plt.title(labels[i], fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend(fontsize=9)
    plt.suptitle('Control Inputs - Multithreaded with Reference Governor',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_controls.png', dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_prefix}_controls.png")

    print("\n[OK] All plots generated!\n")
    plt.show()

# ======================================================================================
# CLI
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description='Multithreaded Quadrotor Flight Controller (HW3)')
    ap.add_argument('--traj', choices=['hover','step_z','figure8'], default='figure8')
    ap.add_argument('--T', type=float, default=6.0)
    ap.add_argument('--Ts', type=float, default=0.008)
    ap.add_argument('--use-rg', action='store_true')
    ap.add_argument('--use-integrator', action='store_true')
    ap.add_argument('--sensor-noise', type=float, default=0.05)
    ap.add_argument('--plant', choices=['linear','nonlinear'], default='nonlinear')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mat-dir', type=str, default='mat')
    ap.add_argument('--save-prefix', type=str, default='hw3_mt')
    args = ap.parse_args()
    
    # Load gains
    gains = load_gains(args.mat_dir)
    
    # Select trajectory
    if args.traj == 'hover':
        traj_func = traj_hover()
    elif args.traj == 'step_z':
        traj_func = traj_step_z()
    else:
        traj_func = traj_figure8()
    
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
    
    # Plot results
    plot_results(data, save_prefix=args.save_prefix)

if __name__ == '__main__':
    main()
