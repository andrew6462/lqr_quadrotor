#!/usr/bin/env python3
"""
Cyber Attack and Software Rejuvenation Module
Integrates with main_multithreaded.py architecture

This module adds:
1. CyberAttackThread - Injects FDI attacks on sensor data
2. SoftwareRejuvenationThread - Monitors and triggers rejuvenation
3. Enhanced Sensor class that can be attacked
4. Modified run function to include attack/SR threads
"""

import threading
import queue
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AttackConfig:
    """Configuration for cyber attack"""
    enabled: bool = True
    start_time: float = 10.0
    duration: float = 15.0
    magnitude: float = 2.0
    # Attack mask: which sensors to attack (1=attack, 0=clean)
    # [u, x, v, y, w, z, p, phi, q, theta, r, psi]
    attack_mask: np.ndarray = None
    
    def __post_init__(self):
        if self.attack_mask is None:
            # Default: attack position measurements (x, y, z)
            self.attack_mask = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])


@dataclass
class SRConfig:
    """Configuration for Software Rejuvenation"""
    enabled: bool = True
    threshold: float = 5.0
    recovery_time: float = 0.5
    check_interval: float = 0.1
    buffer_size: int = 10


# ============================================================================
# CYBER ATTACK THREAD
# ============================================================================

class CyberAttackThread(threading.Thread):
    """
    Monitors sensor data and injects false data (FDI attack)
    
    Architecture:
    - Sits between Sensor output and Controller input
    - Reads clean sensor data from sensor_queue
    - Injects attack signal
    - Outputs corrupted data to controller_queue
    """
    
    def __init__(self, 
                 config: AttackConfig,
                 sensor_queue: queue.Queue,
                 controller_queue: queue.Queue,
                 stop_event: threading.Event,
                 start_time_ref: Callable[[], float]):
        super().__init__(name="CyberAttack", daemon=True)
        self.config = config
        self.sensor_queue = sensor_queue
        self.controller_queue = controller_queue
        self.stop_event = stop_event
        self.get_current_time = start_time_ref
        
        self.attack_active = False
        self.attack_log = []
        
    def compute_attack_signal(self, t: float) -> np.ndarray:
        """Generate time-varying attack signal"""
        if not self.config.enabled:
            return np.zeros(12)
        
        attack_end = self.config.start_time + self.config.duration
        self.attack_active = (self.config.start_time <= t <= attack_end)
        
        if not self.attack_active:
            return np.zeros(12)
        
        # Time since attack started
        t_attack = t - self.config.start_time
        
        # Gradually increasing attack
        ramp = min(t_attack / 5.0, 1.0)
        amplitude = self.config.magnitude * ramp
        
        # Sinusoidal pattern (mimics GPS drift/spoofing)
        attack_signal = np.array([
            0,
            amplitude * math.sin(2 * math.pi * t / 3),      # x attack
            0,
            amplitude * math.cos(2 * math.pi * t / 3),      # y attack
            0,
            amplitude * 0.5 * math.sin(2 * math.pi * t / 5), # z attack
            0, 0, 0, 0, 0, 0
        ])
        
        return attack_signal * self.config.attack_mask
    
    def run(self):
        print(f"[{self.name}] Started - Attack: {self.config.start_time}s-{self.config.start_time + self.config.duration}s")
        
        while not self.stop_event.is_set():
            try:
                # Get clean sensor data
                sensor_data = self.sensor_queue.get(timeout=0.01)
                
                # Get current simulation time
                t = self.get_current_time()
                
                # Compute attack signal
                attack_signal = self.compute_attack_signal(t)
                
                # Inject attack
                corrupted_state = sensor_data.state + attack_signal
                
                # Create corrupted sensor data
                from main_multithreaded import SensorData
                corrupted_data = SensorData(
                    timestamp=sensor_data.timestamp,
                    state=corrupted_state,
                    true_state=sensor_data.true_state
                )
                
                # Log attack event
                if self.attack_active:
                    self.attack_log.append({
                        'time': t,
                        'attack_signal': attack_signal.copy(),
                        'corrupted_state': corrupted_state.copy()
                    })
                
                # Forward to controller
                try:
                    # Clear old data
                    while not self.controller_queue.empty():
                        try:
                            self.controller_queue.get_nowait()
                        except:
                            break
                    self.controller_queue.put(corrupted_data, timeout=0.01)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                time.sleep(0.001)
                continue
        
        print(f"[{self.name}] Finished - {len(self.attack_log)} attack samples logged")


# ============================================================================
# SOFTWARE REJUVENATION THREAD
# ============================================================================

class SoftwareRejuvenationThread(threading.Thread):
    """
    Monitors system safety using Lyapunov function
    Triggers rejuvenation when threshold exceeded
    
    Architecture:
    - Monitors sensor data and reference trajectory
    - Computes Lyapunov function V(x)
    - Triggers state restoration when V > threshold
    - Communicates with Simulator to reset state
    """
    
    def __init__(self,
                 config: SRConfig,
                 P_matrix: np.ndarray,
                 sensor_queue: queue.Queue,
                 traj_queue: queue.Queue,
                 simulator_ref: object,  # Reference to Simulator object
                 stop_event: threading.Event):
        super().__init__(name="SoftwareRejuvenation", daemon=True)
        self.config = config
        self.P = P_matrix
        self.sensor_queue = sensor_queue
        self.traj_queue = traj_queue
        self.simulator = simulator_ref
        self.stop_event = stop_event
        
        # SR state
        self.rejuvenation_active = False
        self.rejuvenation_start_time = None
        self.last_check_time = 0
        self.rejuvenation_count = 0
        self.rejuvenation_log = []
        
        # Clean state buffer
        self.clean_buffer = []
        
        # Current data
        self.current_sensor = None
        self.current_ref = None
        
    def compute_lyapunov(self, state: np.ndarray, ref: np.ndarray) -> float:
        """Compute V(x) = (x - x_ref)^T P (x - x_ref)"""
        error = state - ref
        V = error.T @ self.P @ error
        return V
    
    def should_trigger_sr(self, t: float, V: float) -> bool:
        """Determine if SR should be triggered"""
        if not self.config.enabled:
            return False
        
        # Check interval
        if t - self.last_check_time < self.config.check_interval:
            return False
        
        self.last_check_time = t
        
        # Don't trigger if already active
        if self.rejuvenation_active:
            return False
        
        # Check threshold
        if V > self.config.threshold:
            return True
        
        return False
    
    def execute_rejuvenation(self, t: float):
        """Restore system to clean state"""
        if not self.rejuvenation_active:
            # Start rejuvenation
            self.rejuvenation_active = True
            self.rejuvenation_start_time = t
            self.rejuvenation_count += 1
            print(f"[SR TRIGGER #{self.rejuvenation_count}] t={t:.2f}s")
            self.rejuvenation_log.append({'time': t, 'count': self.rejuvenation_count})
        
        # Check if complete
        elapsed = t - self.rejuvenation_start_time
        if elapsed >= self.config.recovery_time:
            # Rejuvenation complete
            self.rejuvenation_active = False
            print(f"[SR COMPLETE #{self.rejuvenation_count}] t={t:.2f}s")
            
            # Restore to clean state (average of buffer)
            if len(self.clean_buffer) > 0:
                clean_state = np.mean(self.clean_buffer, axis=0)
                # Update simulator state
                with self.simulator.state_lock:
                    self.simulator.x = clean_state.copy()
                print(f"[SR] State restored from buffer (avg of {len(self.clean_buffer)} samples)")
        else:
            # During rejuvenation - gradual recovery
            if len(self.clean_buffer) > 0:
                target = np.mean(self.clean_buffer, axis=0)
                current = self.simulator.get_state()
                alpha = elapsed / self.config.recovery_time
                interpolated = (1 - alpha) * current + alpha * target
                with self.simulator.state_lock:
                    self.simulator.x = interpolated.copy()
    
    def update_clean_buffer(self, state: np.ndarray):
        """Maintain buffer of clean states"""
        self.clean_buffer.append(state.copy())
        if len(self.clean_buffer) > self.config.buffer_size:
            self.clean_buffer.pop(0)
    
    def run(self):
        print(f"[{self.name}] Started - Threshold: {self.config.threshold}")
        
        while not self.stop_event.is_set():
            try:
                # Get sensor data (non-blocking)
                try:
                    sensor_data = self.sensor_queue.get(timeout=0.01)
                    self.current_sensor = sensor_data.state
                except queue.Empty:
                    pass
                
                # Get trajectory reference
                try:
                    traj_point = self.traj_queue.get(timeout=0.01)
                    self.current_ref = traj_point.r
                except queue.Empty:
                    pass
                
                if self.current_sensor is not None and self.current_ref is not None:
                    t = self.simulator.t
                    
                    # Compute Lyapunov function
                    V = self.compute_lyapunov(self.current_sensor, self.current_ref)
                    
                    # Check for SR trigger
                    if self.should_trigger_sr(t, V):
                        self.execute_rejuvenation(t)
                    
                    # If rejuvenation active, continue recovery
                    if self.rejuvenation_active:
                        self.execute_rejuvenation(t)
                    else:
                        # Buffer clean states when no attack
                        # (heuristic: when V is small, assume clean)
                        if V < self.config.threshold * 0.5:
                            self.update_clean_buffer(self.current_sensor)
                
                time.sleep(self.config.check_interval / 2)
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(0.1)
        
        print(f"[{self.name}] Finished - {self.rejuvenation_count} rejuvenations performed")
        print(f"[SR] Trigger times: {[log['time'] for log in self.rejuvenation_log]}")


# ============================================================================
# MODIFIED RUN FUNCTION WITH ATTACK & SR
# ============================================================================

def run_with_attack_and_sr(
        traj_func: Callable[[float], np.ndarray],
        T: float,
        Ts: float,
        gains,  # Gains object from main_multithreaded
        attack_config: AttackConfig,
        sr_config: SRConfig,
        P_matrix: np.ndarray,
        use_rg: bool = False,
        use_integrator: bool = False,
        sensor_noise: float = 0.0,
        plant: str = 'nonlinear',
        seed: Optional[int] = 42
):
    """
    Modified run function that includes cyber attack and SR threads
    
    This wraps the original run_multithreaded_simulation from main_multithreaded.py
    and adds attack/SR layers
    """
    
    # Import from main_multithreaded
    from main_multithreaded import (
        Simulator, Sensor, Controller, Actuator, GroundStation,
        SimulationLogger
    )
    
    print("\n" + "=" * 70)
    print("MULTITHREADED CPS WITH CYBER ATTACK & SOFTWARE REJUVENATION")
    print("=" * 70)
    print(f"Attack: {attack_config.start_time}s-{attack_config.start_time + attack_config.duration}s")
    print(f"SR Enabled: {sr_config.enabled}, Threshold: {sr_config.threshold}")
    print("=" * 70 + "\n")
    
    # Communication queues
    traj_to_controller = queue.Queue(maxsize=1)
    traj_to_sr = queue.Queue(maxsize=1)
    sensor_to_attack = queue.Queue(maxsize=1)  # Clean sensor -> attack
    attack_to_controller = queue.Queue(maxsize=1)  # Attacked data -> controller
    sensor_to_sr = queue.Queue(maxsize=1)  # For SR monitoring
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
        output_queues=[sensor_to_attack, sensor_to_sr, sensor_to_ground],  # Multiple outputs
        stop_event=stop_event,
        seed=seed
    )
    
    # Cyber Attack Thread - sits between sensor and controller
    attack_thread = CyberAttackThread(
        config=attack_config,
        sensor_queue=sensor_to_attack,
        controller_queue=attack_to_controller,
        stop_event=stop_event,
        start_time_ref=lambda: simulator.t
    )
    
    # Software Rejuvenation Thread
    sr_thread = SoftwareRejuvenationThread(
        config=sr_config,
        P_matrix=P_matrix,
        sensor_queue=sensor_to_sr,
        traj_queue=traj_to_sr,
        simulator_ref=simulator,
        stop_event=stop_event
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
    
    # Duplicate trajectory to SR thread
    class TrajDuplicator(threading.Thread):
        def __init__(self):
            super().__init__(name="TrajDup", daemon=True)
        def run(self):
            while not stop_event.is_set():
                try:
                    # This is a hack - ground station should output to both
                    # For now, SR will get trajectory separately
                    time.sleep(0.02)
                except:
                    pass
    
    # Modified ground station to output to both queues
    # (This is a workaround - ideally ground_station would have multiple outputs)
    original_gs_run = ground_station.run
    def modified_gs_run():
        # Wrapper that duplicates output
        while not stop_event.is_set():
            try:
                # Run one iteration (this is a simplification)
                time.sleep(ground_station.dt)
                r = ground_station.traj_func(simulator.t)
                from main_multithreaded import TrajectoryPoint
                traj_point = TrajectoryPoint(timestamp=time.time(), r=r)
                
                # Send to controller
                try:
                    while not traj_to_controller.empty():
                        try:
                            traj_to_controller.get_nowait()
                        except:
                            break
                    traj_to_controller.put(traj_point, timeout=0.01)
                except:
                    pass
                
                # Send to SR
                try:
                    while not traj_to_sr.empty():
                        try:
                            traj_to_sr.get_nowait()
                        except:
                            break
                    traj_to_sr.put(traj_point, timeout=0.01)
                except:
                    pass
                    
            except Exception as e:
                print(f"[TrajDup] Error: {e}")
    
    ground_station.run = modified_gs_run
    
    controller = Controller(
        gains=gains,
        use_integrator=use_integrator,
        sensor_queue=attack_to_controller,  # Gets ATTACKED sensor data
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
    
    # Start all threads
    threads = [simulator, sensor, attack_thread, sr_thread, ground_station, controller, actuator]
    for thread in threads:
        thread.start()
    
    print(f"\nAll modules running (including Attack & SR)...\n")
    
    # Logging loop
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
    print("=" * 70)
    print(f"Attack Events Logged: {len(attack_thread.attack_log)}")
    print(f"SR Triggers: {sr_thread.rejuvenation_count}")
    print(f"SR Times: {[f\"{log['time']:.2f}s\" for log in sr_thread.rejuvenation_log]}")
    print("=" * 70 + "\n")
    
    # Add attack and SR data to results
    data = logger.get_arrays()
    data['meta']['attack_enabled'] = attack_config.enabled
    data['meta']['sr_enabled'] = sr_config.enabled
    data['attack_log'] = attack_thread.attack_log
    data['sr_log'] = sr_thread.rejuvenation_log
    data['sr_count'] = sr_thread.rejuvenation_count
    
    return data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use this module with main_multithreaded.py
    """
    import sys
    sys.path.append('.')  # Ensure main_multithreaded is importable
    
    from main_multithreaded import load_gains, traj_figure8
    
    # Load gains
    gains = load_gains('mat')
    
    # Create P matrix (for Lyapunov - use identity if not available)
    try:
        from scipy.io import loadmat
        P_data = loadmat('mat/P.mat')
        P = P_data['P']
    except:
        print("Warning: P.mat not found, using identity matrix")
        P = np.eye(12)
    
    # Configure attack
    attack_config = AttackConfig(
        enabled=True,
        start_time=10.0,
        duration=15.0,
        magnitude=2.0
    )
    
    # Configure SR
    sr_config = SRConfig(
        enabled=True,
        threshold=5.0,
        recovery_time=0.5,
        check_interval=0.1
    )
    
    # Run simulation
    data = run_with_attack_and_sr(
        traj_func=traj_figure8(),
        T=40.0,
        Ts=0.01,
        gains=gains,
        attack_config=attack_config,
        sr_config=sr_config,
        P_matrix=P,
        use_rg=False,
        use_integrator=False,
        sensor_noise=0.0,
        plant='nonlinear',
        seed=42
    )
    
    # Plot results (use plot_results from main_multithreaded)
    from main_multithreaded import plot_results
    plot_results(data, save_prefix='cps_attack_sr')
