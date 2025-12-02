#!/usr/bin/env python3
"""
SCENARIO 1: GPS SPOOFING ATTACK (FALSE DATA INJECTION)
Uses multithreaded architecture from main_multithreaded.py
Attack: GPS position sensors injected with false data
Defense: Anomaly Detection + Software Rejuvenation
"""

import threading
import queue
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from collections import deque

# Import multithreaded baseline components
from main_multithreaded import (
    Sensor, SensorData, Simulator, Controller, GroundStation, Actuator,
    SimulationLogger, Gains, load_gains, traj_figure8,
    f_nonlinear, rk4_step
)


# =====================================================
# GPS SPOOFING ATTACK - Extends Sensor Thread
# =====================================================

class GPS_SpoofingSensor(Sensor):
    """Sensor with GPS spoofing attack capability"""

    def __init__(self, attack_start: float, attack_duration: float,
                 magnitude: float, pattern: str = 'circular', **kwargs):
        super().__init__(**kwargs)
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.magnitude = magnitude
        self.pattern = pattern
        self.sim_time = 0.0

    def compute_gps_injection(self, t: float) -> np.ndarray:
        """Generate GPS spoofing injection"""
        attack_end = self.attack_start + self.attack_duration
        injection = np.zeros(12)

        if not (self.attack_start <= t <= attack_end):
            return injection

        t_attack = t - self.attack_start
        ramp = min(t_attack / 3.0, 1.0)  # Gradual ramp
        amplitude = self.magnitude * ramp

        if self.pattern == 'circular':
            injection[1] = amplitude * math.sin(2 * math.pi * t / 4)  # x
            injection[3] = amplitude * math.cos(2 * math.pi * t / 4)  # y
            injection[5] = amplitude * 0.3 * math.sin(2 * math.pi * t / 6)  # z
        elif self.pattern == 'drift':
            injection[1] = amplitude * t_attack / 5.0
            injection[3] = amplitude * 0.5 * t_attack / 5.0
            injection[5] = -amplitude * 0.2
        elif self.pattern == 'jump':
            if int(t_attack) % 3 == 0:
                injection[1] = amplitude * (1 if int(t_attack) % 6 == 0 else -1)
                injection[3] = amplitude * (1 if int(t_attack) % 6 == 3 else -1)

        # Only inject on positions (not velocities)
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        return injection * mask

    def run(self):
        """Override run to add GPS spoofing"""
        print(f"[{self.name}] Started with GPS Spoofing Attack")
        print(f"[ATTACK] GPS spoofing: t={self.attack_start}-{self.attack_start + self.attack_duration}s, magnitude={self.magnitude}m, pattern={self.pattern}")

        while not self.stop_event.is_set():
            true_state = self.state_getter()

            # Add sensor noise
            noise = self.rng.normal(0.0, self.noise_std, size=12)

            # Add GPS spoofing injection
            gps_injection = self.compute_gps_injection(self.sim_time)

            # Combined measurement
            measured_state = true_state + noise + gps_injection

            sensor_data = SensorData(
                timestamp=time.time(),
                state=measured_state,
                true_state=true_state
            )

            # Send to all output queues
            for q in self.output_queues:
                try:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except:
                            break
                    q.put(sensor_data, timeout=0.01)
                except queue.Full:
                    pass

            self.sim_time += self.dt
            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# =====================================================
# ANOMALY DETECTOR (Defense Component)
# =====================================================

class GPS_Anomaly_Detector:
    """Chi-square based anomaly detector for GPS spoofing"""

    def __init__(self, threshold=15.0, window_size=15):
        self.threshold = threshold
        self.window_size = window_size
        self.residual_history = deque(maxlen=window_size)
        self.chi_square_history = deque(maxlen=window_size)

        self.attack_detected = False
        self.detection_count = 0
        self.detection_times = []

    def predict_position(self, x_prev, u, dt):
        """Simple prediction using control input"""
        # Predict next position using dynamics
        x_pred = x_prev.copy()
        # Simple integration (can be improved with full dynamics)
        x_pred[1] += x_prev[0] * dt  # x += xdot * dt
        x_pred[3] += x_prev[2] * dt  # y += ydot * dt
        x_pred[5] += x_prev[4] * dt  # z += zdot * dt
        return x_pred

    def compute_residual(self, x_measured, x_predicted):
        """Compute position residual"""
        pos_residual = np.linalg.norm(x_measured[[1,3,5]] - x_predicted[[1,3,5]])
        return pos_residual

    def update(self, t, x_measured, x_predicted):
        """Update detector and check for anomaly"""
        residual = self.compute_residual(x_measured, x_predicted)
        self.residual_history.append(residual)

        # Compute chi-square statistic
        if len(self.residual_history) >= self.window_size:
            chi_square = np.sum(np.array(self.residual_history) ** 2)
            self.chi_square_history.append(chi_square)

            # Check threshold
            if chi_square > self.threshold:
                if not self.attack_detected:
                    self.attack_detected = True
                    self.detection_count += 1
                    self.detection_times.append(t)
                    print(f"\n[DETECTION] Anomaly detected at t={t:.2f}s (chi^2={chi_square:.2f} > {self.threshold})")
                return True
            else:
                self.attack_detected = False

        return False

    def get_current_chi_square(self):
        if len(self.chi_square_history) > 0:
            return self.chi_square_history[-1]
        return 0.0


# =====================================================
# SOFTWARE REJUVENATION (Defense Component)
# =====================================================

class SoftwareRejuvenation:
    """Software Rejuvenation system to restore clean state"""

    def __init__(self, recovery_time=0.8, buffer_size=25):
        self.recovery_time = recovery_time
        self.buffer_size = buffer_size
        self.clean_state_buffer = deque(maxlen=buffer_size)

        self.sr_active = False
        self.sr_start_time = 0
        self.sr_count = 0
        self.sr_times = []

    def update_clean_buffer(self, state):
        """Store known-good states"""
        self.clean_state_buffer.append(state.copy())

    def trigger_rejuvenation(self, t, current_state, attack_detected):
        """Trigger SR if attack detected"""
        if attack_detected and not self.sr_active:
            # Start software rejuvenation
            self.sr_active = True
            self.sr_start_time = t
            self.sr_count += 1
            self.sr_times.append(t)
            print(f"[SR] Software Rejuvenation activated at t={t:.2f}s (restoring to clean state)")

            # Restore to last known clean state
            if len(self.clean_state_buffer) > 0:
                restored_state = self.clean_state_buffer[-1].copy()
                return restored_state
            else:
                return current_state

        elif self.sr_active:
            # Check if recovery period is over
            if t - self.sr_start_time >= self.recovery_time:
                self.sr_active = False
                print(f"[SR] Software Rejuvenation complete at t={t:.2f}s")
                return current_state
            else:
                # Keep using clean state during recovery
                if len(self.clean_state_buffer) > 0:
                    return self.clean_state_buffer[-1].copy()

        return current_state


# =====================================================
# Modified Simulator with SR Integration
# =====================================================

class SR_Simulator(Simulator):
    """Simulator with Software Rejuvenation capability"""

    def __init__(self, detector, sr_system, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.sr_system = sr_system
        self.x_predicted = np.zeros(12)

        # Logging for attack scenario
        self.attack_detected_log = []
        self.sr_active_log = []
        self.chi_square_log = []
        self.residual_log = []

    def run(self):
        print(f"[{self.name}] Started with Anomaly Detection + SR Defense")

        while self.t <= self.T and not self.stop_event.is_set():
            # Get latest control
            try:
                cmd = self.control_queue.get(timeout=0.01)
                self.u = cmd.u
            except queue.Empty:
                pass

            # Get current state before integration
            with self.state_lock:
                x_current = self.x.copy()

            # Anomaly detection (skip during SR recovery)
            attack_detected = False
            if not self.sr_system.sr_active and self.t >= 2.0:  # Start detecting after brief warmup
                # Predict next state
                self.x_predicted = self.detector.predict_position(self.x_predicted, self.u, self.Ts)
                attack_detected = self.detector.update(self.t, x_current, self.x_predicted)

            # Software rejuvenation (just tracks state, doesn't modify dynamics)
            with self.state_lock:
                self.sr_system.trigger_rejuvenation(self.t, x_current, attack_detected)

                # Always integrate normally (SR doesn't teleport the drone)
                self.x = rk4_step(self.x, self.u, self.f, self.Ts)

                # Update clean buffer when not under attack
                if self.t < 3.0:  # Before attack starts at 3.0s
                    self.sr_system.update_clean_buffer(self.x)

            # Log for plotting
            self.attack_detected_log.append(1 if attack_detected else 0)
            self.sr_active_log.append(1 if self.sr_system.sr_active else 0)
            self.chi_square_log.append(self.detector.get_current_chi_square())
            self.residual_log.append(self.detector.residual_history[-1] if len(self.detector.residual_history) > 0 else 0.0)

            self.t += self.Ts
            time.sleep(self.Ts)

        print(f"[{self.name}] Finished at t={self.t:.2f}s")


# =====================================================
# Main Simulation
# =====================================================

def run_gps_spoofing_scenario():
    """Run multithreaded simulation with GPS spoofing attack"""

    print("\n" + "="*80)
    print("SCENARIO 1: GPS SPOOFING ATTACK - MULTITHREADED WITH SR DEFENSE")
    print("="*80)
    print("Attack: GPS False Data Injection on position sensors")
    print("Defense: Anomaly Detection + Software Rejuvenation")
    print("="*80 + "\n")

    # Simulation parameters
    T = 12.0  # Extended to show full figure-8 with attack and recovery
    Ts = 0.01

    # Load gains and create trajectory
    gains = load_gains('mat')
    traj_func = traj_figure8(amp=0.6, period=8.0, z0=0.5, t_start=1.0)

    # Create defense mechanisms
    detector = GPS_Anomaly_Detector(threshold=20.0, window_size=15)
    sr_system = SoftwareRejuvenation(recovery_time=0.8, buffer_size=25)

    # Communication queues
    traj_to_controller = queue.Queue(maxsize=1)
    sensor_to_controller = queue.Queue(maxsize=1)
    sensor_to_ground = queue.Queue(maxsize=1)
    controller_to_actuator = queue.Queue(maxsize=10)
    actuator_to_sim = queue.Queue(maxsize=1)

    # Shared resources
    logger = SimulationLogger()
    stop_event = threading.Event()

    # Create simulator with SR capability
    simulator = SR_Simulator(
        detector=detector,
        sr_system=sr_system,
        T=T, Ts=Ts,
        control_queue=actuator_to_sim,
        logger=logger,
        stop_event=stop_event,
        plant='nonlinear'
    )

    # Create GPS spoofing sensor (ATTACK INJECTED HERE)
    sensor = GPS_SpoofingSensor(
        attack_start=3.0,  # Start attack early in the figure-8
        attack_duration=2.0,  # Short attack for quick recovery demo
        magnitude=1.0,  # Moderate attack to show clear recovery
        pattern='circular',
        rate_hz=100,
        noise_std=0.05,
        state_getter=simulator.get_state,
        output_queues=[sensor_to_controller, sensor_to_ground],
        stop_event=stop_event,
        seed=42
    )

    # Create ground station (NO RG for this scenario)
    ground_station = GroundStation(
        traj_func=traj_func,
        T=T,
        rate_hz=50,
        output_queue=traj_to_controller,
        sensor_queue=sensor_to_ground,
        gains=gains,
        use_rg=False,  # No RG in this scenario
        stop_event=stop_event
    )

    # Create controller
    controller = Controller(
        gains=gains,
        use_integrator=False,
        sensor_queue=sensor_to_controller,
        traj_queue=traj_to_controller,
        output_queue=controller_to_actuator,
        stop_event=stop_event,
        rate_hz=250
    )

    # Create actuator
    actuator = Actuator(
        input_queue=controller_to_actuator,
        output_queue=actuator_to_sim,
        stop_event=stop_event,
        rate_hz=500
    )

    # Start all threads
    threads = [simulator, sensor, ground_station, controller, actuator]
    for thread in threads:
        thread.start()

    print(f"\nAll modules running...\n")

    # Logging loop with real-time position printing
    def logging_loop():
        last_print_time = -1.0  # Print at t=0
        while not stop_event.is_set():
            try:
                current_state = simulator.get_state()
                logger.log(
                    t=simulator.t,
                    state=current_state,
                    ref=ground_station.current_raw_ref.copy(),
                    u=simulator.u.copy(),
                    sensor=controller.current_sensor.copy()
                )

                # Print coordinates every second
                if simulator.t - last_print_time >= 1.0:
                    print(f"[t={simulator.t:.1f}s] Position: X={current_state[1]:6.3f}m, Y={current_state[3]:6.3f}m, Z={current_state[5]:6.3f}m")
                    last_print_time = simulator.t

            except Exception as e:
                pass
            time.sleep(Ts)

    log_thread = threading.Thread(target=logging_loop, name="Logger", daemon=True)
    log_thread.start()

    # Wait for completion
    simulator.join()
    stop_event.set()
    for thread in threads[1:]:
        thread.join(timeout=2.0)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Detections: {detector.detection_count}")
    print(f"SR Activations: {sr_system.sr_count}")
    print("="*80 + "\n")

    # Get logged data
    data = logger.get_arrays()
    data['attack_detected'] = np.array(simulator.attack_detected_log)
    data['sr_active'] = np.array(simulator.sr_active_log)
    data['chi_square'] = np.array(simulator.chi_square_log)
    data['residual'] = np.array(simulator.residual_log)
    data['threshold'] = detector.threshold

    return data


# =====================================================
# Plotting
# =====================================================

def plot_gps_spoofing_results(data):
    """Plot results showing GPS spoofing attack with defenses"""
    from mpl_toolkits.mplot3d import Axes3D

    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']

    # Synchronize array lengths (fix for multithreaded logging)
    min_len = min(len(t), len(data['chi_square']), len(data['residual']), len(data['sr_active']))
    t = t[:min_len]
    X = X[:, :min_len]
    R = R[:, :min_len]
    U = U[:, :min_len]
    data['chi_square'] = data['chi_square'][:min_len]
    data['residual'] = data['residual'][:min_len]
    data['sr_active'] = data['sr_active'][:min_len]
    data['attack_detected'] = data['attack_detected'][:min_len]

    # Attack and defense periods
    attack_start = 3.0
    attack_end = 5.0  # Shortened from 7.0 to show quicker recovery
    attack_mask = (t >= attack_start) & (t <= attack_end)

    # Figure 1: 3D Trajectory - Continuous path with color-coded segments
    fig1 = plt.figure(figsize=(14, 10))
    ax = fig1.add_subplot(111, projection='3d')

    # Plot continuous trajectory with color segments showing attack/recovery
    # Pre-attack (blue)
    pre_attack = t < attack_start
    if np.any(pre_attack):
        ax.plot(X[1, pre_attack], X[3, pre_attack], X[5, pre_attack],
                'b-', linewidth=2.5, alpha=0.9, label='Normal Operation')

    # During attack (orange)
    if np.any(attack_mask):
        ax.plot(X[1, attack_mask], X[3, attack_mask], X[5, attack_mask],
                color='orange', linewidth=2.5, alpha=0.9, label='Under Attack')

    # Post-attack recovery (green)
    post_attack = t > attack_end
    if np.any(post_attack):
        ax.plot(X[1, post_attack], X[3, post_attack], X[5, post_attack],
                'green', linewidth=2.5, alpha=0.9, label='Recovery')

    # Reference trajectory
    ax.plot(R[1, :], R[3, :], R[5, :], 'r--',
            linewidth=1.5, alpha=0.4, label='Reference')

    # Mark start and end
    ax.scatter([X[1, 0]], [X[3, 0]], [X[5, 0]],
               c='darkgreen', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([X[1, -1]], [X[3, -1]], [X[5, -1]],
               c='darkred', s=200, marker='X', label='End', zorder=5, edgecolors='black', linewidths=2)

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_zlabel('Z [m]', fontsize=12)
    ax.set_title('3D Trajectory - GPS Spoofing Attack with SR Defense\n(Blue=Normal, Orange=Attack, Green=Recovery)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([0, 1.0])
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    # Figure 2: Position tracking over time
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9))
    positions = ['X', 'Y', 'Z']
    indices = [1, 3, 5]

    for i, (pos, idx) in enumerate(zip(positions, indices)):
        axes[i].plot(t, X[idx, :], 'b-', linewidth=1.5, label='Actual')
        axes[i].plot(t, R[idx, :], 'r--', linewidth=1.5, alpha=0.7, label='Reference')
        axes[i].axvspan(attack_start, attack_end, alpha=0.2, color='orange', label='Attack Period')
        axes[i].set_ylabel(f'{pos} [m]', fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title('Position Tracking - GPS Spoofing Attack with SR Defense', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()

    # Figure 3: Anomaly Detection and SR Activation
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 9))

    # Chi-square statistic vs threshold
    chi_square = data['chi_square']
    threshold = data['threshold']
    sr_active = data['sr_active']

    axes3[0].plot(t, chi_square, 'b-', linewidth=1.5, label='Chi-square statistic')
    axes3[0].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    axes3[0].fill_between(t, 0, np.max(chi_square)*1.2, where=(sr_active > 0),
                          alpha=0.2, color='green', label='SR Active')
    axes3[0].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period')
    axes3[0].set_ylabel('Chi-square', fontsize=11)
    axes3[0].set_title('Anomaly Detection - Chi-square Threshold', fontsize=13, fontweight='bold')
    axes3[0].legend(fontsize=9, loc='upper left')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].set_ylim([0, max(np.max(chi_square)*1.1, threshold*1.5)])

    # Residual (position error)
    axes3[1].plot(t, data['residual'], 'purple', linewidth=1.5, label='Position Residual')
    axes3[1].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period')
    axes3[1].fill_between(t, 0, np.max(data['residual'])*1.2, where=(sr_active > 0),
                          alpha=0.2, color='green', label='SR Active')
    axes3[1].set_ylabel('Residual [m]', fontsize=11)
    axes3[1].set_title('Position Residual (Measured vs Predicted)', fontsize=13, fontweight='bold')
    axes3[1].legend(fontsize=9)
    axes3[1].grid(True, alpha=0.3)

    # SR activation timeline
    axes3[2].fill_between(t, 0, sr_active, alpha=0.5, color='green', label='Software Rejuvenation Active')
    axes3[2].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period')
    axes3[2].set_ylabel('SR Status', fontsize=11)
    axes3[2].set_xlabel('Time [s]', fontsize=11)
    axes3[2].set_title('Software Rejuvenation Timeline', fontsize=13, fontweight='bold')
    axes3[2].set_ylim([-0.1, 1.2])
    axes3[2].set_yticks([0, 1])
    axes3[2].set_yticklabels(['Inactive', 'Active'])
    axes3[2].legend(fontsize=9)
    axes3[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    # Calculate performance metrics
    xy_error = np.sqrt((X[1, :] - R[1, :]) ** 2 + (X[3, :] - R[3, :]) ** 2)
    print(f"\nPerformance Metrics:")
    print(f"  XY RMS Error: {np.sqrt(np.mean(xy_error**2)):.4f} m")
    print(f"  XY Max Error: {np.max(xy_error):.4f} m")
    print(f"  Z RMS Error: {np.sqrt(np.mean((X[5,:] - R[5,:])**2)):.4f} m\n")


if __name__ == '__main__':
    data = run_gps_spoofing_scenario()
    plot_gps_spoofing_results(data)
