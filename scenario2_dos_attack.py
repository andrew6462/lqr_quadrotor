"""
SCENARIO 2: DENIAL-OF-SERVICE (DoS) ATTACK WITH STATE ESTIMATOR DEFENSE

Attack: Denial of Service - drops sensor packets randomly
Defense: State Estimator + Packet Drop Detection

npm run scenario2
"""

import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time
from typing import Dict, Any
from collections import deque

# Import multithreaded components
from main_multithreaded import (
    Sensor, SensorData, ControlCommand, Simulator, Controller, GroundStation, Actuator,
    SimulationLogger, Gains, load_gains, traj_figure8,
    f_nonlinear, rk4_step,
    FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX
)


# =====================================================
# DoS ATTACK - Packet Dropping Sensor
# =====================================================

class DoS_Sensor(threading.Thread):
    """
    Sensor with DoS attack capability
    Randomly drops packets during attack period
    """
    def __init__(self, attack_start=10.0, attack_duration=15.0, drop_rate=0.6,
                 rate_hz=100, noise_std=0.05, state_getter=None,
                 output_queues=None, stop_event=None, seed=42):
        super().__init__(name="Sensor")
        self.daemon = True

        # Attack parameters
        self.attack_start = attack_start
        self.attack_duration = attack_duration
        self.attack_end = attack_start + attack_duration
        self.drop_rate = drop_rate

        # Sensor parameters
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.noise_std = noise_std
        self.state_getter = state_getter
        self.output_queues = output_queues or []
        self.stop_event = stop_event

        # Random seed for reproducibility
        self.rng = np.random.RandomState(seed)

        # Statistics
        self.packets_sent = 0
        self.packets_dropped = 0
        self.t = 0.0

    def is_attack_active(self, t):
        """Check if attack is currently active"""
        return self.attack_start <= t <= self.attack_end

    def run(self):
        print(f"[{self.name}] Started (DoS Attack: {self.attack_start}s-{self.attack_end}s, Drop Rate: {self.drop_rate*100:.0f}%)")

        while not self.stop_event.is_set():
            # Get true state from simulator
            true_state = self.state_getter()

            # Add sensor noise
            noise = self.rng.normal(0, self.noise_std, size=12)
            measured_state = true_state + noise

            # DoS Attack: Decide whether to drop packet
            dropped = False
            if self.is_attack_active(self.t):
                if self.rng.random() < self.drop_rate:
                    dropped = True
                    self.packets_dropped += 1
                    print(f"[ATTACK] DoS Attack - Packet DROPPED at t={self.t:.2f}s")

            # Send measurement to queues (or None if dropped)
            if not dropped:
                for q in self.output_queues:
                    try:
                        q.put_nowait(DoS_SensorData(self.t, measured_state))
                    except queue.Full:
                        pass
                self.packets_sent += 1
            else:
                # Send None to indicate dropped packet
                for q in self.output_queues:
                    try:
                        q.put_nowait(DoS_SensorData(self.t, None))  # None indicates dropped packet
                    except queue.Full:
                        pass

            time.sleep(self.dt)
            self.t += self.dt

        print(f"[{self.name}] Stopped")
        print(f"[{self.name}] Packets Dropped: {self.packets_dropped}/{self.packets_sent + self.packets_dropped} ({self.packets_dropped/(self.packets_sent+self.packets_dropped)*100:.1f}%)")


class DoS_SensorData:
    """Container for sensor data with timestamp - DoS specific (allows None state)"""
    def __init__(self, timestamp, state):
        self.timestamp = timestamp
        self.state = state  # None if packet was dropped


# =====================================================
# STATE ESTIMATOR DEFENSE
# =====================================================

class PacketDropDetector:
    """
    Detects packet drops and tracks statistics
    """
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.drop_history = deque(maxlen=window_size)
        self.drop_count = 0
        self.total_packets = 0

    def update(self, dropped: bool):
        """Update with latest packet status"""
        self.total_packets += 1
        if dropped:
            self.drop_count += 1
        self.drop_history.append(1 if dropped else 0)
        return dropped

    def get_current_drop_rate(self):
        """Get current drop rate over sliding window"""
        if len(self.drop_history) == 0:
            return 0.0
        return sum(self.drop_history) / len(self.drop_history)


class StateEstimator:
    """
    State estimator for handling packet drops
    Predicts state using physics model when measurements are unavailable
    """
    def __init__(self, dt=0.01):
        self.dt = dt
        self.last_valid_state = np.zeros(12)
        self.estimated_state = np.zeros(12)
        self.last_valid_time = 0.0
        self.estimation_count = 0

    def update(self, t, measurement, last_control):
        """
        Update state estimate
        Returns: (estimated_state, was_estimating)
        """
        if measurement is not None:
            # Valid measurement received
            self.last_valid_state = measurement.copy()
            self.estimated_state = measurement.copy()
            self.last_valid_time = t
            return self.estimated_state, False
        else:
            # Packet dropped - use physics-based prediction
            self.estimation_count += 1

            # Simple prediction: constant velocity model
            # x_pos += x_vel * dt
            self.estimated_state[1] += self.estimated_state[0] * self.dt  # x position
            self.estimated_state[3] += self.estimated_state[2] * self.dt  # y position
            self.estimated_state[5] += self.estimated_state[4] * self.dt  # z position

            # Apply slight decay to velocities (model drag)
            self.estimated_state[0] *= 0.98  # x velocity
            self.estimated_state[2] *= 0.98  # y velocity
            self.estimated_state[4] *= 0.98  # z velocity

            return self.estimated_state, True


# =====================================================
# MODIFIED SIMULATOR WITH DEFENSE INTEGRATION
# =====================================================

class DoS_Simulator(Simulator):
    """
    Simulator with packet drop detection and state estimation
    """
    def __init__(self, detector, estimator, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.estimator = estimator

        # Logging for attack scenario
        self.packet_dropped_log = []
        self.estimating_log = []
        self.drop_rate_log = []

    def run(self):
        print(f"[{self.name}] Started with State Estimator Defense")

        while self.t <= self.T and not self.stop_event.is_set():
            # Get latest control
            try:
                cmd = self.control_queue.get(timeout=0.01)
                self.u = cmd.u
            except queue.Empty:
                pass

            # Always integrate dynamics normally (no state manipulation)
            with self.state_lock:
                self.x = rk4_step(self.x, self.u, self.f, self.Ts)

            # Log packet drop statistics
            current_drop_rate = self.detector.get_current_drop_rate()
            self.drop_rate_log.append(current_drop_rate)

            self.t += self.Ts
            time.sleep(self.Ts)

        print(f"[{self.name}] Finished at t={self.t:.2f}s")


# =====================================================
# MODIFIED CONTROLLER WITH STATE ESTIMATOR
# =====================================================

class DoS_Controller(Controller):
    """
    Controller with state estimator for handling packet drops
    """
    def __init__(self, detector, estimator, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.estimator = estimator
        self.u = np.zeros(4)  # Initialize control vector

    def run(self):
        print(f"[{self.name}] Started with State Estimator")

        while not self.stop_event.is_set():
            # Get sensor data
            try:
                sensor_data = self.sensor_queue.get(timeout=0.01)

                # Check if packet was dropped
                dropped = (sensor_data.state is None)
                self.detector.update(dropped)

                # Use state estimator
                estimated_state, was_estimating = self.estimator.update(
                    sensor_data.timestamp,
                    sensor_data.state,
                    self.u
                )

                if was_estimating:
                    print(f"[DEFENSE] Using estimated state (packet dropped)")

                # Store for logging
                self.current_sensor = estimated_state.copy()

                # Get reference from ground station
                try:
                    ref_data = self.traj_queue.get_nowait()
                    self.current_ref = ref_data.r
                except queue.Empty:
                    pass

                # Compute control using estimated state
                error = estimated_state - self.current_ref
                self.u = -self.gains.K @ error

                # Send to actuator (saturation happens in Actuator thread)
                try:
                    self.output_queue.put_nowait(ControlCommand(sensor_data.timestamp, self.u.copy()))
                except queue.Full:
                    pass

            except queue.Empty:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Stopped")


# =====================================================
# MAIN SIMULATION
# =====================================================

def run_dos_scenario():
    """
    Run Scenario 2: DoS Attack with State Estimator Defense
    """
    print("\n" + "="*80)
    print("SCENARIO 2: DoS ATTACK - MULTITHREADED WITH STATE ESTIMATOR DEFENSE")
    print("="*80)
    print("Attack: Denial of Service (Packet Dropping)")
    print("Defense: Packet Drop Detection + State Estimator")
    print("="*80 + "\n")

    # Simulation parameters
    T = 10.0  # Short demo for presentation (was 30.0)
    Ts = 0.01

    # Load gains and create trajectory
    gains = load_gains('mat')
    traj_func = traj_figure8(amp=0.6, period=8.0, z0=0.5, t_start=1.0)

    # Create defense mechanisms
    detector = PacketDropDetector(window_size=50)
    estimator = StateEstimator(dt=Ts)

    # Create synchronization objects
    stop_event = threading.Event()
    sensor_to_controller = queue.Queue(maxsize=10)
    sensor_to_ground = queue.Queue(maxsize=10)
    ground_to_controller = queue.Queue(maxsize=10)
    controller_to_actuator = queue.Queue(maxsize=10)
    actuator_to_sim = queue.Queue(maxsize=10)

    # Create logger
    logger = SimulationLogger()

    # Create simulator (with defense integration)
    simulator = DoS_Simulator(
        detector=detector,
        estimator=estimator,
        T=T, Ts=Ts,
        control_queue=actuator_to_sim,
        logger=logger,
        stop_event=stop_event,
        plant='nonlinear'
    )

    # Create DoS attack sensor
    sensor = DoS_Sensor(
        attack_start=4.0,  # Start attack at t=4s
        attack_duration=2.0,  # Brief 2-second attack
        drop_rate=0.6,  # 60% packet drop rate
        rate_hz=100,
        noise_std=0.05,
        state_getter=simulator.get_state,
        output_queues=[sensor_to_controller, sensor_to_ground],
        stop_event=stop_event,
        seed=42
    )

    # Create ground station
    ground_station = GroundStation(
        traj_func=traj_func,
        T=T,
        rate_hz=50,
        sensor_queue=sensor_to_ground,
        output_queue=ground_to_controller,
        gains=gains,
        use_rg=False,  # No RG for this scenario
        stop_event=stop_event
    )

    # Create controller with state estimator
    controller = DoS_Controller(
        detector=detector,
        estimator=estimator,
        gains=gains,
        use_integrator=False,
        sensor_queue=sensor_to_controller,
        traj_queue=ground_to_controller,
        output_queue=controller_to_actuator,
        stop_event=stop_event,
        rate_hz=250
    )

    # Create actuator
    actuator = Actuator(
        rate_hz=500,
        input_queue=controller_to_actuator,
        output_queue=actuator_to_sim,
        stop_event=stop_event
    )

    # Start all threads
    threads = [simulator, sensor, ground_station, controller, actuator]
    for thread in threads:
        thread.start()

    print(f"\nAll modules running...\n")

    # Logging loop with real-time position printing
    def logging_loop():
        last_print_time = -1.0
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

    # Wait for simulator to finish (it has a time limit)
    simulator.join()

    # Signal all threads to stop
    stop_event.set()

    # Wait for other threads to finish
    for thread in threads:
        if thread != simulator:  # Already joined above
            thread.join(timeout=2.0)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Packets Dropped: {detector.drop_count}/{detector.total_packets} ({detector.drop_count/max(detector.total_packets,1)*100:.1f}%)")
    print(f"State Estimations: {estimator.estimation_count}")
    print("="*80 + "\n")

    # Get logged data
    data = logger.get_arrays()
    data['packets_dropped'] = detector.drop_count
    data['packets_total'] = detector.total_packets
    data['drop_rate'] = detector.drop_count / max(detector.total_packets, 1)
    data['estimation_count'] = estimator.estimation_count
    data['attack_start'] = sensor.attack_start
    data['attack_end'] = sensor.attack_end
    data['drop_rate_log'] = np.array(simulator.drop_rate_log)

    return data


# =====================================================
# Plotting
# =====================================================

def plot_dos_results(data: Dict[str, Any]):
    """Plot results showing DoS attack with state estimator defense"""
    from mpl_toolkits.mplot3d import Axes3D

    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']

    # Synchronize array lengths
    min_len = min(len(t), len(data['drop_rate_log']))
    t = t[:min_len]
    X = X[:, :min_len]
    R = R[:, :min_len]
    U = U[:, :min_len]
    drop_rate_log = data['drop_rate_log'][:min_len]

    # Attack period
    attack_start = data['attack_start']
    attack_end = data['attack_end']
    attack_mask = (t >= attack_start) & (t <= attack_end)

    # Figure 1: 3D Trajectory - Continuous path with color-coded segments
    fig1 = plt.figure(figsize=(14, 10))
    ax = fig1.add_subplot(111, projection='3d')

    # Plot continuous trajectory with color segments
    # Pre-attack (blue)
    pre_attack = t < attack_start
    if np.any(pre_attack):
        ax.plot(X[1, pre_attack], X[3, pre_attack], X[5, pre_attack],
                'b-', linewidth=2.5, alpha=0.9, label='Normal Operation')

    # During attack (orange)
    if np.any(attack_mask):
        ax.plot(X[1, attack_mask], X[3, attack_mask], X[5, attack_mask],
                color='orange', linewidth=2.5, alpha=0.9, label='Under DoS Attack')

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
    ax.set_title('3D Trajectory - DoS Attack with State Estimator Defense\n(Blue=Normal, Orange=Attack, Green=Recovery)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([0, 1.0])
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    # Figure 2: Position Tracking
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9))
    positions = ['X', 'Y', 'Z']
    indices = [1, 3, 5]

    for i, (pos, idx) in enumerate(zip(positions, indices)):
        axes[i].plot(t, X[idx, :], 'b-', linewidth=2, label='Actual', alpha=0.8)
        axes[i].plot(t, R[idx, :], 'r--', linewidth=2, alpha=0.6, label='Reference')
        axes[i].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period' if i == 0 else '')
        axes[i].set_ylabel(f'{pos} [m]', fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title('Position Tracking - DoS Attack with State Estimator Defense', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()

    # Figure 3: Packet Drop Detection and State Estimation
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 7))

    # Packet drop rate over time
    drop_rate_threshold = 0.3  # Threshold for degraded mode (30%)
    axes3[0].plot(t, drop_rate_log * 100, 'b-', linewidth=1.5, label='Packet Drop Rate (50-sample window)')
    axes3[0].axhline(y=drop_rate_threshold*100, color='r', linestyle='--', linewidth=2,
                     label=f'Degradation Threshold = {drop_rate_threshold*100:.0f}%')
    axes3[0].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period')
    axes3[0].set_ylabel('Drop Rate [%]', fontsize=11)
    axes3[0].set_title('Packet Drop Rate Detection', fontsize=13, fontweight='bold')
    axes3[0].legend(fontsize=9, loc='upper left')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].set_ylim([0, 100])

    # Position error (shows estimator performance)
    position_error = np.sqrt((X[1, :] - R[1, :])**2 + (X[3, :] - R[3, :])**2 + (X[5, :] - R[5, :])**2)
    axes3[1].plot(t, position_error, 'purple', linewidth=1.5, label='Position Error')
    axes3[1].axvspan(attack_start, attack_end, alpha=0.15, color='orange', label='Attack Period')
    axes3[1].set_ylabel('Error [m]', fontsize=11)
    axes3[1].set_xlabel('Time [s]', fontsize=11)
    axes3[1].set_title('Position Tracking Error (shows State Estimator performance)', fontsize=13, fontweight='bold')
    axes3[1].legend(fontsize=9)
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    # Calculate performance metrics
    xy_error = np.sqrt((X[1, :] - R[1, :]) ** 2 + (X[3, :] - R[3, :]) ** 2)
    print(f"\nPerformance Metrics:")
    print(f"  XY RMS Error: {np.sqrt(np.mean(xy_error**2)):.4f} m")
    print(f"  XY Max Error: {np.max(xy_error):.4f} m")
    print(f"  Z RMS Error: {np.sqrt(np.mean((X[5,:] - R[5,:])**2)):.4f} m\n")


if __name__ == '__main__':
    data = run_dos_scenario()
    plot_dos_results(data)
