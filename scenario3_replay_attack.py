"""
SCENARIO 3: REPLAY ATTACK WITH REFERENCE GOVERNOR DEFENSE

Attack: Stuxnet-style replay attack (record → replay + malicious control injection)
Defense: Reference Governor (safety-critical control override)

npm run scenario3
"""

import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time
import math
from typing import Dict, Any
from collections import deque

# Import multithreaded components
from main_multithreaded import (
    Sensor, SensorData, ControlCommand, Simulator, Controller, GroundStation, Actuator,
    SimulationLogger, Gains, load_gains, traj_figure8, govern_reference,
    f_nonlinear, rk4_step,
    FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX, RG_ITERS
)


# =====================================================
# REPLAY ATTACK - Actuator with Recording & Replay
# =====================================================

class ReplayAttack_Actuator(threading.Thread):
    """
    Actuator with Stuxnet-style Replay Attack
    Phase 1: Record normal control commands
    Phase 2: Replay old commands + inject malicious control
    """
    def __init__(self, record_start=2.0, record_duration=3.0,
                 replay_start=7.0, replay_duration=3.0,
                 injection_magnitude=1.5,
                 rate_hz=500, input_queue=None, output_queue=None, stop_event=None):
        super().__init__(name="Actuator")
        self.daemon = True

        # Attack parameters
        self.record_start = record_start
        self.record_duration = record_duration
        self.record_end = record_start + record_duration
        self.replay_start = replay_start
        self.replay_duration = replay_duration
        self.replay_end = replay_start + replay_duration
        self.injection_magnitude = injection_magnitude

        # Actuator parameters
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

        # Recording buffer
        self.recorded_commands = []
        self.replay_index = 0

        # Phase tracking
        self.phase = "idle"
        self.malicious_count = 0

        # Control limits
        self.u_max = np.array([FZ_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX])
        self.u_min = -self.u_max

    def saturate(self, u):
        """Apply control saturation"""
        u_sat = np.clip(u, self.u_min, self.u_max)
        return u_sat

    def generate_malicious_control(self, normal_u, t):
        """
        Generate malicious control commands
        Inject false data to deviate trajectory
        """
        t_attack = t - self.replay_start
        ramp = min(t_attack / 1.0, 1.0)  # Ramp up over 1 second

        malicious_u = normal_u.copy()

        # Inject aggressive commands to deviate drone
        malicious_u[0] += -3.0 * ramp * self.injection_magnitude  # Reduce thrust
        malicious_u[1] += 4.0 * ramp * self.injection_magnitude * math.sin(2 * math.pi * t / 2)  # Roll
        malicious_u[2] += 4.0 * ramp * self.injection_magnitude * math.cos(2 * math.pi * t / 2)  # Pitch
        malicious_u[3] += 2.0 * ramp * self.injection_magnitude * math.sin(math.pi * t)  # Yaw

        self.malicious_count += 1
        return malicious_u

    def run(self):
        print(f"[{self.name}] Started")
        print(f"   Replay Attack: Recording={self.record_start}-{self.record_end}s, Replay={self.replay_start}-{self.replay_end}s")

        cmd_count = 0
        last_status_time = -999.0

        while not self.stop_event.is_set():
            try:
                cmd = self.input_queue.get(timeout=0.01)
                cmd_count += 1

                # Use command timestamp for phase detection (NOT self.t)
                t = cmd.timestamp

                # Debug: Print status every 2 seconds
                if t - last_status_time >= 2.0:
                    print(f"[ACTUATOR] t={t:.2f}s, received {cmd_count} commands total")
                    last_status_time = t

                # Phase 1: Recording normal commands
                if self.record_start <= t <= self.record_end:
                    if self.phase != "recording":
                        self.phase = "recording"
                        print(f"\n[REC] [REPLAY ATTACK - PHASE 1: RECORDING] t={t:.2f}s")
                        print(f"   Eavesdropping on control commands...")

                    # Record the command
                    self.recorded_commands.append(cmd.u.copy())

                    # Pass through normal control
                    u_sat = self.saturate(cmd.u)
                    actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                    self.output_queue.put(actuator_cmd, timeout=0.1)

                # Phase 2: Replay + Malicious Injection
                elif self.replay_start <= t <= self.replay_end:
                    if self.phase != "replay":
                        self.phase = "replay"
                        print(f"\n[REPLAY] [REPLAY ATTACK - PHASE 2: REPLAY + MALICIOUS INJECTION] t={t:.2f}s")
                        print(f"   Replaying {len(self.recorded_commands)} recorded commands")
                        print(f"   Injecting malicious control (magnitude={self.injection_magnitude})...")

                    if len(self.recorded_commands) > 0:
                        # Get replayed command (loop through recorded buffer)
                        replayed_u = self.recorded_commands[self.replay_index % len(self.recorded_commands)]
                        self.replay_index += 1

                        # Inject malicious control on top of replayed command
                        malicious_u = self.generate_malicious_control(replayed_u, t)

                        # Apply saturation and send to plant
                        u_sat = self.saturate(malicious_u)
                        actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                        self.output_queue.put(actuator_cmd, timeout=0.1)
                    else:
                        # Fallback if no recorded commands
                        u_sat = self.saturate(cmd.u)
                        actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                        self.output_queue.put(actuator_cmd, timeout=0.1)

                # Normal operation (idle)
                else:
                    if self.phase == "replay":
                        print(f"\n[END] [REPLAY ATTACK ENDED] t={t:.2f}s")
                        print(f"   Total malicious commands injected: {self.malicious_count}")
                        self.phase = "idle"

                    # Pass through normal control
                    u_sat = self.saturate(cmd.u)
                    actuator_cmd = ControlCommand(timestamp=cmd.timestamp, u=u_sat)
                    self.output_queue.put(actuator_cmd, timeout=0.1)

            except queue.Empty:
                pass
            except queue.Full:
                pass  # Ignore queue full errors during shutdown

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# =====================================================
# CONTROLLER WITH SIMULATION TIME - For Replay Attack
# =====================================================

class SimTimeController(Controller):
    """Controller that uses simulation time for timestamps instead of wall clock time"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = None  # Will be set on first loop

    def run(self):
        print(f"[{self.name}] Started - INT={self.use_integrator}")

        while not self.stop_event.is_set():
            # Initialize start time on first loop
            if self.start_time is None:
                self.start_time = time.time()

            # Calculate simulation time from elapsed wall-clock time
            sim_time = time.time() - self.start_time

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

            # Send to actuator with SIMULATION TIME (elapsed wall-clock time)
            cmd = ControlCommand(timestamp=sim_time, u=u)
            try:
                self.output_queue.put(cmd, timeout=0.01)
            except queue.Full:
                pass

            time.sleep(self.dt)

        print(f"[{self.name}] Finished")


# =====================================================
# REFERENCE GOVERNOR DEFENSE - Ground Station with RG
# =====================================================

class RG_GroundStation(GroundStation):
    """
    Ground Station with Safety-Based Reference Governor
    Monitors state for unsafe conditions and corrects trajectory
    """
    def __init__(self, traj_func, T, gains, rate_hz=50,
                 sensor_queue=None, output_queue=None, stop_event=None,
                 max_tilt_deg=30, min_altitude=0.2, max_velocity=15.0, max_accel=20.0):
        # Initialize WITHOUT RG (we'll implement our own)
        super().__init__(
            traj_func=traj_func,
            T=T,
            rate_hz=rate_hz,
            output_queue=output_queue,
            sensor_queue=sensor_queue,
            gains=gains,
            use_rg=False,  # Disable baseline RG, use custom safety-based RG
            stop_event=stop_event
        )

        # Safety thresholds for RG
        self.max_tilt = math.radians(max_tilt_deg)
        self.min_altitude = min_altitude
        self.max_velocity = max_velocity
        self.max_accel = max_accel

        # Track RG interventions
        self.rg_log = []
        self.intervention_count = 0
        self.intervention_active = False

        # Previous state for derivative calculation
        self.prev_state = np.zeros(12)
        self.prev_time = 0.0

    def check_safety(self, state):
        """Check if current state violates safety constraints"""
        violations = []

        # Check tilt (roll and pitch)
        phi, theta = state[7], state[9]
        tilt = math.sqrt(phi**2 + theta**2)
        if abs(tilt) > self.max_tilt:
            violations.append(f"tilt={math.degrees(tilt):.1f}deg")

        # Check altitude
        z = state[5]
        if z < self.min_altitude:
            violations.append(f"altitude={z:.2f}m")

        # Check velocity magnitude
        vel = math.sqrt(state[0]**2 + state[2]**2 + state[4]**2)
        if vel > self.max_velocity:
            violations.append(f"velocity={vel:.2f}m/s")

        return violations

    def safe_reference(self, current_state, desired_ref, dt):
        """Generate a safe reference that gradually approaches desired while maintaining safety"""
        # Blend between current state and desired reference
        # Use smaller blend factor to slow down aggressive reference changes
        blend = 0.3  # 30% toward desired per timestep

        safe_ref = current_state.copy()

        # Gradually approach desired positions (not velocities)
        safe_ref[1] = current_state[1] + blend * (desired_ref[1] - current_state[1])  # x
        safe_ref[3] = current_state[3] + blend * (desired_ref[3] - current_state[3])  # y
        safe_ref[5] = max(current_state[5] + blend * (desired_ref[5] - current_state[5]), self.min_altitude)  # z

        # Keep velocities reasonable (don't follow desired velocities during attack)
        safe_ref[0] = desired_ref[0] * 0.5  # xdot
        safe_ref[2] = desired_ref[2] * 0.5  # ydot
        safe_ref[4] = desired_ref[4] * 0.5  # zdot

        # Zero out angular references for stability
        safe_ref[6:12] = 0.0

        return safe_ref

    def run(self):
        """Run with safety-based Reference Governor"""
        print(f"[{self.name}] Started - RG=True (Safety-Based Reference Governor ACTIVE)")
        t = 0.0

        while t <= self.T and not self.stop_event.is_set():
            # Get current state
            try:
                sensor_data = self.sensor_queue.get_nowait()
                self.current_state = sensor_data.state
            except queue.Empty:
                pass

            # Generate desired reference
            r_desired = self.traj_func(t)
            self.current_raw_ref = r_desired.copy()

            # Check safety
            violations = self.check_safety(self.current_state)
            needs_intervention = len(violations) > 0

            # Apply RG if unsafe
            if needs_intervention:
                if not self.intervention_active:
                    self.intervention_active = True
                    self.intervention_count += 1
                    print(f"\n[RG!] [REFERENCE GOVERNOR INTERVENTION #{self.intervention_count}] t={t:.2f}s")
                    print(f"   Safety violations detected: {', '.join(violations)}")
                    print(f"   Overriding reference to maintain safe flight")

                # Generate safe reference
                dt = t - self.prev_time
                r_governed = self.safe_reference(self.current_state, r_desired, dt)
                self.rg_log.append(1)  # RG active
            else:
                if self.intervention_active:
                    self.intervention_active = False
                    print(f"[RG-OK] [REFERENCE GOVERNOR RELEASE] t={t:.2f}s")
                    print(f"   System safe, resuming normal trajectory tracking")

                r_governed = r_desired
                self.rg_log.append(0)  # RG inactive

            # Send reference
            from dataclasses import dataclass
            @dataclass
            class TrajPoint:
                r: np.ndarray
            traj_point = TrajPoint(r=r_governed)
            try:
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        break
                self.output_queue.put(traj_point, timeout=0.01)
            except queue.Full:
                pass

            self.prev_state = self.current_state.copy()
            self.prev_time = t
            time.sleep(self.dt)
            t += self.dt

        print(f"[{self.name}] Finished")


# =====================================================
# MAIN SIMULATION
# =====================================================

def run_replay_scenario():
    """
    Run Scenario 3: Replay Attack with Reference Governor Defense
    """
    print("\n" + "="*80)
    print("SCENARIO 3: REPLAY ATTACK - MULTITHREADED WITH REFERENCE GOVERNOR")
    print("="*80)
    print("Attack: Stuxnet-style Replay (Record -> Replay + Malicious Injection)")
    print("Defense: Reference Governor (Safety-Critical Control Override)")
    print("="*80 + "\n")

    # Simulation parameters
    T = 18.0  # Extended time to see full attack + recovery
    Ts = 0.01

    # Load gains and create trajectory
    gains = load_gains('mat')
    traj_func = traj_figure8(amp=0.6, period=8.0, z0=0.5, t_start=1.0)

    # Create synchronization objects
    stop_event = threading.Event()
    sensor_to_controller = queue.Queue(maxsize=10)
    sensor_to_ground = queue.Queue(maxsize=10)
    ground_to_controller = queue.Queue(maxsize=10)
    controller_to_actuator = queue.Queue(maxsize=10)
    actuator_to_sim = queue.Queue(maxsize=10)

    # Create logger
    logger = SimulationLogger()

    # Create simulator
    simulator = Simulator(
        T=T, Ts=Ts,
        control_queue=actuator_to_sim,
        logger=logger,
        stop_event=stop_event,
        plant='nonlinear'
    )

    # Create normal sensor (no attack here)
    sensor = Sensor(
        rate_hz=100,
        noise_std=0.05,
        state_getter=simulator.get_state,
        output_queues=[sensor_to_controller, sensor_to_ground],
        stop_event=stop_event,
        seed=42
    )

    # Create ground station with Reference Governor
    ground_station = RG_GroundStation(
        traj_func=traj_func,
        T=T,
        gains=gains,
        rate_hz=50,
        sensor_queue=sensor_to_ground,
        output_queue=ground_to_controller,
        stop_event=stop_event,
        max_tilt_deg=45,  # Relaxed from 30 to allow normal figure-8 dynamics
        min_altitude=0.05,  # Lowered from 0.2 to avoid false positives during startup
        max_velocity=20.0,  # Increased from 15.0 to allow normal maneuvers
        max_accel=30.0  # Increased from 20.0
    )

    # Create controller with simulation time tracking (needed for replay attack)
    controller = SimTimeController(
        gains=gains,
        use_integrator=False,
        sensor_queue=sensor_to_controller,
        traj_queue=ground_to_controller,
        output_queue=controller_to_actuator,
        stop_event=stop_event,
        rate_hz=250
    )

    # Create actuator with REPLAY ATTACK
    actuator = ReplayAttack_Actuator(
        record_start=2.0,  # Start recording at t=2s
        record_duration=3.0,  # Record for 3 seconds
        replay_start=8.0,  # Start replay at t=8s (after first figure-8)
        replay_duration=3.0,  # Replay for 3 seconds
        injection_magnitude=3.0,  # STRONGER attack to make it visible
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

    # Wait for simulator to finish
    simulator.join()

    # Signal all threads to stop
    stop_event.set()

    # Wait for other threads to finish
    for thread in threads:
        if thread != simulator:
            thread.join(timeout=2.0)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Recorded Commands: {len(actuator.recorded_commands)}")
    print(f"Malicious Commands Injected: {actuator.malicious_count}")
    print(f"RG Interventions: {ground_station.intervention_count}")
    print("="*80 + "\n")

    # Get logged data
    data = logger.get_arrays()
    data['rg_active'] = np.array(ground_station.rg_log)
    data['record_start'] = actuator.record_start
    data['record_end'] = actuator.record_end
    data['replay_start'] = actuator.replay_start
    data['replay_end'] = actuator.replay_end
    data['rg_count'] = ground_station.intervention_count

    return data


# =====================================================
# Plotting
# =====================================================

def plot_replay_results(data: Dict[str, Any]):
    """Plot results showing replay attack with reference governor defense"""
    from mpl_toolkits.mplot3d import Axes3D

    t = data['t']
    X = data['X']
    R = data['R']
    U = data['U']

    # Synchronize array lengths
    min_len = min(len(t), len(data['rg_active']))
    t = t[:min_len]
    X = X[:, :min_len]
    R = R[:, :min_len]
    U = U[:, :min_len]
    rg_active = data['rg_active'][:min_len]

    # Attack periods
    record_start = data['record_start']
    record_end = data['record_end']
    replay_start = data['replay_start']
    replay_end = data['replay_end']

    recording_mask = (t >= record_start) & (t <= record_end)
    replay_mask = (t >= replay_start) & (t <= replay_end)

    # Figure 1: 3D Trajectory with attack phases
    fig1 = plt.figure(figsize=(14, 10))
    ax = fig1.add_subplot(111, projection='3d')

    # Color-coded trajectory segments
    # Normal operation (before recording)
    pre_record = t < record_start
    if np.any(pre_record):
        ax.plot(X[1, pre_record], X[3, pre_record], X[5, pre_record],
                'b-', linewidth=2.5, alpha=0.9, label='Normal Operation')

    # Recording phase (cyan)
    if np.any(recording_mask):
        ax.plot(X[1, recording_mask], X[3, recording_mask], X[5, recording_mask],
                'cyan', linewidth=2.5, alpha=0.9, label='Recording Phase')

    # Between recording and replay
    between = (t > record_end) & (t < replay_start)
    if np.any(between):
        ax.plot(X[1, between], X[3, between], X[5, between],
                'b-', linewidth=2.5, alpha=0.9)

    # Replay attack (orange/red)
    if np.any(replay_mask):
        ax.plot(X[1, replay_mask], X[3, replay_mask], X[5, replay_mask],
                color='orange', linewidth=2.5, alpha=0.9, label='Under Attack (Replay)')

    # Post-attack recovery (green)
    post_attack = t > replay_end
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
    ax.set_title('3D Trajectory - Replay Attack with Reference Governor Defense\n(Blue=Normal, Cyan=Recording, Orange=Attack, Green=Recovery)',
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
        axes[i].axvspan(record_start, record_end, alpha=0.15, color='cyan', label='Recording' if i == 0 else '')
        axes[i].axvspan(replay_start, replay_end, alpha=0.15, color='orange', label='Replay Attack' if i == 0 else '')
        axes[i].set_ylabel(f'{pos} [m]', fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title('Position Tracking - Replay Attack with RG Defense', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()

    # Figure 3: Reference Governor Activity
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 9))

    # Position error
    position_error = np.sqrt((X[1, :] - R[1, :])**2 + (X[3, :] - R[3, :])**2 + (X[5, :] - R[5, :])**2)
    axes3[0].plot(t, position_error, 'purple', linewidth=1.5, label='Position Error')
    axes3[0].axvspan(record_start, record_end, alpha=0.15, color='cyan', label='Recording')
    axes3[0].axvspan(replay_start, replay_end, alpha=0.15, color='orange', label='Replay Attack')
    axes3[0].fill_between(t, 0, np.max(position_error)*1.2, where=(rg_active > 0),
                          alpha=0.2, color='green', label='RG Active')
    axes3[0].set_ylabel('Error [m]', fontsize=11)
    axes3[0].set_title('Position Tracking Error (shows RG activation)', fontsize=13, fontweight='bold')
    axes3[0].legend(fontsize=9)
    axes3[0].grid(True, alpha=0.3)

    # Control magnitude
    control_mag = np.sqrt(U[0, :]**2 + U[1, :]**2 + U[2, :]**2 + U[3, :]**2)
    axes3[1].plot(t, control_mag, 'blue', linewidth=1.5, label='Control Magnitude')
    axes3[1].axvspan(record_start, record_end, alpha=0.15, color='cyan', label='Recording')
    axes3[1].axvspan(replay_start, replay_end, alpha=0.15, color='orange', label='Replay Attack')
    axes3[1].fill_between(t, 0, np.max(control_mag)*1.2, where=(rg_active > 0),
                          alpha=0.2, color='green', label='RG Active')
    axes3[1].set_ylabel('Control Mag', fontsize=11)
    axes3[1].set_title('Control Command Magnitude', fontsize=13, fontweight='bold')
    axes3[1].legend(fontsize=9)
    axes3[1].grid(True, alpha=0.3)

    # RG intervention timeline
    axes3[2].fill_between(t, 0, rg_active, alpha=0.5, color='green', label='Reference Governor Active')
    axes3[2].axvspan(record_start, record_end, alpha=0.15, color='cyan', label='Recording')
    axes3[2].axvspan(replay_start, replay_end, alpha=0.15, color='orange', label='Replay Attack')
    axes3[2].set_ylabel('RG Status', fontsize=11)
    axes3[2].set_xlabel('Time [s]', fontsize=11)
    axes3[2].set_title('Reference Governor Intervention Timeline', fontsize=13, fontweight='bold')
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
    data = run_replay_scenario()
    plot_replay_results(data)
