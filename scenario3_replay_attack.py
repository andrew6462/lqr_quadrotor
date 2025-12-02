"""
SCENARIO 3: REPLAY ATTACK WITH REFERENCE GOVERNOR DEFENSE

npm run scenario3

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import math
from collections import deque

# Import from existing code
from main_cps import load_matrix_K, quadrotor_u, figure8


# =====================================================
# ============   REPLAY ATTACK   ======================
# =====================================================

class ReplayAttack:
    """
    Stuxnet-Style Replay Attack
    Phase 1: Record normal sensor data
    Phase 2: Replay old data + inject malicious actuator commands
    """
    def __init__(self, record_start=5.0, record_duration=5.0,
                 replay_start=15.0, replay_duration=12.0, 
                 attack_intensity='aggressive'):
        self.name = "Replay_Attack"
        self.record_start = record_start
        self.record_duration = record_duration
        self.replay_start = replay_start
        self.replay_duration = replay_duration
        self.attack_intensity = attack_intensity
        
        # Recording phase
        self.recorded_measurements = []
        self.replay_index = 0
        
        # Phase tracking
        self.phase = "idle"  # idle, recording, replay
        self.malicious_injection_count = 0
        
    def compute_malicious_control(self, t, normal_control):
        """
        Generate malicious control commands to crash or deviate drone
        """
        t_attack = t - self.replay_start
        ramp = min(t_attack / 2.0, 1.0)  # Ramp up over 2 seconds
        
        malicious_u = normal_control.copy()
        
        if self.attack_intensity == 'aggressive':
            # Try to flip drone and crash
            malicious_u[0] = -5.0 * ramp  # Negative thrust (drop)
            malicious_u[1] = 6.0 * ramp * math.sin(2 * math.pi * t / 2)  # Roll torque
            malicious_u[2] = 6.0 * ramp * math.cos(2 * math.pi * t / 2)  # Pitch torque
            malicious_u[3] = 3.0 * ramp * math.sin(math.pi * t)  # Yaw torque
            
        elif self.attack_intensity == 'moderate':
            # Try to deviate trajectory significantly
            malicious_u[0] += -2.0 * ramp  # Reduce thrust
            malicious_u[1] += 3.0 * ramp * math.sin(2 * math.pi * t / 3)
            malicious_u[2] += 3.0 * ramp * math.cos(2 * math.pi * t / 3)
            
        elif self.attack_intensity == 'subtle':
            # Subtle deviation (harder to detect)
            malicious_u[0] += -1.0 * ramp
            malicious_u[1] += 1.5 * ramp * math.sin(math.pi * t)
            malicious_u[2] += 1.5 * ramp * math.cos(math.pi * t)
        
        self.malicious_injection_count += 1
        return malicious_u
        
    def update(self, t, true_measurement, normal_control):
        """
        Execute replay attack phases
        Returns: (sensor_output, control_to_plant, phase_info)
        """
        record_end = self.record_start + self.record_duration
        replay_end = self.replay_start + self.replay_duration
        
        # Phase 1: Recording
        if self.record_start <= t <= record_end:
            if self.phase != "recording":
                self.phase = "recording"
                print(f"\n[REC] [REPLAY ATTACK - PHASE 1: RECORDING] t={t:.2f}s")
                print(f"   Eavesdropping on sensor measurements...")
            
            self.recorded_measurements.append(true_measurement.copy())
            return true_measurement, normal_control, "recording"
        
        # Phase 2: Replay + Malicious Control Injection
        elif self.replay_start <= t <= replay_end:
            if self.phase != "replay":
                self.phase = "replay"
                print(f"\n[REPLAY] [REPLAY ATTACK - PHASE 2: REPLAY + FDI] t={t:.2f}s")
                print(f"   Replaying {len(self.recorded_measurements)} recorded measurements")
                print(f"   Injecting malicious control commands ({self.attack_intensity})...")
            
            if len(self.recorded_measurements) > 0:
                # Replay old sensor data to controller (make it think everything is fine)
                replayed_measurement = self.recorded_measurements[
                    self.replay_index % len(self.recorded_measurements)
                ]
                self.replay_index += 1
                
                # Inject malicious control commands
                malicious_control = self.compute_malicious_control(t, normal_control)
                
                # Controller sees replayed (old) data, but malicious commands go to plant
                return replayed_measurement, malicious_control, "replay"
        
        # Normal operation
        else:
            if self.phase != "idle":
                if self.phase == "replay":
                    print(f"\n[END] [REPLAY ATTACK ENDED] t={t:.2f}s")
                    print(f"   Total malicious commands injected: {self.malicious_injection_count}")
                self.phase = "idle"
            return true_measurement, normal_control, "idle"
        
        return true_measurement, normal_control, self.phase
    
    def is_active(self, t):
        replay_end = self.replay_start + self.replay_duration
        return self.replay_start <= t <= replay_end


# =====================================================
# ============   REFERENCE GOVERNOR   =================
# =====================================================

class SafetyReferenceGovernor:
    """
    Reference Governor for Safety-Critical Control
    Monitors control commands and state for safety violations
    Overrides malicious commands with safe alternatives
    """
    def __init__(self, K, max_tilt_deg=30, min_altitude=0.3, 
                 max_velocity=15.0, max_control_magnitude=12.0):
        self.K = K
        self.max_tilt = math.radians(max_tilt_deg)
        self.min_altitude = min_altitude
        self.max_velocity = max_velocity
        self.max_control_magnitude = max_control_magnitude
        
        # Lyapunov matrix for safety regions
        self.P = np.eye(12)
        self.safety_threshold = 25.0
        
        # Governor state
        self.governed_ref = np.zeros(12)
        self.intervention_active = False
        self.intervention_count = 0
        self.intervention_times = []
        self.blocked_commands = 0
        
        # Safety violation tracking
        self.safety_violations = {
            'tilt': [],
            'altitude': [],
            'velocity': [],
            'control': []
        }
    
    def check_control_safety(self, control, t):
        """
        Check if control command is safe
        Returns: (is_safe, violation_type)
        """
        # Check control magnitude (detect malicious large commands)
        control_magnitude = np.linalg.norm(control)
        if control_magnitude > self.max_control_magnitude:
            self.safety_violations['control'].append(t)
            return False, 'control_magnitude'
        
        # Check individual control limits
        if abs(control[0]) > 15:  # Thrust
            self.safety_violations['control'].append(t)
            return False, 'thrust_limit'
        if abs(control[1]) > 10 or abs(control[2]) > 10:  # Torques
            self.safety_violations['control'].append(t)
            return False, 'torque_limit'
        
        return True, None
    
    def check_state_safety(self, state, t):
        """
        Check if state violates safety constraints
        Returns: (is_safe, violation_type)
        """
        # Extract state components
        phi, theta = state[7], state[9]  # Roll, pitch
        z = state[5]  # Altitude
        u, v, w = state[0], state[2], state[4]  # Velocities
        
        # Check tilt angle
        tilt_magnitude = math.sqrt(phi**2 + theta**2)
        if tilt_magnitude > self.max_tilt:
            self.safety_violations['tilt'].append(t)
            return False, 'tilt'
        
        # Check altitude
        if z < self.min_altitude:
            self.safety_violations['altitude'].append(t)
            return False, 'altitude'
        
        # Check velocity
        velocity_magnitude = math.sqrt(u**2 + v**2 + w**2)
        if velocity_magnitude > self.max_velocity:
            self.safety_violations['velocity'].append(t)
            return False, 'velocity'
        
        return True, None
    
    def govern_control(self, t, state, desired_ref, proposed_control):
        """
        Monitor and override control commands if unsafe
        Reference Governor's main safety check
        """
        # Check if proposed control is safe
        control_safe, control_violation = self.check_control_safety(proposed_control, t)
        
        # Check if state is safe
        state_safe, state_violation = self.check_state_safety(state, t)
        
        # Compute Lyapunov function for overall system safety
        error = state - desired_ref
        V = error.T @ self.P @ error
        
        # Determine if intervention is needed
        needs_intervention = (not control_safe) or (not state_safe) or (V > self.safety_threshold)
        
        if needs_intervention:
            if not self.intervention_active:
                self.intervention_active = True
                self.intervention_count += 1
                self.intervention_times.append(t)
                reason = control_violation if not control_safe else (state_violation if not state_safe else "Lyapunov")
                print(f"\n[RG!] [REFERENCE GOVERNOR INTERVENTION #{self.intervention_count}] t={t:.2f}s")
                print(f"    Reason: {reason}")
                print(f"    Blocking malicious command, V={V:.2f}")
                
            self.blocked_commands += 1
            
            # Compute safe reference and control
            safe_ref = self.find_safe_reference(state, desired_ref)
            safe_control = self.compute_safe_control(state, safe_ref, proposed_control)
            self.governed_ref = safe_ref
            
            return safe_control, safe_ref, True
        else:
            # Control and state are safe
            if self.intervention_active:
                self.intervention_active = False
                print(f"[RG-OK] [REFERENCE GOVERNOR RELEASE #{self.intervention_count}] t={t:.2f}s")
                print(f"   System stabilized, returning to normal operation")
            
            self.governed_ref = desired_ref
            return proposed_control, desired_ref, False
    
    def find_safe_reference(self, current_state, desired_ref):
        """
        Find safe intermediate reference using binary search
        Maximizes progress toward desired while maintaining safety
        """
        safe_ref = self.governed_ref.copy()
        
        # Binary search for maximum safe step
        alpha_min, alpha_max = 0.0, 1.0
        for _ in range(10):
            alpha = (alpha_min + alpha_max) / 2.0
            candidate_ref = safe_ref + alpha * (desired_ref - safe_ref)
            
            # Test safety
            error = current_state - candidate_ref
            V_test = error.T @ self.P @ error
            
            # Check reference itself is safe
            if V_test < self.safety_threshold * 0.7 and candidate_ref[5] > self.min_altitude:
                alpha_min = alpha
            else:
                alpha_max = alpha
        
        return safe_ref + alpha_min * (desired_ref - safe_ref)
    
    def compute_safe_control(self, state, safe_ref, proposed_control):
        """
        Compute safe control that respects limits
        """
        # Use LQR control with safe reference
        safe_control = -self.K @ (state - safe_ref)
        
        # Clip to safe limits
        safe_control[0] = np.clip(safe_control[0], -10, 15)  # Thrust
        safe_control[1] = np.clip(safe_control[1], -8, 8)    # Roll torque
        safe_control[2] = np.clip(safe_control[2], -8, 8)    # Pitch torque
        safe_control[3] = np.clip(safe_control[3], -5, 5)    # Yaw torque
        
        return safe_control


# =====================================================
# ============   SIMULATION   =========================
# =====================================================

def run_replay_with_rg_simulation():
    """
    Scenario 3: Replay Attack with Reference Governor Defense
    """
    print("\n" + "="*80)
    print("SCENARIO 3: REPLAY ATTACK WITH REFERENCE GOVERNOR DEFENSE")
    print("="*80)
    print("Attack: Two-phase replay (record → replay + malicious control injection)")
    print("Defense: Reference Governor (safety-critical control override)")
    print("="*80)
    
    # Load controller
    K = load_matrix_K("mat/K.mat")
    
    # Simulation parameters
    Ts = 0.01
    T = 40
    tt = np.arange(0, T+Ts, Ts)
    Ns = tt.size
    n = 12
    
    # Initialize attack and defense
    attack = ReplayAttack(record_start=5.0, record_duration=5.0,
                         replay_start=15.0, replay_duration=10.0,
                         attack_intensity='aggressive')
    governor = SafetyReferenceGovernor(K, max_tilt_deg=25, min_altitude=0.3,
                                       max_velocity=12.0, max_control_magnitude=10.0)
    
    print(f"\n[ATTACK] {attack.name}")
    print(f"   Phase 1 (Recording): {attack.record_start}s - {attack.record_start + attack.record_duration}s")
    print(f"   Phase 2 (Replay+Inject): {attack.replay_start}s - {attack.replay_start + attack.replay_duration}s")
    print(f"   Intensity: {attack.attack_intensity}")
    print(f"\n[DEFENSE] Reference Governor")
    print(f"   Max tilt: {math.degrees(governor.max_tilt):.1f} degrees")
    print(f"   Min altitude: {governor.min_altitude}m")
    print(f"   Max control magnitude: {governor.max_control_magnitude}")
    
    # Storage arrays
    x = np.zeros(n)
    
    states = np.zeros((Ns, n))
    references = np.zeros((Ns, n))
    governed_refs = np.zeros((Ns, n))
    proposed_controls = np.zeros((Ns, 4))
    actual_controls = np.zeros((Ns, 4))
    safety_metrics = np.zeros((Ns, 4))  # tilt, altitude, velocity, control_mag
    governor_flags = np.zeros(Ns)
    recording_flags = np.zeros(Ns)
    replay_flags = np.zeros(Ns)
    
    # Initialize
    governor.governed_ref = np.zeros(n)
    
    print("\n" + "="*80)
    print("Starting simulation...")
    
    for j, t in enumerate(tt):
        # Generate desired reference (figure-8)
        x_ref = np.zeros(n)
        xr, yr, zr = figure8(t)
        x_ref[1], x_ref[3], x_ref[5] = xr, yr, zr
        
        # Compute normal LQR control based on current state
        normal_u = -K @ (x - x_ref)
        
        # Attack executes (may replay sensors and inject malicious control)
        x_sensor, proposed_u, phase = attack.update(t, x, normal_u)
        
        # REFERENCE GOVERNOR: Check safety and override if necessary
        safe_u, governed_ref, intervention = governor.govern_control(
            t, x, x_ref, proposed_u
        )
        
        # Apply safe control to plant
        x_dot = quadrotor_u(x, safe_u)
        x = x + x_dot * Ts
        
        # Compute safety metrics
        phi, theta = x[7], x[9]
        tilt = math.sqrt(phi**2 + theta**2)
        altitude = x[5]
        velocity = math.sqrt(x[0]**2 + x[2]**2 + x[4]**2)
        control_mag = np.linalg.norm(safe_u)
        
        # Store data
        states[j] = x
        references[j] = x_ref
        governed_refs[j] = governed_ref
        proposed_controls[j] = proposed_u
        actual_controls[j] = safe_u
        safety_metrics[j] = [math.degrees(tilt), altitude, velocity, control_mag]
        governor_flags[j] = 1 if intervention else 0
        recording_flags[j] = 1 if phase == "recording" else 0
        replay_flags[j] = 1 if phase == "replay" else 0
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Reference Governor Interventions: {governor.intervention_count}")
    print(f"Intervention Times: {[f'{t:.2f}s' for t in governor.intervention_times[:5]]}")
    if len(governor.intervention_times) > 5:
        print(f"                     ... and {len(governor.intervention_times)-5} more")
    print(f"Malicious Commands Blocked: {governor.blocked_commands}")
    print(f"\nSafety Violations Prevented:")
    print(f"  - Control magnitude: {len(set(governor.safety_violations['control']))}")
    print(f"  - Tilt: {len(set(governor.safety_violations['tilt']))}")
    print(f"  - Altitude: {len(set(governor.safety_violations['altitude']))}")
    print(f"  - Velocity: {len(set(governor.safety_violations['velocity']))}")
    print(f"Recorded Measurements: {len(attack.recorded_measurements)}")
    print("="*80 + "\n")
    
    # Plot results
    plot_replay_rg_scenario(tt, states, references, governed_refs,
                           proposed_controls, actual_controls, safety_metrics,
                           governor_flags, recording_flags, replay_flags,
                           attack, governor)


# =====================================================
# ============   PLOTTING   ===========================
# =====================================================

def plot_replay_rg_scenario(tt, states, refs, gov_refs, prop_controls, 
                            actual_controls, safety_metrics, gov_flags,
                            rec_flags, rep_flags, attack, governor):
    """Visualize replay attack with reference governor defense"""
    
    # Figure 1: 3D Trajectory
    fig1 = plt.figure(figsize=(12, 9))
    ax = fig1.add_subplot(111, projection='3d')
    
    # Separate phases (skip governor active states to show clean result)
    normal_idx, recording_idx, replay_idx, protected_idx = [], [], [], []
    for i in range(len(tt)):
        if gov_flags[i]:
            continue  # Skip during active intervention for cleaner viz
        if rec_flags[i]:
            recording_idx.append(i)
        elif rep_flags[i]:
            replay_idx.append(i)
        elif i > 0 and gov_flags[i-1]:
            protected_idx.append(i)
        else:
            normal_idx.append(i)
    
    if normal_idx:
        ax.scatter(states[normal_idx, 1], states[normal_idx, 3], states[normal_idx, 5],
                   c='blue', s=2, alpha=0.6, label='Normal')
    if recording_idx:
        ax.scatter(states[recording_idx, 1], states[recording_idx, 3], states[recording_idx, 5],
                   c='cyan', s=2, alpha=0.6, label='Recording Phase')
    if replay_idx:
        ax.scatter(states[replay_idx, 1], states[replay_idx, 3], states[replay_idx, 5],
                   c='red', s=3, alpha=0.7, label='Under Attack (RG Active)')
    if protected_idx:
        ax.scatter(states[protected_idx, 1], states[protected_idx, 3], states[protected_idx, 5],
                   c='green', s=5, alpha=0.8, label='RG Protected')
    
    ax.plot(refs[:, 1], refs[:, 3], refs[:, 5], 'k--',
            linewidth=2, alpha=0.4, label='Reference')
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title('3D Trajectory - Replay Attack with Reference Governor Defense',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-10, 15])
    ax.set_ylim([-15, 10])
    ax.set_zlim([-1, 8])
    plt.tight_layout()
    
    # Figure 2: Safety Metrics
    fig2, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Tilt Angle
    axes[0].plot(tt, safety_metrics[:, 0], 'b-', linewidth=1.5, label='Tilt Angle')
    axes[0].axhline(y=math.degrees(governor.max_tilt), color='r',
                    linestyle='--', linewidth=2, label=f'Safety Limit')
    axes[0].fill_between(tt, 0, safety_metrics[:, 0],
                         where=(rep_flags > 0), alpha=0.15, color='red')
    axes[0].fill_between(tt, 0, 100, where=(gov_flags > 0),
                         alpha=0.2, color='green', label='Governor Active')
    axes[0].set_ylabel('Tilt [°]', fontsize=11)
    axes[0].set_title('Safety Monitoring - Reference Governor Protection',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 50])
    
    # Altitude
    axes[1].plot(tt, safety_metrics[:, 1], 'b-', linewidth=1.5, label='Altitude')
    axes[1].axhline(y=governor.min_altitude, color='r',
                    linestyle='--', linewidth=2, label=f'Min Safe Alt')
    axes[1].fill_between(tt, 0, safety_metrics[:, 1],
                         where=(rep_flags > 0), alpha=0.15, color='red')
    axes[1].fill_between(tt, 0, 10, where=(gov_flags > 0),
                         alpha=0.2, color='green')
    axes[1].set_ylabel('Altitude [m]', fontsize=11)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.5, 6])
    
    # Velocity
    axes[2].plot(tt, safety_metrics[:, 2], 'b-', linewidth=1.5, label='Velocity')
    axes[2].axhline(y=governor.max_velocity, color='r',
                    linestyle='--', linewidth=2, label=f'Max Safe Vel')
    axes[2].fill_between(tt, 0, safety_metrics[:, 2],
                         where=(rep_flags > 0), alpha=0.15, color='red')
    axes[2].fill_between(tt, 0, 20, where=(gov_flags > 0),
                         alpha=0.2, color='green')
    axes[2].set_ylabel('Velocity [m/s]', fontsize=11)
    axes[2].legend(fontsize=9, loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 20])
    
    # Control Magnitude
    axes[3].plot(tt, safety_metrics[:, 3], 'purple', linewidth=1.5, label='Control Magnitude')
    axes[3].axhline(y=governor.max_control_magnitude, color='r',
                    linestyle='--', linewidth=2, label=f'Max Safe Control')
    axes[3].fill_between(tt, 0, safety_metrics[:, 3],
                         where=(rep_flags > 0), alpha=0.15, color='red', label='Replay Attack')
    axes[3].fill_between(tt, 0, 30, where=(gov_flags > 0),
                         alpha=0.2, color='green', label='Gov Active')
    axes[3].set_ylabel('Control Mag', fontsize=11)
    axes[3].set_xlabel('Time [s]', fontsize=11)
    axes[3].legend(fontsize=9, loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([0, 25])
    
    plt.tight_layout()
    
    # Figure 3: Control Comparison
    fig3, axes = plt.subplots(4, 1, figsize=(14, 10))
    control_labels = ['Thrust (F_z)', 'Roll Torque (τ_x)', 'Pitch Torque (τ_y)', 'Yaw Torque (τ_z)']
    
    for i in range(4):
        # Show proposed (malicious) vs actual (safe) control
        axes[i].plot(tt, prop_controls[:, i], 'r:', linewidth=2,
                     alpha=0.6, label='Proposed (Malicious)')
        axes[i].plot(tt, actual_controls[:, i], 'b-', linewidth=1.5,
                     label='Actual (RG Override)')
        axes[i].fill_between(tt, -20, 20, where=(rep_flags > 0),
                             alpha=0.15, color='red', label='Replay Period')
        axes[i].fill_between(tt, -20, 20, where=(gov_flags > 0),
                             alpha=0.2, color='green', label='Governor Active')
        axes[i].set_ylabel(control_labels[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8, loc='upper right')
        axes[i].set_ylim([-18, 18])
    
    axes[0].set_title('Control Signals - Reference Governor Override',
                      fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    run_replay_with_rg_simulation()