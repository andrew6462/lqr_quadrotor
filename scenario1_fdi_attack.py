"""
SCENARIO 1: GPS SPOOFING WITH ANOMALY DETECTION + SOFTWARE REJUVENATION

npm run scenario1

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import math
from collections import deque

# Import from existing code
from main_cps import load_matrix_K, quadrotor_u, figure8


# =====================================================
# ============   GPS SPOOFING ATTACK   ================
# =====================================================

class GPS_Spoofing_Attack:
    """
    False Data Injection on GPS Position Sensors
    Gradually injects false position data
    """
    def __init__(self, start_time=12.0, duration=12.0, magnitude=3.0, pattern='circular'):
        self.name = "GPS_Spoofing"
        self.start_time = start_time
        self.duration = duration
        self.magnitude = magnitude
        self.pattern = pattern  # 'circular', 'drift', or 'jump'
        
    def compute_injection(self, t):
        """Generate false GPS data injection"""
        attack_end = self.start_time + self.duration
        if not (self.start_time <= t <= attack_end):
            return np.zeros(12)
        
        t_attack = t - self.start_time
        
        # Gradual ramp-up to avoid immediate detection
        ramp = min(t_attack / 3.0, 1.0)
        amplitude = self.magnitude * ramp
        
        injection = np.zeros(12)
        
        if self.pattern == 'circular':
            # Circular GPS error (makes drone think it's displaced)
            injection[1] = amplitude * math.sin(2 * math.pi * t / 4)  # x position
            injection[3] = amplitude * math.cos(2 * math.pi * t / 4)  # y position
            injection[5] = amplitude * 0.3 * math.sin(2 * math.pi * t / 6)  # z position
            
        elif self.pattern == 'drift':
            # Linear drift (constant bias)
            injection[1] = amplitude * t_attack / 5.0  # x drifts
            injection[3] = amplitude * 0.5 * t_attack / 5.0  # y drifts
            injection[5] = -amplitude * 0.2  # z drops slightly
            
        elif self.pattern == 'jump':
            # Sudden jumps (step changes)
            if int(t_attack) % 3 == 0:  # Every 3 seconds
                injection[1] = amplitude * (1 if int(t_attack) % 6 == 0 else -1)
                injection[3] = amplitude * (1 if int(t_attack) % 6 == 3 else -1)
        
        # Only inject on position measurements
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        return injection * mask
    
    def is_active(self, t):
        return self.start_time <= t <= self.start_time + self.duration


# =====================================================
# ============   ANOMALY DETECTOR   ===================
# =====================================================

class GPS_Anomaly_Detector:
    """
    Chi-square based anomaly detector for GPS spoofing
    Monitors residual between predicted and measured positions
    """
    def __init__(self, threshold=4.0, window_size=15):
        self.threshold = threshold
        self.window_size = window_size
        self.residual_history = deque(maxlen=window_size)
        self.chi_square_history = deque(maxlen=window_size)
        
        self.attack_detected = False
        self.detection_count = 0
        self.detection_times = []
        
        # For prediction model (simple dynamics)
        self.last_prediction = None
        
    def predict_position(self, state, control, dt=0.01):
        """Simple forward prediction of next position"""
        # Use simple integration: x_next ≈ x + v*dt
        predicted = state.copy()
        predicted[1] += state[0] * dt  # x += u*dt
        predicted[3] += state[2] * dt  # y += v*dt
        predicted[5] += state[4] * dt  # z += w*dt
        return predicted
    
    def compute_residual(self, measured, predicted):
        """Compute residual for position states only"""
        pos_measured = np.array([measured[1], measured[3], measured[5]])
        pos_predicted = np.array([predicted[1], predicted[3], predicted[5]])
        residual = np.linalg.norm(pos_measured - pos_predicted)
        return residual
    
    def update(self, t, measured_state, predicted_state):
        """Update detector with new measurement"""
        residual = self.compute_residual(measured_state, predicted_state)
        self.residual_history.append(residual)
        
        # Compute chi-square statistic
        if len(self.residual_history) >= 5:
            recent = list(self.residual_history)[-5:]
            mean_residual = np.mean(recent)
            std_residual = np.std(recent) + 1e-6
            chi_square = (mean_residual / std_residual) ** 2
            self.chi_square_history.append(chi_square)
            
            # Detection logic with hysteresis
            if chi_square > self.threshold and not self.attack_detected:
                self.attack_detected = True
                self.detection_count += 1
                self.detection_times.append(t)
                print(f"[!] [GPS SPOOFING DETECTED #{self.detection_count}] t={t:.2f}s, chi^2={chi_square:.2f}")
                return True
            elif chi_square < self.threshold * 0.4:
                if self.attack_detected:
                    print(f"   [Attack subsided] t={t:.2f}s")
                self.attack_detected = False
        
        return self.attack_detected
    
    def get_current_chi_square(self):
        if len(self.chi_square_history) > 0:
            return self.chi_square_history[-1]
        return 0.0


# =====================================================
# ============   SOFTWARE REJUVENATION   ==============
# =====================================================

class SoftwareRejuvenation:
    """
    Restores drone state to clean values when attack detected
    """
    def __init__(self, recovery_time=0.8, buffer_size=25):
        self.recovery_time = recovery_time
        self.clean_state_buffer = deque(maxlen=buffer_size)
        
        self.sr_active = False
        self.sr_start_time = None
        self.sr_count = 0
        self.sr_times = []
        
    def update_clean_buffer(self, state):
        """Store clean states during normal operation"""
        self.clean_state_buffer.append(state.copy())
    
    def trigger_rejuvenation(self, t, corrupted_state, detector_alarm):
        """
        Trigger SR when detector raises alarm
        Gradually restore state to clean buffer average
        """
        # Trigger SR on detector alarm
        if detector_alarm and not self.sr_active:
            self.sr_active = True
            self.sr_start_time = t
            self.sr_count += 1
            self.sr_times.append(t)
            print(f"[SR] [SOFTWARE REJUVENATION #{self.sr_count}] t={t:.2f}s - Restoring state")
        
        # During SR - restore state
        if self.sr_active:
            elapsed = t - self.sr_start_time
            
            if elapsed >= self.recovery_time:
                self.sr_active = False
                print(f"[OK] [SR COMPLETE #{self.sr_count}] t={t:.2f}s - State restored")
                # Instant restoration to clean average
                if len(self.clean_state_buffer) > 0:
                    clean_states = np.array(list(self.clean_state_buffer))
                    return np.mean(clean_states, axis=0)
            else:
                # Gradual restoration using weighted average
                if len(self.clean_state_buffer) > 0:
                    clean_states = np.array(list(self.clean_state_buffer))
                    target = np.mean(clean_states, axis=0)
                    alpha = elapsed / self.recovery_time
                    restored = (1 - alpha) * corrupted_state + alpha * target
                    return restored
        
        return corrupted_state


# =====================================================
# ============   SIMULATION   =========================
# =====================================================

def run_gps_spoofing_simulation():
    """
    Scenario 2: GPS Spoofing with Anomaly Detection + SR
    """
    print("\n" + "="*80)
    print("SCENARIO 2: GPS SPOOFING WITH ANOMALY DETECTION + SOFTWARE REJUVENATION")
    print("="*80)
    print("Attack: False Data Injection (GPS position sensors)")
    print("Defense: Chi-square Anomaly Detector + Software Rejuvenation")
    print("="*80 + "\n")
    
    # Load controller
    K = load_matrix_K("mat/K.mat")
    
    # Simulation parameters
    Ts = 0.01
    T = 40
    tt = np.arange(0, T+Ts, Ts)
    Ns = tt.size
    n = 12
    
    # Initialize attack and defense
    attack = GPS_Spoofing_Attack(start_time=12.0, duration=12.0, 
                                  magnitude=3.0, pattern='circular')
    detector = GPS_Anomaly_Detector(threshold=4.0, window_size=15)
    sr_system = SoftwareRejuvenation(recovery_time=0.8, buffer_size=25)
    
    print(f"[ATTACK] {attack.name} ({attack.pattern} pattern)")
    print(f"   Start: {attack.start_time}s, Duration: {attack.duration}s")
    print(f"   Magnitude: {attack.magnitude}m")
    print(f"[DEFENSE] Anomaly Detector + Software Rejuvenation")
    print(f"   Detection threshold: chi^2 > {detector.threshold}")
    print(f"   Recovery time: {sr_system.recovery_time}s\n")
    
    # Storage arrays
    x = np.zeros(n)
    x_predicted = np.zeros(n)
    
    states = np.zeros((Ns, n))
    references = np.zeros((Ns, n))
    injections = np.zeros((Ns, n))
    residuals = np.zeros(Ns)
    chi_squares = np.zeros(Ns)
    detector_flags = np.zeros(Ns)
    sr_flags = np.zeros(Ns)
    attack_flags = np.zeros(Ns)
    
    print("Starting simulation...\n")
    
    for j, t in enumerate(tt):
        # Generate reference (figure-8)
        x_ref = np.zeros(n)
        xr, yr, zr = figure8(t)
        x_ref[1], x_ref[3], x_ref[5] = xr, yr, zr
        
        # Compute GPS injection
        gps_injection = attack.compute_injection(t)
        
        # Corrupted measurement (GPS spoofed)
        x_measured = x + gps_injection
        
        # Predict next state for detection
        x_predicted = detector.predict_position(x_predicted, 
                                                 -K @ (x_measured - x_ref), Ts)
        
        # Anomaly detection
        attack_detected = detector.update(t, x_measured, x_predicted)
        
        # Software rejuvenation (restore state if attack detected)
        x = sr_system.trigger_rejuvenation(t, x_measured, attack_detected)
        
        # Update clean buffer when not under attack
        if not attack.is_active(t) and not sr_system.sr_active:
            sr_system.update_clean_buffer(x)
        
        # Controller uses current state (possibly restored)
        cu = -K @ (x - x_ref)
        
        # Apply to plant
        x_dot = quadrotor_u(x, cu)
        x = x + x_dot * Ts
        
        # Store data
        states[j] = x
        references[j] = x_ref
        injections[j] = gps_injection
        residuals[j] = detector.compute_residual(x_measured, x_predicted)
        chi_squares[j] = detector.get_current_chi_square()
        detector_flags[j] = 1 if attack_detected else 0
        sr_flags[j] = 1 if sr_system.sr_active else 0
        attack_flags[j] = 1 if attack.is_active(t) else 0
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Detections: {detector.detection_count}")
    print(f"Detection Times: {[f'{t:.2f}s' for t in detector.detection_times]}")
    print(f"SR Activations: {sr_system.sr_count}")
    print(f"SR Times: {[f'{t:.2f}s' for t in sr_system.sr_times]}")
    print("="*80 + "\n")
    
    # Plot results
    plot_gps_spoofing_scenario(tt, states, references, injections,
                                residuals, chi_squares, detector_flags,
                                sr_flags, attack_flags, attack, detector)


# =====================================================
# ============   PLOTTING   ===========================
# =====================================================

def plot_gps_spoofing_scenario(tt, states, refs, injections, residuals, 
                                chi_squares, det_flags, sr_flags, attack_flags,
                                attack, detector):
    """Visualize GPS spoofing scenario"""
    
    # Figure 1: 3D Trajectory
    fig1 = plt.figure(figsize=(12, 9))
    ax = fig1.add_subplot(111, projection='3d')
    
    # Separate phases
    normal_idx, attack_idx, recovery_idx = [], [], []
    for i in range(len(tt)):
        if sr_flags[i]:
            continue
        if attack_flags[i]:
            attack_idx.append(i)
        elif i > 0 and sr_flags[i-1]:
            recovery_idx.append(i)
        else:
            normal_idx.append(i)
    
    if normal_idx:
        ax.scatter(states[normal_idx, 1], states[normal_idx, 3], states[normal_idx, 5],
                   c='blue', s=2, alpha=0.6, label='Normal Flight')
    if attack_idx:
        ax.scatter(states[attack_idx, 1], states[attack_idx, 3], states[attack_idx, 5],
                   c='red', s=2, alpha=0.6, label='GPS Spoofed')
    if recovery_idx:
        ax.scatter(states[recovery_idx, 1], states[recovery_idx, 3], states[recovery_idx, 5],
                   c='green', s=5, alpha=0.8, label='After SR')
    
    ax.plot(refs[:, 1], refs[:, 3], refs[:, 5], 'k--', 
            linewidth=2, alpha=0.4, label='Reference')
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title('3D Trajectory - GPS Spoofing Attack', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-15, 20])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-2, 10])
    plt.tight_layout()
    
    # Figure 2: Position Tracking
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9))
    positions = ['X', 'Y', 'Z']
    indices = [1, 3, 5]
    
    for i, (pos, idx) in enumerate(zip(positions, indices)):
        axes[i].plot(tt, states[:, idx], 'b-', linewidth=1.5, label='Actual')
        axes[i].plot(tt, refs[:, idx], 'k--', linewidth=1.5, alpha=0.6, label='Reference')
        
        # Show GPS injection magnitude
        axes[i].plot(tt, refs[:, idx] + injections[:, idx], 'r:', 
                     linewidth=2, alpha=0.5, label='Spoofed GPS')
        
        axes[i].fill_between(tt, -20, 20, where=(attack_flags > 0),
                             alpha=0.15, color='red', label='Attack Period')
        axes[i].fill_between(tt, -20, 20, where=(sr_flags > 0),
                             alpha=0.25, color='green', label='SR Active')
        
        axes[i].set_ylabel(f'{pos} [m]', fontsize=11)
        axes[i].legend(fontsize=9, loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    axes[0].set_title('Position Tracking - GPS Spoofing Effects', 
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()
    
    # Figure 3: Detection and Rejuvenation
    fig3, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    # Residuals
    axes[0].plot(tt, residuals, 'b-', linewidth=1.5)
    axes[0].fill_between(tt, 0, residuals, where=(det_flags > 0),
                         alpha=0.3, color='red', label='Detection Active')
    axes[0].fill_between(tt, 0, max(residuals)*1.1, where=(attack_flags > 0),
                         alpha=0.1, color='red')
    axes[0].set_ylabel('Residual [m]', fontsize=11)
    axes[0].set_title('Anomaly Detection - Residual Monitoring', 
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Chi-square statistic
    axes[1].plot(tt, chi_squares, 'purple', linewidth=1.5)
    axes[1].axhline(y=detector.threshold, color='r', linestyle='--',
                    linewidth=2, label=f'Threshold={detector.threshold}')
    axes[1].fill_between(tt, 0, chi_squares, where=(det_flags > 0),
                         alpha=0.3, color='red')
    axes[1].set_ylabel('χ² Statistic', fontsize=11)
    axes[1].set_title('Chi-Square Test Statistic', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, max(chi_squares)*1.2 if max(chi_squares) > 0 else 10])
    
    # Software Rejuvenation
    axes[2].fill_between(tt, 0, sr_flags, alpha=0.4, color='green')
    axes[2].plot(tt, sr_flags, 'g-', linewidth=2, label='SR Active')
    axes[2].fill_between(tt, 0, 1, where=(attack_flags > 0),
                         alpha=0.15, color='red', label='Attack Period')
    axes[2].set_ylabel('SR Status', fontsize=11)
    axes[2].set_xlabel('Time [s]', fontsize=11)
    axes[2].set_title('Software Rejuvenation Activity', fontsize=12, fontweight='bold')
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_gps_spoofing_simulation()