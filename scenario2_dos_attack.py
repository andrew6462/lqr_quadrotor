"""
SCENARIO 2: DENIAL-OF-SERVICE ATTACK WITH REDUNDANCY & GRACEFUL DEGRADATION

npm run scenario2

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import math
from collections import deque

# Import from existing code
from main_cps import load_matrix_K, quadrotor_u, figure8


# =====================================================
# ============   DoS ATTACK   =========================
# =====================================================

class DoS_Attack:
    """
    Denial of Service Attack
    Drops sensor packets and control commands
    """
    def __init__(self, start_time=12.0, duration=10.0, 
                 sensor_drop_rate=0.5, control_drop_rate=0.3):
        self.name = "DoS"
        self.start_time = start_time
        self.duration = duration
        self.sensor_drop_rate = sensor_drop_rate
        self.control_drop_rate = control_drop_rate
        
        # Track packet loss
        self.sensor_packets_dropped = 0
        self.control_packets_dropped = 0
        self.total_sensor_packets = 0
        self.total_control_packets = 0
        
        np.random.seed(42)
        
    def drop_sensor_packet(self, t, measurement):
        """
        Randomly drop sensor packets
        Returns: (measurement, was_dropped)
        """
        self.total_sensor_packets += 1
        
        if not self.is_active(t):
            return measurement, False
        
        # Probabilistic packet drop
        if np.random.random() < self.sensor_drop_rate:
            self.sensor_packets_dropped += 1
            return None, True  # Packet dropped
        
        return measurement, False
    
    def drop_control_packet(self, t, control):
        """
        Randomly drop control packets
        Returns: (control, was_dropped)
        """
        self.total_control_packets += 1
        
        if not self.is_active(t):
            return control, False
        
        # Probabilistic packet drop
        if np.random.random() < self.control_drop_rate:
            self.control_packets_dropped += 1
            return None, True
        
        return control, False
    
    def is_active(self, t):
        return self.start_time <= t <= self.start_time + self.duration
    
    def get_statistics(self):
        """Get packet loss statistics"""
        sensor_loss_pct = (self.sensor_packets_dropped / max(self.total_sensor_packets, 1)) * 100
        control_loss_pct = (self.control_packets_dropped / max(self.total_control_packets, 1)) * 100
        return {
            'sensor_dropped': self.sensor_packets_dropped,
            'sensor_total': self.total_sensor_packets,
            'sensor_loss_pct': sensor_loss_pct,
            'control_dropped': self.control_packets_dropped,
            'control_total': self.total_control_packets,
            'control_loss_pct': control_loss_pct
        }


# =====================================================
# ============   STATE ESTIMATOR   ====================
# =====================================================

class StateEstimator:
    """
    State estimator to handle missing sensor data
    Uses simple prediction when packets are dropped
    """
    def __init__(self):
        self.estimated_state = np.zeros(12)
        self.last_valid_measurement = np.zeros(12)
        self.last_valid_time = 0
        self.estimation_active = False
        self.estimation_count = 0
        
    def update(self, t, measurement, control, is_dropped, dt=0.01):
        """
        Update estimate based on measurement availability
        """
        if measurement is None or is_dropped:
            # Packet dropped - use prediction
            self.estimation_active = True
            self.estimation_count += 1
            
            # Simple prediction: integrate velocity
            self.estimated_state[1] += self.estimated_state[0] * dt  # x += u*dt
            self.estimated_state[3] += self.estimated_state[2] * dt  # y += v*dt
            self.estimated_state[5] += self.estimated_state[4] * dt  # z += w*dt
            
            # Decay velocity slightly (model drag)
            self.estimated_state[0] *= 0.99
            self.estimated_state[2] *= 0.99
            self.estimated_state[4] *= 0.99
            
            return self.estimated_state, True
        else:
            # Valid measurement received
            self.estimation_active = False
            self.last_valid_measurement = measurement.copy()
            self.last_valid_time = t
            self.estimated_state = measurement.copy()
            return measurement, False


# =====================================================
# ============   GRACEFUL DEGRADATION   ===============
# =====================================================

class GracefulDegradationController:
    """
    Adapts controller behavior during degraded mode
    Reduces aggressiveness when communication is poor
    """
    def __init__(self, K, degradation_factor=0.6):
        self.K = K
        self.nominal_K = K.copy()
        self.degradation_factor = degradation_factor
        
        self.degraded_mode = False
        self.packet_loss_window = deque(maxlen=50)
        self.degradation_trigger_threshold = 0.3
        
    def update_mode(self, sensor_dropped, control_dropped):
        """Update controller mode based on packet loss"""
        # Track packet loss in sliding window
        self.packet_loss_window.append(1 if (sensor_dropped or control_dropped) else 0)
        
        if len(self.packet_loss_window) >= 20:
            loss_rate = sum(self.packet_loss_window) / len(self.packet_loss_window)
            
            if loss_rate > self.degradation_trigger_threshold and not self.degraded_mode:
                self.degraded_mode = True
                self.K = self.nominal_K * self.degradation_factor
                print(f"   [Degraded Mode ACTIVE] - Reducing control gain")
                return True
            elif loss_rate < self.degradation_trigger_threshold * 0.5 and self.degraded_mode:
                self.degraded_mode = False
                self.K = self.nominal_K.copy()
                print(f"   [Degraded Mode OFF] - Restoring nominal control")
                return False
        
        return self.degraded_mode
    
    def compute_control(self, state, reference):
        """Compute control with current gains"""
        return -self.K @ (state - reference)


# =====================================================
# ============   SIMULATION   =========================
# =====================================================

def run_dos_attack_simulation():
    """
    Scenario 3: DoS Attack with Redundancy and Graceful Degradation
    """
    print("\n" + "="*80)
    print("SCENARIO 3: DoS ATTACK WITH REDUNDANCY & GRACEFUL DEGRADATION")
    print("="*80)
    print("Attack: Denial of Service (packet drops)")
    print("Defense: State Estimator + Graceful Degradation + Redundancy")
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
    attack = DoS_Attack(start_time=12.0, duration=10.0,
                        sensor_drop_rate=0.5, control_drop_rate=0.3)
    estimator = StateEstimator()
    controller = GracefulDegradationController(K, degradation_factor=0.6)
    
    print(f"[ATTACK] {attack.name}")
    print(f"   Start: {attack.start_time}s, Duration: {attack.duration}s")
    print(f"   Sensor drop rate: {attack.sensor_drop_rate*100:.0f}%")
    print(f"   Control drop rate: {attack.control_drop_rate*100:.0f}%")
    print(f"[DEFENSE] State Estimator + Graceful Degradation")
    print(f"   Degradation threshold: {controller.degradation_trigger_threshold*100:.0f}% loss")
    print(f"   Degradation factor: {controller.degradation_factor}\n")
    
    # Storage arrays
    x = np.zeros(n)
    x_last_control = None  # Last successful control
    
    states = np.zeros((Ns, n))
    estimated_states = np.zeros((Ns, n))
    references = np.zeros((Ns, n))
    controls = np.zeros((Ns, 4))
    sensor_drop_flags = np.zeros(Ns)
    control_drop_flags = np.zeros(Ns)
    estimation_flags = np.zeros(Ns)
    degradation_flags = np.zeros(Ns)
    attack_flags = np.zeros(Ns)
    packet_loss_rate = np.zeros(Ns)
    
    # Initialize estimator
    estimator.estimated_state = x.copy()
    
    print("Starting simulation...\n")
    
    for j, t in enumerate(tt):
        # Generate reference (figure-8)
        x_ref = np.zeros(n)
        xr, yr, zr = figure8(t)
        x_ref[1], x_ref[3], x_ref[5] = xr, yr, zr
        
        # Sensor packet transmission (may be dropped)
        x_measured, sensor_dropped = attack.drop_sensor_packet(t, x)
        
        # State estimation (handles missing measurements)
        x_estimate, estimating = estimator.update(t, x_measured if not sensor_dropped else None,
                                                   x_last_control, sensor_dropped, Ts)
        
        # Compute control (uses estimate when sensor dropped)
        cu = controller.compute_control(x_estimate, x_ref)
        
        # Control packet transmission (may be dropped)
        cu_delivered, control_dropped = attack.drop_control_packet(t, cu)
        
        # Update degradation mode
        degraded = controller.update_mode(sensor_dropped, control_dropped)
        
        # Apply control to plant
        if cu_delivered is not None:
            x_last_control = cu_delivered
            x_dot = quadrotor_u(x, cu_delivered)
            x = x + x_dot * Ts
        else:
            # Control dropped - plant evolves without new command
            # Use last successful control (zero-order hold)
            if x_last_control is not None:
                x_dot = quadrotor_u(x, x_last_control)
            else:
                x_dot = quadrotor_u(x, np.zeros(4))
            x = x + x_dot * Ts
        
        # Compute packet loss rate
        if j > 0:
            recent_drops = sum(sensor_drop_flags[max(0,j-50):j]) + sum(control_drop_flags[max(0,j-50):j])
            recent_total = min(j, 50) * 2
            packet_loss_rate[j] = recent_drops / recent_total if recent_total > 0 else 0
        
        # Store data
        states[j] = x
        estimated_states[j] = x_estimate
        references[j] = x_ref
        controls[j] = cu_delivered if cu_delivered is not None else (x_last_control if x_last_control is not None else np.zeros(4))
        sensor_drop_flags[j] = 1 if sensor_dropped else 0
        control_drop_flags[j] = 1 if control_dropped else 0
        estimation_flags[j] = 1 if estimating else 0
        degradation_flags[j] = 1 if degraded else 0
        attack_flags[j] = 1 if attack.is_active(t) else 0
    
    # Get statistics
    stats = attack.get_statistics()
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Packet Loss Statistics:")
    print(f"  Sensor packets dropped: {stats['sensor_dropped']}/{stats['sensor_total']} ({stats['sensor_loss_pct']:.1f}%)")
    print(f"  Control packets dropped: {stats['control_dropped']}/{stats['control_total']} ({stats['control_loss_pct']:.1f}%)")
    print(f"  Estimation activations: {estimator.estimation_count}")
    print(f"  Degraded mode: {'Active during attack' if any(degradation_flags) else 'Not triggered'}")
    print("="*80 + "\n")
    
    # Plot results
    plot_dos_scenario(tt, states, estimated_states, references, controls,
                      sensor_drop_flags, control_drop_flags, estimation_flags,
                      degradation_flags, attack_flags, packet_loss_rate,
                      attack, controller)


# =====================================================
# ============   PLOTTING   ===========================
# =====================================================

def plot_dos_scenario(tt, states, est_states, refs, controls,
                      sensor_drops, control_drops, est_flags, deg_flags,
                      attack_flags, loss_rate, attack, controller):
    """Visualize DoS attack scenario"""
    
    # Figure 1: 3D Trajectory
    fig1 = plt.figure(figsize=(12, 9))
    ax = fig1.add_subplot(111, projection='3d')
    
    # Separate phases
    normal_idx, attack_idx, degraded_idx = [], [], []
    for i in range(len(tt)):
        if attack_flags[i]:
            if deg_flags[i]:
                degraded_idx.append(i)
            else:
                attack_idx.append(i)
        else:
            normal_idx.append(i)
    
    if normal_idx:
        ax.scatter(states[normal_idx, 1], states[normal_idx, 3], states[normal_idx, 5],
                   c='blue', s=2, alpha=0.6, label='Normal')
    if attack_idx:
        ax.scatter(states[attack_idx, 1], states[attack_idx, 3], states[attack_idx, 5],
                   c='orange', s=2, alpha=0.6, label='Under DoS')
    if degraded_idx:
        ax.scatter(states[degraded_idx, 1], states[degraded_idx, 3], states[degraded_idx, 5],
                   c='red', s=2, alpha=0.7, label='Degraded Mode')
    
    ax.plot(refs[:, 1], refs[:, 3], refs[:, 5], 'k--',
            linewidth=2, alpha=0.4, label='Reference')
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title('3D Trajectory - DoS Attack with Graceful Degradation',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-10, 15])
    ax.set_ylim([-15, 10])
    ax.set_zlim([-1, 8])
    plt.tight_layout()
    
    # Figure 2: Position Tracking with Estimation
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9))
    positions = ['X', 'Y', 'Z']
    indices = [1, 3, 5]
    
    for i, (pos, idx) in enumerate(zip(positions, indices)):
        axes[i].plot(tt, states[:, idx], 'b-', linewidth=1.5, label='Actual')
        axes[i].plot(tt, est_states[:, idx], 'g:', linewidth=2, alpha=0.7, label='Estimated')
        axes[i].plot(tt, refs[:, idx], 'k--', linewidth=1.5, alpha=0.6, label='Reference')
        
        axes[i].fill_between(tt, -20, 20, where=(attack_flags > 0),
                             alpha=0.15, color='red', label='Attack Period')
        axes[i].fill_between(tt, -20, 20, where=(est_flags > 0),
                             alpha=0.2, color='yellow', label='Estimating')
        
        axes[i].set_ylabel(f'{pos} [m]', fontsize=11)
        axes[i].legend(fontsize=9, loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    axes[0].set_title('Position Tracking - DoS with State Estimation',
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()
    
    # Figure 3: Communication and Degradation Status
    fig3, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Packet drops
    axes[0].scatter(tt, sensor_drops, c='red', s=10, alpha=0.5, label='Sensor Drops')
    axes[0].scatter(tt, control_drops*0.5, c='orange', s=10, alpha=0.5, label='Control Drops')
    axes[0].fill_between(tt, 0, 1, where=(attack_flags > 0),
                         alpha=0.15, color='red', label='Attack Period')
    axes[0].set_ylabel('Packet Drops', fontsize=11)
    axes[0].set_title('Communication Channel Status',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Packet loss rate
    axes[1].plot(tt, loss_rate * 100, 'purple', linewidth=1.5)
    axes[1].axhline(y=controller.degradation_trigger_threshold*100, color='r',
                    linestyle='--', linewidth=2, label=f'Degradation Threshold')
    axes[1].fill_between(tt, 0, loss_rate*100, where=(attack_flags > 0),
                         alpha=0.15, color='red')
    axes[1].set_ylabel('Loss Rate [%]', fontsize=11)
    axes[1].set_title('Packet Loss Rate (50-sample window)',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    # Estimation activity
    axes[2].fill_between(tt, 0, est_flags, alpha=0.4, color='yellow')
    axes[2].plot(tt, est_flags, color='orange', linewidth=2, label='Estimator Active')
    axes[2].fill_between(tt, 0, 1, where=(attack_flags > 0),
                         alpha=0.15, color='red', label='Attack Period')
    axes[2].set_ylabel('Estimation', fontsize=11)
    axes[2].set_title('State Estimator Activity',
                      fontsize=12, fontweight='bold')
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # Degraded mode
    axes[3].fill_between(tt, 0, deg_flags, alpha=0.4, color='red')
    axes[3].plot(tt, deg_flags, 'r-', linewidth=2, label='Degraded Mode')
    axes[3].fill_between(tt, 0, 1, where=(attack_flags > 0),
                         alpha=0.15, color='red', label='Attack Period')
    axes[3].set_ylabel('Degraded Mode', fontsize=11)
    axes[3].set_xlabel('Time [s]', fontsize=11)
    axes[3].set_title('Graceful Degradation Status',
                      fontsize=12, fontweight='bold')
    axes[3].set_ylim([-0.1, 1.1])
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_dos_attack_simulation()