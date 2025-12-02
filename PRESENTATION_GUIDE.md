# CPS Drone Cyber Defense - Presentation Guide

## Overview

This project demonstrates software rejuvenation techniques to defend against cyber attacks on a UAV quadrotor system. You'll show 4 demos: 1 baseline + 3 attack scenarios with defenses.

## Key Improvements in This Version

### Improved Multithreaded Architecture (5 Threads)
- **Ground Station** (50 Hz): Trajectory generation
- **Sensor** (100 Hz): State measurement + noise injection
- **Controller** (250 Hz): LQR control law computation
- **Actuator** (500 Hz): Control saturation
- **Simulator** (100 Hz): Nonlinear dynamics integration

### Enhanced Features
- ✅ Proper trajectory derivatives (velocity references included)
- ✅ Default sensor noise σ=0.05 for realistic operation
- ✅ Separate 2D and 3D trajectory visualizations
- ✅ Detailed tracking performance metrics
- ✅ Improved control gains for better tracking
- ✅ Better queue synchronization between threads

---

## Demo 1: BASELINE - Normal Operation

### Command:
```bash
npm test
```

### What It Does:
- Runs improved multithreaded UAV simulation with **5 threads**:
  - **Ground Station**: Generates trajectory references
  - **Sensor**: Measures state with realistic noise
  - **Controller**: Computes LQR control law
  - **Actuator**: Applies control saturation
  - **Simulator**: Integrates nonlinear quadrotor dynamics
- Executes figure-8 trajectory for 9 seconds (exactly 1 complete cycle)
- Sensor noise (σ=0.05) simulates realistic measurements
- Real-time position logging every second shows the drone's path

### What to Say:
1. **"This is our baseline - multithreaded architecture with proper separation of concerns"**
   - Point out the 5 threads starting in console: `[Simulator]`, `[Sensor]`, `[GroundStation]`, `[Controller]`, `[Actuator]`
   - Architecture mirrors real drone systems with separate modules
   - Console shows: `[GroundStation] Started - RG=False` (no Reference Governor in baseline)

2. **"Notice the real-time position logging every second"**
   - Console displays: `[t=1.0s] Position: X=0.123m, Y=0.456m, Z=0.500m`
   - Shows the drone's path as it flies the figure-8
   - Helps visualize the trajectory in real-time

3. **"We have sensor noise to simulate real-world conditions"**
   - Noise σ=0.05 on all 12 state measurements (positions, velocities, angles)
   - This creates realistic imperfections you'd see in actual drone flight
   - The slight deviation from the perfect reference line is ONLY due to sensor noise
   - Controller handles noisy data robustly using LQR gains from the professor's design

4. **"Look at the figure-8 trajectory plots"**
   - **Figure 1 (2D)**: Clean figure-8 shape in XY plane
     - Red dashed line = perfect reference trajectory
     - Blue solid line = actual flight path
     - The actual path closely follows the reference - any deviation is from sensor noise
     - Shows start (green circle) and end (red X) markers
   - **Figure 2 (3D)**: Full 3D trajectory with altitude control
     - Demonstrates smooth altitude hold at 0.5m throughout the figure-8
   - **Figure 3**: Position tracking over time (X, Y, Z separately)
     - Shows how closely the drone follows the reference in each axis
   - **Figure 4**: Control inputs (thrust, roll/pitch/yaw torques)
     - All controls stay well within safety limits

5. **"Performance metrics show excellent tracking despite sensor noise"**
   - XY Tracking RMS Error: typically 0.10-0.15m
   - Max Error: typically < 0.35m
   - Altitude RMS Error: typically < 0.05m
   - The drone follows the perfect figure-8 reference very closely
   - Small deviations are entirely due to realistic sensor noise (σ=0.05)
   - This demonstrates our baseline system works well even with noisy measurements
   - This is our baseline to compare attacks against

### Key Points:
- ✅ 5-thread multithreaded architecture
- ✅ NO Reference Governor in baseline (clean tracking demonstration)
- ✅ Sensor noise (σ=0.05) creates realistic flight imperfections
- ✅ Proper trajectory derivatives (positions + velocities)
- ✅ Real-time position logging every second
- ✅ Separate 2D and 3D visualization plots
- ✅ Excellent trajectory tracking - deviations are ONLY from sensor noise
- ✅ Exactly 1 complete figure-8 cycle (9 seconds total)

---

## Demo 2: SCENARIO 1 - GPS Spoofing Attack

### Command:
```bash
npm run scenario1
```

### What It Does:
- **ATTACK:** GPS position sensors injected with false data (circular pattern)
- Attack starts at t=3.0s, runs for 2.0 seconds (ends at t=5.0s)
- Moderate magnitude (1.0m) spoofing with circular pattern
- Total simulation: 12 seconds (full figure-8 with attack and recovery)
- **DEFENSE:** Chi-square anomaly detection (threshold=20.0) + Software Rejuvenation (SR)

### What to Say:
1. **"GPS spoofing is a False Data Injection attack"**
   - Attacker injects fake GPS coordinates in a circular pattern
   - Makes drone think it's in wrong location
   - Could cause crashes or navigation to wrong area
   - Attack magnitude: 1.0m displacement

2. **"Attack starts at 3 seconds - watch the trajectory diverge in the 3D plot"**
   - First 3s: normal figure-8 (blue path in Figure 1)
   - t=3-5s: trajectory diverges under attack (orange path)
   - t=5-12s: recovery phase after SR activations (green path)
   - Without defense, drone would continue flying off-course

3. **"Our chi-square anomaly detector catches the spoofed GPS data"**
   - Uses chi-square statistical test with threshold=20.0
   - Compares measured position vs predicted position from dynamics model
   - When chi² > threshold, attack is detected
   - Console shows: `[!] [GPS SPOOFING DETECTED]`
   - Detection typically occurs within 0.1-0.2 seconds of attack start

   **How Detection Works (Simple Explanation):**
   - **Step 1**: Our physics model predicts where the drone SHOULD be based on the control commands
   - **Step 2**: GPS sensor tells us where the drone actually IS
   - **Step 3**: We calculate the difference (residual) between predicted vs measured position
   - **Step 4**: Chi-square test checks if this difference is "too big" (beyond normal sensor noise)
   - **Step 5**: If difference exceeds threshold (chi² > 20.0), we know something is wrong = ATTACK DETECTED
   - **Think of it like**: If you tell your friend to walk 10 steps forward, but GPS says they moved 50 steps, you know the GPS is lying!

4. **"Software Rejuvenation restores the clean state"**
   - Console shows: `[SR] [SOFTWARE REJUVENATION]`
   - Restores state from clean buffer (pre-attack sensor data)
   - Recovery time: 0.8 seconds per SR activation
   - Green path in Figure 1 shows trajectory returning to reference
   - Multiple SR activations may occur during attack period

5. **"Look at the 3D trajectory and detection plots"**
   - **Figure 1 (3D Trajectory)**: Continuous path with color coding
     - Blue = Normal operation (t=0-3s)
     - Orange = Under attack (t=3-5s)
     - Green = Recovery after SR (t=5-12s)
   - **Figure 3 (Detection & SR Timeline)**:
     - Top: Position residual magnitude spikes during attack
     - Middle: Chi-square statistic exceeds threshold (red dashed line at 20.0)
     - Bottom: SR activation timeline (green bars show when SR triggers)

### Interactive Demo - Modify Attack Strength:
**Show defense works for different attack magnitudes:**

Edit `scenario1_fdi_attack.py` around line 340 in the GPS_SpoofingSensor initialization:
```python
# Current default (moderate attack)
sensor = GPS_SpoofingSensor(
    attack_start=3.0,
    attack_duration=2.0,
    magnitude=1.0,  # Current: 1.0m displacement
    pattern='circular',
)

# Weak attack (easy to handle)
magnitude=0.5  # Easier to detect and recover

# Strong attack (stress test)
magnitude=2.0  # More aggressive, tests SR robustness
```

Run again and show:
- Weak attack (0.5m): Quick detection, minimal trajectory deviation
- Strong attack (2.0m): More SR activations, larger recovery phase
- Chi-square detector works across all magnitudes

**Try different attack patterns:**
```python
pattern='circular'  # Circular GPS error (default - smooth deviation)
pattern='drift'     # Linear drift attack (gradual offset)
pattern='jump'      # Sudden position jumps (aggressive)
```

### Key Points:
- ❌ Attack: GPS False Data Injection
- ✅ Defense: Anomaly Detection (Chi-square test)
- ✅ Defense: Software Rejuvenation (state restoration)
- 📊 Show detection count and SR activation count at end

---

## Demo 3: SCENARIO 2 - DoS Attack

### Command:
```bash
npm run scenario2
```

### What It Does:
- **ATTACK:** Denial of Service - drops sensor packets randomly
- Attack starts at t=4s, runs for 2s (ends at t=6s)
- 60% packet drop rate during attack
- Total simulation: 10 seconds (quick demo for presentation)
- **DEFENSE:** State Estimator + Graceful Degradation (Software Rejuvenation)

### What to Say:
1. **"DoS attacks drop communication packets"**
   - Attacker floods network or jams communication
   - 60% of sensor data doesn't reach controller during attack (t=4-6s)
   - Could cause complete loss of control without defense

2. **"Watch packet drops in the console"**
   - Console shows: `[ATTACK] DoS Attack - Packet DROPPED at t=X.XXs`
   - Orange sections in plots show attack period
   - Without defense, control becomes erratic and dangerous

3. **"State Estimator predicts missing data (our SR defense)"**
   - When packet dropped, use physics model to estimate state
   - Uses constant velocity model: predicts where drone should be
   - Continues control even without sensor data
   - This is a form of **graceful degradation** - a software rejuvenation technique

   **How State Estimator Works (Simple Explanation):**
   - **Normal Operation**: Use real sensor measurements from GPS/IMU
   - **Packet Dropped**: Instead of crashing or using stale data, switch to estimation mode
   - **Prediction**: Use physics (constant velocity model) to predict where drone is now
     - New position = last known position + velocity × time
     - Apply slight drag to velocities for realism
   - **Recovery**: When sensor data returns, immediately switch back to real measurements
   - **Think of it like**: If your GPS freezes while driving, you estimate your position by assuming you kept moving at the same speed - better than stopping or using old location!

4. **"Graceful Degradation maintains stability"**
   - Console shows: `[DEFENSE] Using estimated state`
   - Controller adapts to degraded sensor availability
   - Maintains trajectory tracking despite 60% packet loss
   - No system crash or reboot needed - continuous operation

5. **"Performance degrades but remains safe"**
   - Tracking error increases during attack (see Figure 3)
   - But drone doesn't crash or lose control
   - Recovers immediately when attack stops at t=6s
   - This demonstrates resilience through software rejuvenation

### Interactive Demo - Modify Drop Rate:
**Show defense works for different attack intensities:**

Edit `scenario2_dos_attack.py` around line 346-347:
```python
# Light attack (30% drops)
sensor = DoS_Sensor(
    attack_start=4.0,
    attack_duration=2.0,
    drop_rate=0.3,  # Change this
    ...
)

# Medium attack (60% drops - default)
drop_rate=0.6  # Current setting

# Severe attack (90% drops - stress test)
drop_rate=0.9  # Extreme packet loss
```

Run each and show:
- 30%: Minimal impact, estimator handles easily
- 60%: Noticeable degradation but stable (default)
- 90%: Significant tracking error but NO crash

**Try different attack durations:**
```python
# Line 346
attack_duration=1.0   # Short 1-second burst
attack_duration=4.0   # Prolonged 4-second attack
```

### Key Points:
- ❌ Attack: DoS (Packet Dropping)
- ✅ Defense: State Estimation (physics-based prediction)
- ✅ Defense: Graceful Degradation (adaptive control)
- 📊 Show packet drop count vs successful deliveries

---

## Demo 4: SCENARIO 3 - Replay Attack

### Command:
```bash
npm run scenario3
```

### What It Does:
- **ATTACK:** Stuxnet-style replay attack (2-phase)
  - Phase 1 (t=2-5s): Record normal control commands
  - Phase 2 (t=8-11s): Replay old commands + inject malicious control (magnitude=3.0)
- **DEFENSE ATTEMPT:** Reference Governor (safety-based constraints)
- **RESULT:** Defense is **INSUFFICIENT** - attack causes catastrophic failure
- Total simulation: 18 seconds

### What to Say:
1. **"Replay attacks are inspired by Stuxnet malware"**
   - Attacker records normal operation silently (Phase 1: t=2-5s)
   - Then replays those old commands while injecting malicious control (Phase 2: t=8-11s)
   - This makes the system think everything is normal while it's actually under attack
   - Famous attack used on Iranian nuclear centrifuges - very sophisticated!

2. **"Phase 1: Recording normal commands"**
   - Console shows: `[REC] [REPLAY ATTACK - PHASE 1: RECORDING]`
   - Attacker silently records control commands during normal figure-8 flight
   - Records ~194 commands over 3 seconds
   - Everything appears completely normal - no indication of attack yet

3. **"Phase 2: Replay + Malicious Injection"**
   - Console shows: `[REPLAY] [REPLAY ATTACK - PHASE 2: REPLAY + MALICIOUS INJECTION]`
   - Old recorded commands are replayed in a loop
   - **MALICIOUS INJECTION**: Attacker multiplies control commands by 3.0x (300% thrust!)
   - Adds random noise to make control even more chaotic
   - Console shows: `Injecting malicious control (magnitude=3.0)...`
   - This causes the drone to receive massively excessive thrust commands

4. **"We tried using Reference Governor as a defense"**
   - **What is a Reference Governor?** Think of it like a safety guard that watches the drone
   - It checks if the drone is doing something dangerous (flying too fast, tilting too much, going too low)
   - If it detects danger, it tries to correct the trajectory to keep the drone safe

   **Our Safety Thresholds (what the RG watches for):**
   - **Max Tilt Angle**: 45° (if drone tilts more, it might flip over)
   - **Min Altitude**: 0.05m (don't crash into the ground)
   - **Max Velocity**: 20 m/s (don't fly too fast)
   - **Max Acceleration**: 30 m/s² (don't accelerate too quickly)

   **How We Tried to Make It Work:**
   - First try: Used strict thresholds (30° tilt, 15 m/s velocity)
     - **Problem**: RG kept triggering during normal flight - too sensitive!
   - Second try: Relaxed thresholds (45° tilt, 20 m/s velocity)
     - **Goal**: Only trigger during real attacks, not normal figure-8 maneuvers
     - **Result**: Better! RG only triggers 8 times (mostly at startup + when attack begins)

5. **"But the Reference Governor defense FAILS against this attack"**
   - Console shows: `[RG!] [REFERENCE GOVERNOR INTERVENTION #8] t=7.98s` - RG detects tilt=52.2°
   - RG tries to override the unsafe trajectory
   - **BUT IT'S TOO LATE**: The malicious commands already caused massive damage
   - By t=17s, drone is over **4.6 KILOMETERS away** (should be within 0.6m!)

   **Why RG Failed:**
   - RG can only detect problems AFTER they start happening (reactive, not proactive)
   - The malicious injection (3.0x thrust) is so strong, damage happens instantly
   - By the time RG detects the violation, the drone has already been sent flying
   - **Lesson**: Some attacks are too fast/powerful for safety-based defenses alone

6. **"Look at the dramatic trajectory plots"**
   - **Figure 1 (3D Trajectory)**:
     - Blue = Normal operation (t=0-8s) - nice figure-8
     - Orange = Under attack (t=8-11s) - trajectory EXPLODES off course
     - Drone goes from 0.6m radius to multiple kilometers away
   - **Figure 2 (Position Tracking)**:
     - Massive position errors during orange attack phase
     - X, Y coordinates shoot up to hundreds/thousands of meters
     - Clearly shows catastrophic failure
   - **Figure 3 (Error Timeline & RG Interventions)**:
     - Top: Tracking error spikes during attack
     - Bottom: RG intervention timeline (green bars)
       - Shows RG triggers at startup (t=0-0.6s) - brief interventions during takeoff
       - Shows RG trigger at attack start (t=7.98s) - detects tilt violation
       - But interventions are too infrequent to stop the damage

### What This Demonstrates:
**This scenario is a "failure case" that teaches important lessons:**

✅ **Attack Success**: Replay attacks with malicious injection are EXTREMELY dangerous
- Even with 300% thrust injection, the attack executes successfully
- Causes catastrophic trajectory deviation (4+ km away)
- Demonstrates why Stuxnet was so devastating in real world

❌ **Defense Insufficient**: Reference Governor alone cannot stop this attack
- RG is reactive (responds to violations) not proactive (prevents violations)
- By the time RG detects danger (52° tilt), massive damage already done
- Safety thresholds can't prevent already-executed malicious commands

📚 **Key Lesson**: Not all defenses work for all attacks
- GPS Spoofing → Anomaly Detection + SR works great ✓
- DoS Attacks → State Estimation works great ✓
- Replay Attacks → Need better defense (cryptographic signing, command authentication, etc.) ✗

**Honest Discussion Point for Presentation:**
"We tried using a Reference Governor to defend against replay attacks, but it wasn't effective enough. This demonstrates that matching the right defense to the attack type is critical. For replay attacks, you really need cryptographic authentication of commands, not just safety monitoring. This 'failure case' is a valuable learning experience!"

### Interactive Demo - Modify Attack Strength:
**Show how attack severity affects outcome:**

Edit `scenario3_replay_attack.py` around line 436:
```python
# Current (catastrophic attack)
actuator = ReplayAttackActuator(
    record_start=2.0,
    record_end=5.0,
    replay_start=8.0,
    replay_end=11.0,
    injection_magnitude=3.0,  # 300% thrust - catastrophic
    ...
)

# Weaker attack (still severe)
injection_magnitude=2.0  # 200% thrust - still very bad

# Moderate attack (RG might help more)
injection_magnitude=1.5  # 150% thrust - RG has better chance
```

Run each and show:
- 3.0x: Complete catastrophic failure (4+ km deviation)
- 2.0x: Severe failure (still flies off)
- 1.5x: RG might provide more help (but still insufficient)

### Key Points:
- ❌ Attack: Replay Attack (Stuxnet-style) - Records then replays + malicious injection
- ⚠️ Defense Attempted: Reference Governor (safety constraints)
- ❌ **Defense Failed**: RG cannot prevent catastrophic damage from this attack
- 📊 Console shows: Recorded Commands: 194, Malicious Injected: 204, RG Interventions: 8
- 🎓 **Learning Outcome**: This demonstrates that not all defenses work for all attacks - important lesson!

---

## Presentation Flow Recommendation

### 1. Introduction (2 min)
"Today we're demonstrating software rejuvenation techniques for cyber-physical systems, specifically a UAV quadrotor under three different cyber attacks."

### 2. Baseline Demo (3 min)
- Run `npm test`
- Explain multithreading, noise, Reference Governor
- Show figure-8 plots and tracking performance

### 3. Attack Scenario 1: GPS Spoofing (4 min)
- Run `npm run scenario1`
- Explain attack and defense
- Show detection and SR in action
- LIVE: Modify magnitude, run again to show adaptability

### 4. Attack Scenario 2: DoS (4 min)
- Run `npm run scenario2`
- Explain attack and defense
- Show packet drops and estimation
- LIVE: Change drop rate to 90%, show it still works

### 5. Attack Scenario 3: Replay (4 min)
- Run `npm run scenario3`
- Explain Stuxnet-inspired attack
- Show RG intervention attempt
- **Emphasize this is a failure case**: RG cannot stop this attack
- Explain why it failed (reactive vs proactive defense)

### 6. Conclusion (1 min)
"We demonstrated three diverse cyber attacks. Two defenses worked successfully (anomaly detection for GPS spoofing, state estimation for DoS), but the Reference Governor failed against replay attacks. This teaches us that matching the right defense to the attack type is critical - not all defenses work for all attacks. For replay attacks, you need cryptographic authentication, not just safety monitoring. Learning from failures is as important as celebrating successes!"

---

## Quick Reference Commands

```bash
npm test           # Baseline - show normal operation
npm run scenario1  # GPS Spoofing attack
npm run scenario2  # DoS attack
npm run scenario3  # Replay attack
npm run all        # Run everything (for practice)
```

---

## Files to Modify for Live Demos

**Scenario 1 (GPS Spoofing):**
- File: `scenario1_fdi_attack.py`
- Line 27: `magnitude=` (try 1.5, 3.0, 5.0)
- Line 27: `pattern=` (try 'circular', 'drift', 'jump')

**Scenario 2 (DoS):**
- File: `scenario2_dos_attack.py`
- Line 347: `drop_rate=` (try 0.3, 0.6, 0.9)
- Line 346: `attack_duration=` (try 1.0, 2.0, 4.0)

**Scenario 3 (Replay):**
- File: `scenario3_replay_attack.py`
- Line 38: `record_duration=` (try 5.0, 8.0, 12.0)
- Line 40: `injection_magnitude=` (try 1.5, 3.0, 5.0)

---

## Expected Outputs Summary

| Test | Attack Type | Defense Mechanism | Defense Result | Key Metrics | Duration |
|------|-------------|-------------------|----------------|-------------|----------|
| Baseline | None | None (clean baseline) | N/A | XY RMS Error < 0.15m | 9s |
| Scenario 1 | GPS Spoofing (FDI) | Anomaly Detection + SR | ✅ **SUCCESS** | Detections: ~5-10, SR Activations: ~3-5 | 12s |
| Scenario 2 | DoS (Packet Drops) | State Estimator + Graceful Degradation | ✅ **SUCCESS** | 60% drops, ~120 estimations | 10s |
| Scenario 3 | Replay (Stuxnet) | Reference Governor (Safety) | ❌ **FAILED** | RG: 8 interventions, Deviation: 4+ km | 18s |

**Note on Scenario 3:** The Reference Governor defense is intentionally insufficient for this attack type. This demonstrates an important lesson: not all defenses work for all attacks!

---

## Troubleshooting

**If plots don't show figure-8:**
- Check that you're running the correct command
- Baseline uses `npm test` (not npm run baseline)

**If console is too noisy:**
- That's normal - shows real-time operation
- Focus on the status messages: `[RG]`, `[SR]`, `[ATTACK]`, etc.

**If you want to skip plot windows:**
- Close plot windows to continue to next test
- Or comment out `plt.show()` in the Python files
