# CPS Drone Cyber Defense - Presentation Guide

## Overview

This project demonstrates software rejuvenation techniques to defend against cyber attacks on a UAV quadrotor system. You'll show 4 demos: 1 baseline + 3 attack scenarios with defenses.

---

## Demo 1: BASELINE - Normal Operation

### Command:
```bash
npm test
```

### What It Does:
- Runs multithreaded UAV simulation (4 threads: Ground Station, Sensor, Controller, Simulator)
- Executes figure-8 trajectory for 12 seconds (2 complete loops)
- Reference Governor enforces safety constraints
- Sensor noise (σ=0.05) simulates realistic measurements

### What to Say:
1. **"This is our baseline - normal operation with no attacks"**
   - Point out the 4 threads starting in console
   - Show real-time coordinate logging every second

2. **"We have sensor noise to simulate real-world conditions"**
   - Noise σ=0.05 on all 12 state measurements
   - Controller handles noisy data effectively

3. **"Reference Governor provides safety constraints"**
   - Watch console for `[RG: OK]` or `[RG ACTIVE]` messages
   - RG prevents unsafe control inputs by constraining references
   - At end, show "Total Interventions" count

4. **"Look at the figure-8 trajectory"**
   - Show 2D plot: clean figure-8 shape in XY plane
   - Show 3D plot: demonstrates altitude control
   - Blue line (actual) tracks red dashed line (reference) closely

5. **"Performance metrics show accurate tracking"**
   - Mean error: typically < 0.05m
   - Max error: typically < 0.6m
   - This is our baseline to compare attacks against

### Key Points:
- ✅ Multithreaded architecture
- ✅ Reference Governor active
- ✅ Noise handling
- ✅ Accurate trajectory tracking

---

## Demo 2: SCENARIO 1 - GPS Spoofing Attack

### Command:
```bash
npm run scenario1
```

### What It Does:
- **ATTACK:** GPS position sensors injected with false data (circular pattern)
- Attack starts at t=12s, runs for 12s
- Gradually ramps up to 3m magnitude spoofing
- **DEFENSE:** Chi-square anomaly detection + Software Rejuvenation (SR)

### What to Say:
1. **"GPS spoofing is a False Data Injection attack"**
   - Attacker injects fake GPS coordinates
   - Makes drone think it's in wrong location
   - Could cause crashes or navigation to wrong area

2. **"Attack starts at 12 seconds - watch the trajectory diverge"**
   - First 12s: normal figure-8
   - After 12s: trajectory goes off-course (red section in plot)
   - Without defense, drone would fly far from intended path

3. **"Our anomaly detector catches the spoofed GPS data"**
   - Uses chi-square statistical test
   - Compares measured vs predicted position
   - When chi^2 > threshold, alarm triggers
   - Console shows: `[!] [GPS SPOOFING DETECTED]`

4. **"Software Rejuvenation restores the clean state"**
   - Console shows: `[SR] [SOFTWARE REJUVENATION]`
   - Restores state from clean buffer (pre-attack data)
   - Recovery time: 0.8 seconds
   - Green sections in plot show post-SR recovery

5. **"Look at the detection plot (Figure 3)"**
   - Top: Residual spikes when attack active
   - Middle: Chi-square exceeds threshold (red line)
   - Bottom: SR activations (green bars)

### Interactive Demo - Modify Attack Strength:
**Show defense works for different attack magnitudes:**

Edit `scenario1_fdi_attack.py` line 27:
```python
# Weak attack (easy to handle)
attack = GPS_Spoofing_Attack(start_time=12.0, duration=12.0, magnitude=1.5, pattern='circular')

# Strong attack (stress test)
attack = GPS_Spoofing_Attack(start_time=12.0, duration=12.0, magnitude=5.0, pattern='circular')
```

Run again and show:
- Weak attack: Detection still works, faster recovery
- Strong attack: Detection still works, more SR activations

**Try different attack patterns:**
```python
pattern='drift'     # Linear drift attack
pattern='jump'      # Sudden position jumps
pattern='circular'  # Circular GPS error (default)
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
- Attack starts at t=10s, runs for 15s
- 60% packet drop rate during attack
- **DEFENSE:** State Estimator + Graceful Degradation

### What to Say:
1. **"DoS attacks drop communication packets"**
   - Attacker floods network or jams communication
   - 60% of sensor data doesn't reach controller
   - Could cause complete loss of control

2. **"Watch packet drops in the console"**
   - Console shows: `[ATTACK] DoS Attack - Packet DROPPED`
   - Orange sections in plots show attack period
   - Without defense, control becomes erratic

3. **"State Estimator predicts missing data"**
   - When packet dropped, use physics model to estimate state
   - Predicts where drone should be based on dynamics
   - Continues control even without sensor data

4. **"Graceful Degradation maintains stability"**
   - Console shows: `[DEFENSE] Using estimated state`
   - Controller adapts to degraded sensor availability
   - Maintains trajectory tracking despite 60% packet loss

5. **"Performance degrades but remains safe"**
   - Tracking error increases during attack (see plot)
   - But drone doesn't crash or lose control
   - Recovers immediately when attack stops

### Interactive Demo - Modify Drop Rate:
**Show defense works for different attack intensities:**

Edit `scenario2_dos_attack.py` line 23:
```python
# Light attack (30% drops)
attack = DoS_Attack(start_time=10.0, duration=15.0, drop_rate=0.3)

# Medium attack (60% drops - default)
attack = DoS_Attack(start_time=10.0, duration=15.0, drop_rate=0.6)

# Severe attack (90% drops - stress test)
attack = DoS_Attack(start_time=10.0, duration=15.0, drop_rate=0.9)
```

Run each and show:
- 30%: Minimal impact, estimator handles easily
- 60%: Noticeable degradation but stable
- 90%: Significant tracking error but NO crash

**Try different attack durations:**
```python
duration=5.0   # Short burst attack
duration=20.0  # Prolonged attack
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
- **ATTACK:** Stuxnet-style replay attack
- Phase 1 (t=0-8s): Record normal control commands
- Phase 2 (t=10-18s): Replay old commands + inject false data
- **DEFENSE:** Reference Governor (safety constraints)

### What to Say:
1. **"Replay attacks are inspired by Stuxnet malware"**
   - Attacker records normal operation (Phase 1)
   - Then replays old commands while showing fake sensor data
   - Operators see normal data but system is compromised
   - Famous attack on Iranian nuclear centrifuges

2. **"Phase 1: Recording normal commands"**
   - Console shows: `[REC] [REPLAY ATTACK - PHASE 1: RECORDING]`
   - Attacker silently records 8 seconds of control commands
   - Everything appears normal

3. **"Phase 2: Replay + False Data Injection"**
   - Console shows: `[REPLAY] [REPLAY ATTACK - PHASE 2: REPLAY + FDI]`
   - Old commands replayed in loop
   - Fresh reference data injected to confuse system
   - This causes trajectory to diverge dangerously

4. **"Reference Governor prevents unsafe states"**
   - RG checks if reference would violate safety constraints
   - Console shows: `[RG!] [REFERENCE GOVERNOR INTERVENTION]`
   - Overrides unsafe references with safe alternatives
   - Keeps control inputs within limits

5. **"Look at the intervention timeline (Figure 3)"**
   - Top: Position divergence during replay
   - Middle: Control saturation (limits hit)
   - Bottom: RG interventions (red bars) preventing violations

### Interactive Demo - Modify Attack Timing:
**Show defense works for different replay scenarios:**

Edit `scenario3_replay_attack.py` lines 38-39:
```python
# Short recording window (less data to replay)
attack = ReplayAttack(record_start=0.0, record_duration=5.0,
                      replay_start=8.0, replay_duration=10.0, ...)

# Long recording window (more sophisticated attack)
attack = ReplayAttack(record_start=0.0, record_duration=12.0,
                      replay_start=15.0, replay_duration=15.0, ...)
```

**Try different injection magnitudes:**
```python
# Line 40
injection_magnitude=1.5  # Moderate injection (default)
injection_magnitude=3.0  # Strong injection (stress test)
```

Run each and show:
- RG intervention count increases with stronger attacks
- System remains stable even with aggressive replays

### Key Points:
- ❌ Attack: Replay Attack (Stuxnet-style)
- ✅ Defense: Reference Governor (safety constraints)
- 📊 Show RG intervention count and successful overrides

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
- Show RG interventions
- Highlight safety constraint enforcement

### 6. Conclusion (1 min)
"We demonstrated three diverse cyber attacks and showed that software rejuvenation techniques—anomaly detection, state estimation, and reference governor—can effectively defend against them. The key is matching the right defense to each attack type."

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
- Line 23: `drop_rate=` (try 0.3, 0.6, 0.9)
- Line 23: `duration=` (try 5.0, 15.0, 20.0)

**Scenario 3 (Replay):**
- File: `scenario3_replay_attack.py`
- Line 38: `record_duration=` (try 5.0, 8.0, 12.0)
- Line 40: `injection_magnitude=` (try 1.5, 3.0, 5.0)

---

## Expected Outputs Summary

| Test | Attack Type | Defense Mechanism | Key Metric |
|------|-------------|-------------------|------------|
| Baseline | None | Reference Governor | Mean Error < 0.05m |
| Scenario 1 | GPS Spoofing (FDI) | Anomaly Detection + SR | Detection Count, SR Count |
| Scenario 2 | DoS (Packet Drops) | State Estimator + Graceful Degradation | Drop Rate %, Packets Lost |
| Scenario 3 | Replay (Stuxnet) | Reference Governor | RG Intervention Count |

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
