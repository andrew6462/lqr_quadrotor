# src/main_min.py
# Minimal driver that *uses* main_cps_refactor to run a few tests concurrently.

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# --- import from repo root ---
try:
    from main_cps_refactor import load_gains, simulate_one, plot_sim
except ModuleNotFoundError:
    import os, sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, REPO_ROOT)
    from main_cps_refactor import load_gains, simulate_one, plot_sim

# name, plant, traj, use_rg, use_integrator, overrides (optional)
SCENARIOS = [
    #("linear_step",     "linear",    "step_z",  False, True,  {}),            # linear step (integrator if Kc exists)
    #("nonlinear_step",  "nonlinear", "step_z",  True,  False, {}),            # nonlinear step with RG
    ("p8_almost",       "nonlinear", "figure8", True,  False, {"T": 6.0}),    # partial figure-8 (shorter T)
    # NEW: noisy figure-8 (sensor noise with fixed seed)
    ("fig8_noisy",      "nonlinear", "figure8", True,  False, {"T": 6.0,"sensor_noise": 0.05, "seed": 42}),
    #("bulbs",       "nonlinear", "figure8", True,  False, {"T": 12.0, "Ts": 0.004})
]

def run_scenarios(T, Ts, mat_dir, save_prefix, z_final, step_time):
    gains = load_gains(mat_dir)

    def submit(name, plant, traj, use_rg, use_int, overrides):
        # apply per-scenario overrides
        overrides = overrides or {}
        T_eff   = overrides.get("T", T)
        Ts_eff  = overrides.get("Ts", Ts)
        kwargs  = {
            "name": name, "plant": plant, "traj": traj,
            "T": T_eff, "Ts": Ts_eff, "use_rg": use_rg, "use_integrator": use_int,
            "gains": gains, "z_final": z_final, "step_time": step_time
        }
        # Pass optional noise args only if simulate_one supports them
        if "sensor_noise" in overrides:
            kwargs["sensor_noise"] = overrides["sensor_noise"]
        if "seed" in overrides:
            kwargs["seed"] = overrides["seed"]

        try:
            return simulate_one(**kwargs)
        except TypeError:
            # Older main_cps_refactor without noise support: strip noise args and retry
            kwargs.pop("sensor_noise", None)
            kwargs.pop("seed", None)
            return simulate_one(**kwargs)

    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(SCENARIOS))) as pool:
        futs = [pool.submit(submit, *sc) for sc in SCENARIOS]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Plot after sims finish (avoid matplotlib concurrency issues)
    for res in results:
        plot_sim(res, save_prefix=save_prefix)

def main():
    ap = argparse.ArgumentParser(description="Minimal Quad LQR test runner")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--Ts", type=float, default=0.008)
    ap.add_argument("--mat-dir", type=str, default="mat")
    ap.add_argument("--save-prefix", type=str, default="mini")
    ap.add_argument("--z-final", type=float, default=1.0)
    ap.add_argument("--step-time", type=float, default=0.2)
    args = ap.parse_args()

    run_scenarios(
        T=args.T, Ts=args.Ts, mat_dir=args.mat_dir,
        save_prefix=args.save_prefix, z_final=args.z_final, step_time=args.step_time
    )

if __name__ == "__main__":
    main()
