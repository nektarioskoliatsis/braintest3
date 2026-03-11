#!/usr/bin/env python3
"""Main entry point: run a CPG-driven fly simulation.

Executes the full connectome-to-behavior pipeline:
  1. Initialize spiking CPG network
  2. Initialize FlyGym environment
  3. Run simulation: CPG → spike decoder → joint angles → physics
  4. Generate analysis plots and video

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --duration 3.0 --mode connectome --no-render
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cpg.spiking_cpg import SpikingCPG, CPGConfig
from cpg.connectome_cpg import ConnectomeCPG, ConnectomeConfig
from embodiment.fly_env import FlyEnvironment, EnvConfig
from embodiment.neural_controller import NeuralController, ControllerConfig
from analysis.gait_analysis import GaitAnalyzer
from analysis.neural_analysis import NeuralAnalyzer
from visualization.visualize import SimulationVisualizer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_simulation(
    config: dict,
    mode: str = "connectome",
    duration: float | None = None,
    render: bool = True,
    output_dir: str = "output",
) -> dict:
    """Run the full CPG-driven fly simulation.

    Args:
        config: Configuration dictionary.
        mode: "connectome" for connectome-inspired CPG, "handtuned" for baseline.
        duration: Override simulation duration in seconds.
        render: Whether to render video.
        output_dir: Directory for output files.

    Returns:
        Dictionary with simulation results and metrics.
    """
    # --- Setup ---
    cpg_cfg = CPGConfig.from_dict(config.get("cpg", {}))
    ctrl_cfg = ControllerConfig.from_dict(config.get("controller", {}))
    env_cfg = EnvConfig.from_dict(config.get("environment", {}))

    if duration is not None:
        env_cfg.sim_duration = duration

    if not render:
        env_cfg.render_mode = "none"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    # --- Initialize CPG ---
    print(f"Initializing {mode} CPG network...")
    if mode == "connectome":
        conn_cfg = ConnectomeConfig.from_dict(config.get("connectome", {}))
        cpg = ConnectomeCPG(cpg_config=cpg_cfg, connectome_config=conn_cfg)
        stats = cpg.get_weight_statistics()
        print(f"  Connectome CPG: {cpg.N} neurons, {stats['total_connections']} connections")
        print(f"  E/I ratio: {stats['ei_ratio']:.2f}, density: {stats['density']:.3f}")
    else:
        cpg = SpikingCPG(config=cpg_cfg)
        print(f"  Hand-tuned CPG: {cpg.N} neurons")

    # --- Initialize Controller ---
    controller = NeuralController(cpg, config=ctrl_cfg)

    # --- Initialize Environment ---
    print("Initializing FlyGym environment...")
    env = FlyEnvironment(config=env_cfg)

    try:
        env.initialize()
        obs = env.reset()
        has_env = True
        print(f"  FlyGym ready: {env.n_joints} joints, dt={env_cfg.physics_dt}s")
        # Use the actual init pose from the simulation as the controller rest pose
        if hasattr(env, 'init_joint_angles'):
            controller.set_rest_pose(env.init_joint_angles)
            print(f"  Rest pose set from simulation init pose")
    except (ImportError, Exception) as e:
        print(f"  FlyGym not available: {e}")
        print("  Running neural simulation only (no physics).")
        has_env = False

    # --- Run Simulation ---
    total_physics_steps = int(env_cfg.sim_duration / env_cfg.physics_dt)
    neural_steps_per_physics = env_cfg.neural_steps_per_physics

    print(f"\nRunning simulation for {env_cfg.sim_duration}s "
          f"({total_physics_steps} physics steps)...")
    start_time = time.time()

    # Storage for neural-only mode
    all_spike_data = []
    all_joint_targets = []
    all_timestamps = []
    all_positions = []
    all_contacts = []

    for step in tqdm(range(total_physics_steps), desc="Simulating"):
        # Run neural simulation (multiple steps per physics step)
        for _ in range(neural_steps_per_physics):
            spikes = cpg.step()

        # Decode spikes to joint targets
        spike_np = spikes.cpu().numpy() if hasattr(spikes, 'cpu') else spikes
        joint_targets = controller.decode_spikes(spike_np)

        # Step physics
        if has_env:
            obs, done = env.step(joint_targets)
            if done:
                print(f"  Simulation terminated at step {step}")
                break

        # Record
        t = step * env_cfg.physics_dt
        all_timestamps.append(t)
        all_joint_targets.append(joint_targets.copy())

        if has_env and env.record.body_position:
            all_positions.append(env.record.body_position[-1])
            all_contacts.append(env.record.foot_contacts[-1])
        else:
            all_positions.append(np.zeros(3))
            all_contacts.append(np.ones(6, dtype=bool))

    elapsed = time.time() - start_time
    print(f"Simulation complete in {elapsed:.1f}s "
          f"({total_physics_steps / elapsed:.0f} steps/s)")

    # --- Analysis ---
    print("\nAnalyzing results...")

    spike_history = cpg.get_spike_history()
    timestamps = np.array(all_timestamps)
    positions = np.array(all_positions)
    contacts = np.array(all_contacts)

    # Gait analysis
    gait_analyzer = GaitAnalyzer(contacts, timestamps, positions)
    metrics = gait_analyzer.compute_metrics()
    print("\nGait Metrics:")
    print(metrics.summary_string())

    # Neural analysis
    neural_analyzer = NeuralAnalyzer(
        spike_history, cpg_cfg.dt, cpg_cfg.neurons_per_module
    )

    # --- Visualization ---
    print("\nGenerating plots...")

    frames = env.get_frames() if has_env else []

    visualizer = SimulationVisualizer(
        spike_history=spike_history,
        neural_dt_ms=cpg_cfg.dt,
        neurons_per_module=cpg_cfg.neurons_per_module,
        foot_contacts=contacts,
        timestamps=timestamps,
        body_positions=positions,
        frames=frames,
    )

    plot_dir = os.path.join(output_dir, "plots")
    visualizer.generate_analysis_plots(plot_dir)

    if render and frames:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        # Save raw fly video directly via mediapy
        raw_video_path = os.path.join(video_dir, f"fly_{mode}.mp4")
        try:
            import mediapy as media
            media.write_video(raw_video_path, frames, fps=env_cfg.render_fps)
            print(f"Fly video saved to {raw_video_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"Could not save fly video: {e}")

        # Save combined neural+fly video
        combined_path = os.path.join(video_dir, f"simulation_{mode}.mp4")
        visualizer.generate_video(combined_path, fps=env_cfg.render_fps)

    # --- Cleanup ---
    if has_env:
        env.close()

    results = {
        "mode": mode,
        "duration": env_cfg.sim_duration,
        "metrics": metrics.to_dict(),
        "spike_history_shape": spike_history.shape,
        "n_physics_steps": len(timestamps),
    }

    print(f"\nResults saved to {output_dir}/")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPG-driven Drosophila simulation"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode", type=str, default="connectome",
        choices=["connectome", "handtuned"],
        help="CPG wiring mode",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Override simulation duration (seconds)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable video rendering",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    results = run_simulation(
        config=config,
        mode=args.mode,
        duration=args.duration,
        render=not args.no_render,
        output_dir=args.output,
    )

    print("\n=== Simulation Summary ===")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
