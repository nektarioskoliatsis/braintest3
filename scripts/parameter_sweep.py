#!/usr/bin/env python3
"""Parameter sweep over CPG parameters to find stable gait regimes.

Sweeps oscillation frequency, coupling strength, and noise level,
measuring gait quality metrics at each point. Produces heatmaps
showing where stable tripod gait emerges.

Usage:
    python scripts/parameter_sweep.py
    python scripts/parameter_sweep.py --output output/sweep --duration 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cpg.spiking_cpg import SpikingCPG, CPGConfig
from cpg.connectome_cpg import ConnectomeCPG, ConnectomeConfig
from embodiment.neural_controller import NeuralController, ControllerConfig


def evaluate_cpg_gait(
    cpg_cfg: CPGConfig,
    duration_ms: float = 1000.0,
    ctrl_cfg: ControllerConfig | None = None,
) -> dict[str, float]:
    """Evaluate gait quality of a CPG configuration (neural only).

    Since we may not have FlyGym, we evaluate gait quality from the
    neural output alone:
      - Rhythmicity: autocorrelation peak strength
      - Tripod coordination: phase relationship between tripod groups
      - Stability: regularity of oscillation period
    """
    cpg = SpikingCPG(config=cpg_cfg)
    controller = NeuralController(cpg, config=ctrl_cfg or ControllerConfig())

    # Run CPG
    spikes = cpg.run(duration_ms)
    spike_np = spikes.cpu().numpy() if hasattr(spikes, 'cpu') else spikes

    if spike_np.shape[0] < 100:
        return {"rhythmicity": 0, "tripod_score": 0, "regularity": 0, "mean_rate": 0}

    n = cpg_cfg.neurons_per_module
    half = cpg_cfg.half_module
    T = spike_np.shape[0]

    # --- Compute module activity signals ---
    from scipy.ndimage import gaussian_filter1d
    sigma = max(1, int(10.0 / cpg_cfg.dt))

    module_signals = []
    for leg in range(6):
        base = leg * n
        flex = spike_np[:, base:base + half].sum(axis=1).astype(float)
        ext = spike_np[:, base + half:base + n].sum(axis=1).astype(float)
        signal = gaussian_filter1d(flex - ext, sigma)
        module_signals.append(signal)

    module_signals = np.array(module_signals)  # (6, T)

    # --- Rhythmicity: autocorrelation of module signals ---
    from scipy.signal import correlate

    rhythmicity_scores = []
    for leg in range(6):
        sig = module_signals[leg] - module_signals[leg].mean()
        if np.std(sig) < 1e-8:
            continue
        acorr = correlate(sig, sig, mode="full")
        acorr = acorr[len(acorr) // 2:]  # positive lags only
        acorr /= acorr[0] + 1e-8  # normalize

        # Find first peak after zero-crossing
        zero_cross = np.where(np.diff(np.sign(acorr)))[0]
        if len(zero_cross) > 0:
            search_start = zero_cross[0]
            peaks_region = acorr[search_start:]
            if len(peaks_region) > 0:
                peak_val = np.max(peaks_region)
                rhythmicity_scores.append(max(0, peak_val))

    rhythmicity = np.mean(rhythmicity_scores) if rhythmicity_scores else 0.0

    # --- Tripod coordination ---
    # Phase coherence between legs in the same tripod group
    tripod_a = [0, 4, 2]  # L1, R2, L3
    tripod_b = [3, 1, 5]  # R1, L2, R3

    def tripod_coherence(group: list[int]) -> float:
        """Correlation between legs in the same tripod group."""
        corrs = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sig_i = module_signals[group[i]]
                sig_j = module_signals[group[j]]
                std_i, std_j = np.std(sig_i), np.std(sig_j)
                if std_i > 1e-8 and std_j > 1e-8:
                    c = np.corrcoef(sig_i, sig_j)[0, 1]
                    corrs.append(c)
        return np.mean(corrs) if corrs else 0.0

    def anti_phase_score(group_a: list[int], group_b: list[int]) -> float:
        """Anti-correlation between opposing tripod groups."""
        corrs = []
        for a in group_a:
            for b in group_b:
                sig_a = module_signals[a]
                sig_b = module_signals[b]
                std_a, std_b = np.std(sig_a), np.std(sig_b)
                if std_a > 1e-8 and std_b > 1e-8:
                    c = np.corrcoef(sig_a, sig_b)[0, 1]
                    corrs.append(-c)  # negative correlation = good anti-phase
        return np.mean(corrs) if corrs else 0.0

    within_a = tripod_coherence(tripod_a)
    within_b = tripod_coherence(tripod_b)
    between = anti_phase_score(tripod_a, tripod_b)
    tripod_score = (within_a + within_b + between) / 3.0

    # --- Regularity: CV of inter-peak intervals ---
    all_intervals = []
    for leg in range(6):
        sig = module_signals[leg]
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(sig, distance=int(20.0 / cpg_cfg.dt))
        if len(peaks) > 1:
            intervals = np.diff(peaks) * cpg_cfg.dt
            all_intervals.extend(intervals.tolist())

    if len(all_intervals) > 2:
        intervals_arr = np.array(all_intervals)
        regularity = 1.0 - min(1.0, np.std(intervals_arr) / (np.mean(intervals_arr) + 1e-8))
    else:
        regularity = 0.0

    # --- Mean firing rate ---
    mean_rate = float(spike_np.sum()) / (T * cpg_cfg.dt * 1e-3 * cpg_cfg.total_neurons)

    return {
        "rhythmicity": float(np.clip(rhythmicity, 0, 1)),
        "tripod_score": float(np.clip(tripod_score, 0, 1)),
        "regularity": float(np.clip(regularity, 0, 1)),
        "mean_rate": mean_rate,
    }


def parameter_sweep(
    config: dict,
    output_dir: str = "output/sweep",
    duration_ms: float = 1000.0,
) -> None:
    """Run parameter sweep and generate heatmaps."""
    os.makedirs(output_dir, exist_ok=True)

    # Sweep ranges
    frequencies = np.array([5, 7, 10, 12, 15, 18, 20])
    coupling_strengths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    noise_levels = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

    base_cfg = config.get("cpg", {})

    # --- Sweep 1: Frequency × Coupling Strength (fixed noise) ---
    print("Sweep 1: Frequency × Coupling Strength")
    n_freq = len(frequencies)
    n_coup = len(coupling_strengths)
    results_fc = {
        "rhythmicity": np.zeros((n_freq, n_coup)),
        "tripod_score": np.zeros((n_freq, n_coup)),
        "regularity": np.zeros((n_freq, n_coup)),
        "mean_rate": np.zeros((n_freq, n_coup)),
    }

    total = n_freq * n_coup
    with tqdm(total=total, desc="  Freq x Coupling") as pbar:
        for i, freq in enumerate(frequencies):
            for j, coup in enumerate(coupling_strengths):
                cfg = CPGConfig.from_dict(base_cfg)
                cfg.oscillation_freq = freq
                cfg.w_ipsilateral = coup
                cfg.w_contralateral = -coup * 1.2

                metrics = evaluate_cpg_gait(cfg, duration_ms)
                for key in results_fc:
                    results_fc[key][i, j] = metrics[key]
                pbar.update(1)

    # --- Sweep 2: Frequency × Noise (fixed coupling) ---
    print("Sweep 2: Frequency × Noise Level")
    n_noise = len(noise_levels)
    results_fn = {
        "rhythmicity": np.zeros((n_freq, n_noise)),
        "tripod_score": np.zeros((n_freq, n_noise)),
        "regularity": np.zeros((n_freq, n_noise)),
    }

    with tqdm(total=n_freq * n_noise, desc="  Freq x Noise") as pbar:
        for i, freq in enumerate(frequencies):
            for j, noise in enumerate(noise_levels):
                cfg = CPGConfig.from_dict(base_cfg)
                cfg.oscillation_freq = freq
                cfg.noise_std = noise

                metrics = evaluate_cpg_gait(cfg, duration_ms)
                for key in results_fn:
                    results_fn[key][i, j] = metrics[key]
                pbar.update(1)

    # --- Generate Heatmaps ---
    print("\nGenerating heatmaps...")

    # Composite gait quality = mean of rhythmicity, tripod_score, regularity
    composite_fc = (results_fc["rhythmicity"] + results_fc["tripod_score"] + results_fc["regularity"]) / 3.0
    composite_fn = (results_fn["rhythmicity"] + results_fn["tripod_score"] + results_fn["regularity"]) / 3.0

    # Heatmap 1: Frequency × Coupling
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (key, title) in zip(axes.flat, [
        ("rhythmicity", "Rhythmicity"),
        ("tripod_score", "Tripod Coordination"),
        ("regularity", "Stride Regularity"),
        ("mean_rate", "Mean Firing Rate (Hz)"),
    ]):
        data = results_fc[key]
        norm = None if key == "mean_rate" else Normalize(0, 1)
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis", norm=norm)
        ax.set_xticks(range(n_coup))
        ax.set_xticklabels([f"{c:.1f}" for c in coupling_strengths])
        ax.set_yticks(range(n_freq))
        ax.set_yticklabels([f"{f:.0f}" for f in frequencies])
        ax.set_xlabel("Coupling Strength")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("CPG Parameter Sweep: Frequency × Coupling Strength", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_freq_coupling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Heatmap 2: Composite gait quality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(composite_fc, aspect="auto", origin="lower", cmap="RdYlGn",
                     norm=Normalize(0, 1))
    ax1.set_xticks(range(n_coup))
    ax1.set_xticklabels([f"{c:.1f}" for c in coupling_strengths])
    ax1.set_yticks(range(n_freq))
    ax1.set_yticklabels([f"{f:.0f}" for f in frequencies])
    ax1.set_xlabel("Coupling Strength")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Gait Quality (Freq × Coupling)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(composite_fn, aspect="auto", origin="lower", cmap="RdYlGn",
                     norm=Normalize(0, 1))
    ax2.set_xticks(range(n_noise))
    ax2.set_xticklabels([f"{n:.1f}" for n in noise_levels])
    ax2.set_yticks(range(n_freq))
    ax2.set_yticklabels([f"{f:.0f}" for f in frequencies])
    ax2.set_xlabel("Noise Level (nA)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Gait Quality (Freq × Noise)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Stable Tripod Gait Regimes", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gait_quality_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Find optimal parameters
    best_idx = np.unravel_index(composite_fc.argmax(), composite_fc.shape)
    print(f"\nBest parameters (Freq × Coupling):")
    print(f"  Frequency: {frequencies[best_idx[0]]:.0f} Hz")
    print(f"  Coupling:  {coupling_strengths[best_idx[1]]:.1f}")
    print(f"  Gait quality: {composite_fc[best_idx]:.3f}")

    best_idx2 = np.unravel_index(composite_fn.argmax(), composite_fn.shape)
    print(f"\nBest parameters (Freq × Noise):")
    print(f"  Frequency: {frequencies[best_idx2[0]]:.0f} Hz")
    print(f"  Noise:     {noise_levels[best_idx2[1]]:.1f} nA")
    print(f"  Gait quality: {composite_fn[best_idx2]:.3f}")

    print(f"\nSweep results saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPG parameter sweep")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--output", type=str, default="output/sweep")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Duration per trial in seconds")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    parameter_sweep(config, args.output, duration_ms=args.duration * 1000)


if __name__ == "__main__":
    main()
