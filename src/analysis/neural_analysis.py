"""Neural analysis of CPG spike train activity.

Provides visualization and quantification of:
  - Spike raster plots (color-coded by leg module)
  - Phase relationships between leg modules
  - Firing rate time series
  - Cross-correlograms between contralateral pairs
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d


LEG_NAMES = ["L1", "L2", "L3", "R1", "R2", "R3"]
LEG_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

# Contralateral pairs for cross-correlation
CONTRA_PAIRS = [(0, 3), (1, 4), (2, 5)]  # L1-R1, L2-R2, L3-R3


class NeuralAnalyzer:
    """Analyzes spiking activity from the CPG network."""

    def __init__(
        self,
        spike_history: np.ndarray,
        dt_ms: float = 0.1,
        neurons_per_module: int = 20,
        num_legs: int = 6,
    ) -> None:
        """Initialize neural analyzer.

        Args:
            spike_history: Binary spike matrix of shape (T, N).
            dt_ms: Simulation timestep in milliseconds.
            neurons_per_module: Neurons per leg module.
            num_legs: Number of leg modules.
        """
        self.spikes = spike_history
        self.dt = dt_ms
        self.n_per_module = neurons_per_module
        self.num_legs = num_legs
        self.half_module = neurons_per_module // 2
        self.T = spike_history.shape[0]
        self.times = np.arange(self.T) * dt_ms  # in ms

    def plot_raster(
        self,
        ax: Optional[plt.Axes] = None,
        time_range: Optional[tuple[float, float]] = None,
        title: str = "CPG Spike Raster",
    ) -> Figure:
        """Plot spike raster color-coded by leg module.

        Args:
            ax: Optional axes to plot on.
            time_range: Optional (start_ms, end_ms) to zoom in.
            title: Plot title.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig = ax.figure

        # Determine time range
        if time_range is not None:
            t_start = int(time_range[0] / self.dt)
            t_end = min(int(time_range[1] / self.dt), self.T)
        else:
            t_start, t_end = 0, self.T

        for leg in range(self.num_legs):
            base = leg * self.n_per_module
            for n in range(self.n_per_module):
                neuron_id = base + n
                spike_times = np.where(self.spikes[t_start:t_end, neuron_id] > 0)[0]
                spike_times_ms = (spike_times + t_start) * self.dt

                if len(spike_times_ms) > 0:
                    ax.scatter(
                        spike_times_ms, np.full_like(spike_times_ms, neuron_id),
                        s=0.5, c=LEG_COLORS[leg], alpha=0.7, marker="|",
                        linewidths=0.5,
                    )

        # Add leg labels
        for leg in range(self.num_legs):
            mid = leg * self.n_per_module + self.n_per_module // 2
            ax.text(
                self.times[t_start] - (t_end - t_start) * self.dt * 0.02,
                mid, LEG_NAMES[leg],
                fontsize=9, fontweight="bold", color=LEG_COLORS[leg],
                ha="right", va="center",
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron ID")
        ax.set_title(title)
        ax.set_xlim(t_start * self.dt, t_end * self.dt)

        # Add horizontal lines between modules
        for leg in range(1, self.num_legs):
            ax.axhline(leg * self.n_per_module - 0.5, color="gray",
                       linewidth=0.5, linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig

    def compute_module_rates(self, sigma_ms: float = 20.0) -> np.ndarray:
        """Compute smoothed firing rates per module.

        Args:
            sigma_ms: Gaussian smoothing kernel width in ms.

        Returns:
            Array of shape (T, num_legs, 2): [flexor_rate, extensor_rate] in Hz.
        """
        sigma_steps = max(1, int(sigma_ms / self.dt))
        rates = np.zeros((self.T, self.num_legs, 2))

        for leg in range(self.num_legs):
            base = leg * self.n_per_module

            # Flexor pool
            flex = self.spikes[:, base:base + self.half_module].sum(axis=1).astype(float)
            flex_smooth = gaussian_filter1d(flex, sigma_steps)
            rates[:, leg, 0] = flex_smooth / (self.half_module * self.dt * 1e-3)

            # Extensor pool
            ext = self.spikes[:, base + self.half_module:base + self.n_per_module].sum(axis=1).astype(float)
            ext_smooth = gaussian_filter1d(ext, sigma_steps)
            rates[:, leg, 1] = ext_smooth / (self.half_module * self.dt * 1e-3)

        return rates

    def plot_firing_rates(
        self,
        ax: Optional[plt.Axes] = None,
        sigma_ms: float = 20.0,
        title: str = "Module Firing Rates",
    ) -> Figure:
        """Plot smoothed firing rates for each leg module."""
        rates = self.compute_module_rates(sigma_ms)

        if ax is None:
            fig, axes = plt.subplots(self.num_legs, 1, figsize=(12, 8), sharex=True)
        else:
            fig = ax.figure
            axes = [ax]

        time_s = self.times / 1000.0  # convert to seconds

        for leg in range(min(self.num_legs, len(axes))):
            a = axes[leg] if isinstance(axes, np.ndarray) else axes
            a.fill_between(time_s, rates[:, leg, 0], alpha=0.5,
                           color=LEG_COLORS[leg], label="Flexor")
            a.fill_between(time_s, -rates[:, leg, 1], alpha=0.5,
                           color=LEG_COLORS[leg], label="Extensor")
            a.axhline(0, color="gray", linewidth=0.5)
            a.set_ylabel(f"{LEG_NAMES[leg]}\n(Hz)")
            a.set_ylim(-200, 200)
            if leg == 0:
                a.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title(title)

        fig.tight_layout()
        return fig

    def compute_phase_relationships(self, sigma_ms: float = 20.0) -> np.ndarray:
        """Compute instantaneous phase per module using Hilbert transform.

        Returns:
            Phase array of shape (T, num_legs) in radians [0, 2*pi].
        """
        from scipy.signal import hilbert

        sigma_steps = max(1, int(sigma_ms / self.dt))
        phases = np.zeros((self.T, self.num_legs))

        for leg in range(self.num_legs):
            base = leg * self.n_per_module
            # Use flexor-extensor difference as the oscillatory signal
            flex = self.spikes[:, base:base + self.half_module].sum(axis=1).astype(float)
            ext = self.spikes[:, base + self.half_module:base + self.n_per_module].sum(axis=1).astype(float)

            signal = gaussian_filter1d(flex - ext, sigma_steps)

            # Hilbert transform for instantaneous phase
            analytic = hilbert(signal)
            phases[:, leg] = np.angle(analytic) % (2 * np.pi)

        return phases

    def plot_phase_relationships(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Inter-leg Phase Relationships",
    ) -> Figure:
        """Plot phase differences between contralateral leg pairs."""
        phases = self.compute_phase_relationships()

        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig = ax.figure
            axes = [ax]

        pair_labels = ["L1 vs R1", "L2 vs R2", "L3 vs R3"]
        for i, (left, right) in enumerate(CONTRA_PAIRS):
            if i >= len(axes):
                break
            a = axes[i]
            phase_diff = (phases[:, left] - phases[:, right]) % (2 * np.pi)

            # Circular histogram
            a.hist(phase_diff, bins=36, density=True, color=LEG_COLORS[left], alpha=0.7)
            a.axvline(np.pi, color="red", linestyle="--", linewidth=2,
                      label="180° (anti-phase)")
            a.set_xlabel("Phase difference (rad)")
            a.set_ylabel("Density")
            a.set_title(pair_labels[i])
            a.legend(fontsize=8)
            a.set_xlim(0, 2 * np.pi)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def compute_cross_correlogram(
        self, leg_a: int, leg_b: int, max_lag_ms: float = 200.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlogram between two leg modules.

        Args:
            leg_a, leg_b: Leg indices (0-5).
            max_lag_ms: Maximum lag to compute in ms.

        Returns:
            Tuple of (lags_ms, correlation) arrays.
        """
        # Use flexor pool spike counts
        base_a = leg_a * self.n_per_module
        base_b = leg_b * self.n_per_module

        sig_a = self.spikes[:, base_a:base_a + self.half_module].sum(axis=1).astype(float)
        sig_b = self.spikes[:, base_b:base_b + self.half_module].sum(axis=1).astype(float)

        # Smooth
        sigma = max(1, int(5.0 / self.dt))
        sig_a = gaussian_filter1d(sig_a, sigma)
        sig_b = gaussian_filter1d(sig_b, sigma)

        # Zero-mean
        sig_a -= sig_a.mean()
        sig_b -= sig_b.mean()

        # Cross-correlation
        max_lag_steps = int(max_lag_ms / self.dt)
        corr = correlate(sig_a, sig_b, mode="full")
        mid = len(corr) // 2

        # Normalize
        norm = np.sqrt(np.sum(sig_a**2) * np.sum(sig_b**2))
        if norm > 0:
            corr /= norm

        # Extract within max_lag
        start = max(0, mid - max_lag_steps)
        end = min(len(corr), mid + max_lag_steps + 1)
        corr_trimmed = corr[start:end]
        lags = (np.arange(start, end) - mid) * self.dt

        return lags, corr_trimmed

    def plot_cross_correlograms(
        self,
        ax: Optional[plt.Axes] = None,
        max_lag_ms: float = 200.0,
        title: str = "Cross-Correlograms (Contralateral Pairs)",
    ) -> Figure:
        """Plot cross-correlograms for contralateral leg pairs."""
        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            fig = ax.figure
            axes = [ax]

        pair_labels = ["L1 vs R1", "L2 vs R2", "L3 vs R3"]
        for i, (left, right) in enumerate(CONTRA_PAIRS):
            if i >= len(axes):
                break
            lags, corr = self.compute_cross_correlogram(left, right, max_lag_ms)
            a = axes[i]
            a.plot(lags, corr, color=LEG_COLORS[left], linewidth=1.5)
            a.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            a.axhline(0, color="gray", linewidth=0.5)
            a.set_xlabel("Lag (ms)")
            a.set_ylabel("Correlation")
            a.set_title(pair_labels[i])

        fig.suptitle(title)
        fig.tight_layout()
        return fig
