"""Gait analysis for Drosophila locomotion.

Computes standard gait metrics from foot contact data:
  - Gait diagram (stance/swing phases)
  - Tripod coordination index
  - Stride frequency, stride length, walking speed
  - Duty factor

Reference values for Drosophila walking (from DeAngelis et al., 2019):
  - Walking speed: ~20-30 mm/s
  - Stride frequency: ~8-15 Hz
  - Duty factor: ~0.5-0.7 (stance fraction)
  - Tripod coordination: >0.8 for fast walking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


LEG_LABELS = ["L1 (LF)", "L2 (LM)", "L3 (LH)", "R1 (RF)", "R2 (RM)", "R3 (RH)"]

# Tripod groups
TRIPOD_A = [0, 4, 2]  # L1, R2, L3
TRIPOD_B = [3, 1, 5]  # R1, L2, R3


@dataclass
class GaitMetrics:
    """Container for computed gait metrics."""

    walking_speed: float = 0.0          # mm/s (average forward speed)
    stride_frequency: float = 0.0       # Hz (average across legs)
    stride_length: float = 0.0          # mm
    duty_factor: float = 0.0            # fraction of cycle in stance (0-1)
    tripod_index: float = 0.0           # tripod coordination index (0-1)
    stability: float = 0.0             # CoM lateral variance (lower = more stable)
    stride_regularity: float = 0.0      # coefficient of variation of stride period

    def to_dict(self) -> dict[str, float]:
        return {
            "walking_speed_mm_s": self.walking_speed,
            "stride_frequency_Hz": self.stride_frequency,
            "stride_length_mm": self.stride_length,
            "duty_factor": self.duty_factor,
            "tripod_index": self.tripod_index,
            "stability_CoM_var": self.stability,
            "stride_regularity_cv": self.stride_regularity,
        }

    def summary_string(self) -> str:
        lines = [
            f"Walking speed:     {self.walking_speed:.2f} mm/s",
            f"Stride frequency:  {self.stride_frequency:.2f} Hz",
            f"Stride length:     {self.stride_length:.2f} mm",
            f"Duty factor:       {self.duty_factor:.3f}",
            f"Tripod index:      {self.tripod_index:.3f}",
            f"Stability (CoM):   {self.stability:.4f}",
            f"Stride regularity: {self.stride_regularity:.3f}",
        ]
        return "\n".join(lines)


class GaitAnalyzer:
    """Analyzes locomotion gait from simulation recordings."""

    def __init__(
        self,
        foot_contacts: np.ndarray,
        timestamps: np.ndarray,
        body_positions: Optional[np.ndarray] = None,
        min_stride_duration: float = 0.03,
    ) -> None:
        """Initialize gait analyzer.

        Args:
            foot_contacts: Boolean array of shape (T, 6), True = stance.
            timestamps: Time array of shape (T,) in seconds.
            body_positions: Position array of shape (T, 3), optional.
            min_stride_duration: Minimum stride duration in seconds.
        """
        self.contacts = foot_contacts.astype(bool)
        self.timestamps = timestamps
        self.positions = body_positions
        self.min_stride = min_stride_duration
        self.dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.001
        self.T = len(timestamps)

    def compute_metrics(self) -> GaitMetrics:
        """Compute all gait metrics."""
        metrics = GaitMetrics()
        metrics.walking_speed = self._walking_speed()
        metrics.stride_frequency = self._stride_frequency()
        metrics.duty_factor = self._duty_factor()
        metrics.tripod_index = self._tripod_index()
        metrics.stability = self._stability()
        metrics.stride_regularity = self._stride_regularity()

        if metrics.stride_frequency > 0:
            metrics.stride_length = metrics.walking_speed / metrics.stride_frequency
        else:
            metrics.stride_length = 0.0

        return metrics

    def _walking_speed(self) -> float:
        """Compute average forward walking speed in mm/s."""
        if self.positions is None or len(self.positions) < 2:
            return 0.0
        # Forward direction is x-axis in FlyGym
        displacement = self.positions[-1, 0] - self.positions[0, 0]
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        # Convert from model units (m) to mm
        return abs(displacement) * 1000.0 / duration

    def _stride_frequency(self) -> float:
        """Compute average stride frequency across all legs."""
        frequencies = []
        for leg in range(6):
            periods = self._detect_stride_periods(leg)
            if len(periods) > 0:
                frequencies.append(1.0 / np.mean(periods))
        return np.mean(frequencies) if frequencies else 0.0

    def _detect_stride_periods(self, leg: int) -> list[float]:
        """Detect stride periods (swing onset to swing onset) for a leg."""
        contacts = self.contacts[:, leg]

        # Find stance→swing transitions (contact drops)
        transitions = np.where(np.diff(contacts.astype(int)) == -1)[0]

        if len(transitions) < 2:
            return []

        periods = []
        for i in range(1, len(transitions)):
            period = self.timestamps[transitions[i]] - self.timestamps[transitions[i - 1]]
            if period >= self.min_stride:
                periods.append(period)

        return periods

    def _duty_factor(self) -> float:
        """Compute average duty factor (fraction of time in stance)."""
        if self.T == 0:
            return 0.0
        return float(self.contacts.mean())

    def _tripod_index(self) -> float:
        """Compute tripod coordination index.

        Measures the fraction of time where a valid tripod support pattern
        is maintained: either tripod A (L1, R2, L3) or tripod B (R1, L2, R3)
        has at least 2 out of 3 legs in stance while the other group has
        at least 2 legs in swing.
        """
        if self.T == 0:
            return 0.0

        tripod_count = 0
        for t in range(self.T):
            c = self.contacts[t]
            # Tripod A in stance, B in swing
            a_stance = sum(c[i] for i in TRIPOD_A)
            b_stance = sum(c[i] for i in TRIPOD_B)

            # Valid tripod: one group mostly in stance, other mostly in swing
            if (a_stance >= 2 and b_stance <= 1) or (b_stance >= 2 and a_stance <= 1):
                tripod_count += 1

        return tripod_count / self.T

    def _stability(self) -> float:
        """Compute lateral stability as CoM y-position variance."""
        if self.positions is None or len(self.positions) < 2:
            return 0.0
        return float(np.var(self.positions[:, 1]))

    def _stride_regularity(self) -> float:
        """Compute stride regularity as coefficient of variation of stride periods."""
        all_periods = []
        for leg in range(6):
            all_periods.extend(self._detect_stride_periods(leg))
        if len(all_periods) < 2:
            return 1.0  # No regularity measurable
        periods = np.array(all_periods)
        return float(np.std(periods) / (np.mean(periods) + 1e-8))

    def plot_gait_diagram(
        self, ax: Optional[plt.Axes] = None, title: str = "Gait Diagram"
    ) -> Figure:
        """Plot gait diagram: horizontal bars showing stance/swing per leg.

        Stance is shown as filled bars, swing as gaps.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

        for leg in range(6):
            contacts = self.contacts[:, leg]
            # Find contiguous stance periods
            stance_starts = []
            stance_ends = []
            in_stance = False

            for t in range(self.T):
                if contacts[t] and not in_stance:
                    stance_starts.append(self.timestamps[t])
                    in_stance = True
                elif not contacts[t] and in_stance:
                    stance_ends.append(self.timestamps[t])
                    in_stance = False

            if in_stance:
                stance_ends.append(self.timestamps[-1])

            # Draw stance bars
            for start, end in zip(stance_starts, stance_ends):
                ax.barh(
                    5 - leg, end - start, left=start, height=0.7,
                    color=colors[leg], alpha=0.8, edgecolor="black", linewidth=0.5,
                )

        ax.set_yticks(range(6))
        ax.set_yticklabels(list(reversed(LEG_LABELS)))
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.set_xlim(self.timestamps[0], self.timestamps[-1])
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_body_trajectory(self, ax: Optional[plt.Axes] = None) -> Optional[Figure]:
        """Plot top-down body trajectory (x vs y)."""
        if self.positions is None:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        x = self.positions[:, 0] * 1000  # to mm
        y = self.positions[:, 1] * 1000

        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7)
        ax.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax.plot(x[-1], y[-1], "r^", markersize=10, label="End")

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Body Trajectory (top-down)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig
