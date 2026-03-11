"""Visualization: combined neural activity + embodied simulation output.

Generates side-by-side visualizations:
  - Left panel: CPG spike raster (scrolling)
  - Right panel: Fly walking in MuJoCo
  - Bottom: Gait diagram overlay
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec

try:
    from ..analysis.neural_analysis import NeuralAnalyzer, LEG_NAMES, LEG_COLORS
    from ..analysis.gait_analysis import GaitAnalyzer
except (ImportError, ValueError):
    from analysis.neural_analysis import NeuralAnalyzer, LEG_NAMES, LEG_COLORS
    from analysis.gait_analysis import GaitAnalyzer


class SimulationVisualizer:
    """Creates combined visualizations of neural + embodied simulation."""

    def __init__(
        self,
        spike_history: np.ndarray,
        neural_dt_ms: float = 0.1,
        neurons_per_module: int = 20,
        foot_contacts: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        body_positions: Optional[np.ndarray] = None,
        frames: Optional[list[np.ndarray]] = None,
    ) -> None:
        self.spikes = spike_history
        self.neural_dt = neural_dt_ms
        self.n_per_module = neurons_per_module
        self.foot_contacts = foot_contacts
        self.timestamps = timestamps
        self.positions = body_positions
        self.frames = frames or []

        self.neural_analyzer = NeuralAnalyzer(
            spike_history, neural_dt_ms, neurons_per_module
        )

    def generate_summary_figure(
        self,
        save_path: Optional[str] = None,
        time_range_ms: Optional[tuple[float, float]] = None,
    ) -> Figure:
        """Generate a comprehensive summary figure.

        Layout:
          - Top-left: Spike raster
          - Top-right: MuJoCo frame (middle of simulation)
          - Middle: Firing rates
          - Bottom: Gait diagram
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 0.8])

        # Spike raster (top-left)
        ax_raster = fig.add_subplot(gs[0, 0])
        self.neural_analyzer.plot_raster(ax=ax_raster, time_range=time_range_ms)

        # MuJoCo frame (top-right)
        ax_frame = fig.add_subplot(gs[0, 1])
        if self.frames:
            mid_idx = len(self.frames) // 2
            ax_frame.imshow(self.frames[mid_idx])
            ax_frame.set_title("MuJoCo Simulation (mid-point)")
        else:
            ax_frame.text(0.5, 0.5, "No video frames\navailable",
                          transform=ax_frame.transAxes, ha="center", va="center",
                          fontsize=14, color="gray")
            ax_frame.set_title("MuJoCo Simulation")
        ax_frame.axis("off")

        # Firing rates (middle, full width)
        ax_rates = fig.add_subplot(gs[1, :])
        rates = self.neural_analyzer.compute_module_rates(sigma_ms=20.0)
        time_s = self.neural_analyzer.times / 1000.0
        for leg in range(6):
            net_rate = rates[:, leg, 0] - rates[:, leg, 1]
            ax_rates.plot(time_s, net_rate + leg * 150, color=LEG_COLORS[leg],
                          linewidth=0.8, label=LEG_NAMES[leg])
        ax_rates.set_xlabel("Time (s)")
        ax_rates.set_ylabel("Net firing rate (offset)")
        ax_rates.set_title("CPG Module Activity (flexor - extensor)")
        ax_rates.legend(loc="upper right", ncol=6, fontsize=8)

        # Gait diagram (bottom, full width)
        if self.foot_contacts is not None and self.timestamps is not None:
            ax_gait = fig.add_subplot(gs[2, :])
            gait_analyzer = GaitAnalyzer(
                self.foot_contacts, self.timestamps, self.positions
            )
            gait_analyzer.plot_gait_diagram(ax=ax_gait)

        fig.suptitle("Connectome-Driven Embodied Drosophila Simulation", fontsize=14)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Summary figure saved to {save_path}")

        return fig

    def generate_video(
        self,
        save_path: str,
        fps: int = 30,
        duration_s: Optional[float] = None,
        raster_window_ms: float = 500.0,
    ) -> None:
        """Generate side-by-side video: neural activity + fly movement.

        Left panel: scrolling spike raster
        Right panel: MuJoCo rendered frames
        Bottom: rolling gait diagram
        """
        if not self.frames:
            print("No video frames available. Generating neural-only video.")
            self._generate_neural_only_video(save_path, fps, raster_window_ms)
            return

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        n_frames = len(self.frames)
        total_time_ms = self.spikes.shape[0] * self.neural_dt
        neural_steps_per_frame = max(1, self.spikes.shape[0] // n_frames)

        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1])

        ax_raster = fig.add_subplot(gs[0, 0])
        ax_fly = fig.add_subplot(gs[0, 1])
        ax_gait = fig.add_subplot(gs[1, :])

        ax_fly.axis("off")

        def update(frame_idx: int) -> list:
            # Neural timestep corresponding to this frame
            t_neural = frame_idx * neural_steps_per_frame
            t_ms = t_neural * self.neural_dt
            window_start = max(0, t_ms - raster_window_ms)

            # Clear and redraw raster
            ax_raster.clear()
            t_start_step = max(0, int(window_start / self.neural_dt))
            t_end_step = min(t_neural + 1, self.spikes.shape[0])

            for leg in range(6):
                base = leg * self.n_per_module
                for n in range(self.n_per_module):
                    nid = base + n
                    spike_t = np.where(self.spikes[t_start_step:t_end_step, nid] > 0)[0]
                    spike_t_ms = (spike_t + t_start_step) * self.neural_dt
                    if len(spike_t_ms) > 0:
                        ax_raster.scatter(
                            spike_t_ms, np.full_like(spike_t_ms, nid),
                            s=0.3, c=LEG_COLORS[leg], marker="|", linewidths=0.3,
                        )

            ax_raster.set_xlim(window_start, t_ms + 10)
            ax_raster.set_ylabel("Neuron")
            ax_raster.set_xlabel("Time (ms)")
            ax_raster.set_title("CPG Spike Raster")

            # Fly frame
            ax_fly.clear()
            ax_fly.axis("off")
            if frame_idx < len(self.frames):
                ax_fly.imshow(self.frames[frame_idx])
            ax_fly.set_title(f"MuJoCo (t={t_ms / 1000:.3f} s)")

            # Gait diagram (rolling)
            if self.foot_contacts is not None and self.timestamps is not None:
                ax_gait.clear()
                t_phys = t_ms / 1000.0
                t_start_gait = max(0, t_phys - 0.5)

                phys_start = np.searchsorted(self.timestamps, t_start_gait)
                phys_end = np.searchsorted(self.timestamps, t_phys)
                phys_end = min(phys_end, len(self.timestamps) - 1)

                for leg in range(6):
                    contacts = self.foot_contacts[phys_start:phys_end + 1, leg]
                    times = self.timestamps[phys_start:phys_end + 1]

                    in_stance = False
                    starts = []
                    ends = []
                    for ti in range(len(contacts)):
                        if contacts[ti] and not in_stance:
                            starts.append(times[ti])
                            in_stance = True
                        elif not contacts[ti] and in_stance:
                            ends.append(times[ti])
                            in_stance = False
                    if in_stance:
                        ends.append(times[-1] if len(times) > 0 else t_phys)

                    for s, e in zip(starts, ends):
                        ax_gait.barh(5 - leg, e - s, left=s, height=0.6,
                                     color=LEG_COLORS[leg], alpha=0.8)

                ax_gait.set_yticks(range(6))
                ax_gait.set_yticklabels([LEG_NAMES[5 - i] for i in range(6)])
                ax_gait.set_xlabel("Time (s)")
                ax_gait.set_xlim(t_start_gait, t_phys + 0.01)
                ax_gait.set_title("Gait Diagram")

            return []

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

        try:
            writer = FFMpegWriter(fps=fps, metadata={"title": "Embodied Fly Simulation"})
            anim.save(save_path, writer=writer, dpi=100)
            print(f"Video saved to {save_path}")
        except Exception as e:
            print(f"FFMpeg not available, trying pillow writer: {e}")
            try:
                gif_path = save_path.replace(".mp4", ".gif")
                anim.save(gif_path, writer="pillow", fps=fps, dpi=80)
                print(f"GIF saved to {gif_path}")
            except Exception as e2:
                print(f"Could not save video: {e2}")

        plt.close(fig)

    def _generate_neural_only_video(
        self, save_path: str, fps: int, raster_window_ms: float
    ) -> None:
        """Generate a video with neural activity only (no MuJoCo frames)."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        total_ms = self.spikes.shape[0] * self.neural_dt
        n_frames = int(total_ms / 1000.0 * fps)
        ms_per_frame = total_ms / max(1, n_frames)

        fig, (ax_raster, ax_rates) = plt.subplots(2, 1, figsize=(12, 8))

        rates = self.neural_analyzer.compute_module_rates(sigma_ms=20.0)

        def update(frame_idx: int) -> list:
            t_ms = frame_idx * ms_per_frame

            ax_raster.clear()
            t_start = max(0, t_ms - raster_window_ms)
            t_start_step = max(0, int(t_start / self.neural_dt))
            t_end_step = min(int(t_ms / self.neural_dt), self.spikes.shape[0])

            for leg in range(6):
                base = leg * self.n_per_module
                for n in range(self.n_per_module):
                    nid = base + n
                    st = np.where(self.spikes[t_start_step:t_end_step, nid] > 0)[0]
                    st_ms = (st + t_start_step) * self.neural_dt
                    if len(st_ms) > 0:
                        ax_raster.scatter(st_ms, np.full_like(st_ms, nid),
                                          s=0.3, c=LEG_COLORS[leg], marker="|",
                                          linewidths=0.3)

            ax_raster.set_xlim(t_start, t_ms + 10)
            ax_raster.set_ylabel("Neuron")
            ax_raster.set_title(f"CPG Spike Raster (t={t_ms:.0f} ms)")

            ax_rates.clear()
            t_step = min(int(t_ms / self.neural_dt), self.spikes.shape[0] - 1)
            time_s = self.neural_analyzer.times[:t_step] / 1000.0
            for leg in range(6):
                net = rates[:t_step, leg, 0] - rates[:t_step, leg, 1]
                ax_rates.plot(time_s, net, color=LEG_COLORS[leg],
                              linewidth=0.8, label=LEG_NAMES[leg])
            ax_rates.set_xlabel("Time (s)")
            ax_rates.set_ylabel("Net rate (Hz)")
            ax_rates.set_title("Module Activity")
            if frame_idx == 0:
                ax_rates.legend(ncol=6, fontsize=8)

            return []

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=100)
            print(f"Neural video saved to {save_path}")
        except Exception:
            gif_path = save_path.replace(".mp4", ".gif")
            try:
                anim.save(gif_path, writer="pillow", fps=min(fps, 10), dpi=60)
                print(f"Neural GIF saved to {gif_path}")
            except Exception as e:
                print(f"Could not save video: {e}")

        plt.close(fig)

    def generate_analysis_plots(self, output_dir: str) -> None:
        """Generate and save all analysis plots.

        Args:
            output_dir: Directory to save plot files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Spike raster
        fig = self.neural_analyzer.plot_raster()
        fig.savefig(os.path.join(output_dir, "spike_raster.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Firing rates
        fig = self.neural_analyzer.plot_firing_rates()
        fig.savefig(os.path.join(output_dir, "firing_rates.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 3. Phase relationships
        fig = self.neural_analyzer.plot_phase_relationships()
        fig.savefig(os.path.join(output_dir, "phase_relationships.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4. Cross-correlograms
        fig = self.neural_analyzer.plot_cross_correlograms()
        fig.savefig(os.path.join(output_dir, "cross_correlograms.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 5. Gait diagram
        if self.foot_contacts is not None and self.timestamps is not None:
            gait = GaitAnalyzer(self.foot_contacts, self.timestamps, self.positions)
            fig = gait.plot_gait_diagram()
            fig.savefig(os.path.join(output_dir, "gait_diagram.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 6. Body trajectory
            fig = gait.plot_body_trajectory()
            if fig:
                fig.savefig(os.path.join(output_dir, "body_trajectory.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

        # 7. Summary figure
        self.generate_summary_figure(
            save_path=os.path.join(output_dir, "summary.png")
        )

        print(f"All analysis plots saved to {output_dir}")
