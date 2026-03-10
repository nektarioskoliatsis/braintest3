"""Spiking neural CPG using Leaky Integrate-and-Fire (LIF) neurons.

Implements a half-center oscillator architecture: each leg module has two
mutually inhibiting neuron pools (flexor and extensor) that produce
anti-phase rhythmic activity. Inter-module coupling produces coordinated
multi-leg gaits (e.g., tripod gait in insects).

Dynamics per neuron:
    tau_m * dV/dt = -(V - V_rest) + R_m * (I_syn + I_ext + I_noise)
    When V >= V_thresh: spike, V -> V_reset, refractory for t_refrac

All simulation is implemented in pure PyTorch for GPU compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np


# Leg identifiers and their indices
LEG_NAMES: list[str] = ["L1", "L2", "L3", "R1", "R2", "R3"]

# Tripod groups: legs that move together
TRIPOD_A: list[int] = [0, 3, 4]  # L1, R2, L3  (indices into LEG_NAMES... wait)
# Actually: L1=0, L2=1, L3=2, R1=3, R2=4, R3=5
# Tripod A: L1(0), R2(4), L3(2)
# Tripod B: R1(3), L2(1), R3(5)
TRIPOD_A_INDICES: list[int] = [0, 4, 2]  # L1, R2, L3
TRIPOD_B_INDICES: list[int] = [3, 1, 5]  # R1, L2, R3

# Contralateral pairs: (left, right) legs on the same segment
CONTRALATERAL_PAIRS: list[tuple[int, int]] = [(0, 3), (1, 4), (2, 5)]

# Ipsilateral neighbors: sequential legs on the same side
IPSILATERAL_PAIRS: list[tuple[int, int]] = [
    (0, 1), (1, 2),  # Left: L1→L2→L3
    (3, 4), (4, 5),  # Right: R1→R2→R3
]


@dataclass
class CPGConfig:
    """Configuration for the spiking CPG network."""

    neurons_per_module: int = 20
    num_legs: int = 6

    # LIF parameters
    tau_m: float = 10.0        # ms
    v_rest: float = -65.0      # mV
    v_thresh: float = -50.0    # mV
    v_reset: float = -70.0     # mV
    t_refrac: float = 2.0      # ms
    R_m: float = 10.0          # MOhm

    # Synaptic time constants
    tau_syn_exc: float = 5.0   # ms
    tau_syn_inh: float = 10.0  # ms

    # Connection weights
    w_exc_self: float = 3.0
    w_inh_cross: float = -6.0
    w_contralateral: float = -2.5
    w_ipsilateral: float = 1.5

    # Drive
    I_drive: float = 5.0       # nA
    noise_std: float = 1.0     # nA
    oscillation_freq: float = 10.0  # Hz

    # Simulation
    dt: float = 0.1            # ms

    @property
    def total_neurons(self) -> int:
        return self.neurons_per_module * self.num_legs

    @property
    def half_module(self) -> int:
        """Neurons per half-center (flexor or extensor pool)."""
        return self.neurons_per_module // 2

    @classmethod
    def from_dict(cls, d: dict) -> "CPGConfig":
        """Create config from a dictionary (e.g., parsed YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SpikingCPG:
    """Spiking neural network CPG with half-center oscillator architecture.

    Each of the 6 leg modules contains two pools of neurons:
      - Flexor pool (first half): drives swing phase
      - Extensor pool (second half): drives stance phase

    Mutual inhibition between flexor and extensor creates oscillation.
    Inter-module coupling creates coordinated gaits.
    """

    def __init__(self, config: Optional[CPGConfig] = None, device: str = "cpu") -> None:
        self.config = config or CPGConfig()
        self.device = torch.device(device)
        self.N = self.config.total_neurons

        # Build weight matrix
        self.W = self._build_weight_matrix()

        # State variables
        self.reset()

    def _build_weight_matrix(self) -> torch.Tensor:
        """Build the synaptic weight matrix for the CPG network.

        Architecture per module (neurons_per_module = 20):
          - Neurons 0..9: Flexor pool (excitatory internal, inhibits extensor)
          - Neurons 10..19: Extensor pool (excitatory internal, inhibits flexor)
        """
        cfg = self.config
        N = self.N
        n = cfg.neurons_per_module
        half = cfg.half_module

        W = torch.zeros(N, N, device=self.device)

        for leg in range(cfg.num_legs):
            base = leg * n

            # Intra-pool excitatory recurrence (flexor excites flexor, extensor excites extensor)
            flex_slice = slice(base, base + half)
            ext_slice = slice(base + half, base + n)

            W[flex_slice, flex_slice] = cfg.w_exc_self / half
            W[ext_slice, ext_slice] = cfg.w_exc_self / half

            # Cross-inhibition: flexor inhibits extensor and vice versa
            W[ext_slice, flex_slice] = cfg.w_inh_cross / half
            W[flex_slice, ext_slice] = cfg.w_inh_cross / half

        # Inter-module coupling: contralateral inhibition
        for left, right in CONTRALATERAL_PAIRS:
            self._add_intermodule_connection(W, left, right, cfg.w_contralateral)
            self._add_intermodule_connection(W, right, left, cfg.w_contralateral)

        # Inter-module coupling: ipsilateral excitatory coupling
        for front, back in IPSILATERAL_PAIRS:
            self._add_intermodule_connection(W, front, back, cfg.w_ipsilateral)
            self._add_intermodule_connection(W, back, front, cfg.w_ipsilateral * 0.5)

        # Tripod coupling: synchronize within-tripod legs via weak excitation
        # Flexor→Flexor coupling within the same tripod group
        tripod_w = cfg.w_ipsilateral * 0.3
        for group in [TRIPOD_A_INDICES, TRIPOD_B_INDICES]:
            for i in range(len(group)):
                for j in range(len(group)):
                    if i != j:
                        self._add_intermodule_connection(
                            W, group[i], group[j], tripod_w, same_phase=True
                        )

        # Zero diagonal (no self-connections)
        W.fill_diagonal_(0.0)

        return W

    def _add_intermodule_connection(
        self,
        W: torch.Tensor,
        src_leg: int,
        dst_leg: int,
        weight: float,
        same_phase: bool = False,
    ) -> None:
        """Add connections between two leg modules.

        Args:
            W: Weight matrix to modify in-place.
            src_leg: Source leg index.
            dst_leg: Destination leg index.
            weight: Connection weight (positive=excitatory, negative=inhibitory).
            same_phase: If True, connect same pools (flex→flex). If False,
                        connect cross-pools (flex→ext) for anti-phase coupling.
        """
        n = self.config.neurons_per_module
        half = self.config.half_module

        src_base = src_leg * n
        dst_base = dst_leg * n

        if same_phase:
            # Flexor → Flexor, Extensor → Extensor
            W[dst_base:dst_base + half, src_base:src_base + half] = weight / half
            W[dst_base + half:dst_base + n, src_base + half:src_base + n] = weight / half
        else:
            # Flexor → Extensor, Extensor → Flexor (anti-phase)
            W[dst_base + half:dst_base + n, src_base:src_base + half] = weight / half
            W[dst_base:dst_base + half, src_base + half:src_base + n] = weight / half

    def reset(self) -> None:
        """Reset all state variables to initial conditions."""
        cfg = self.config
        N = self.N

        # Membrane potentials: start near rest with small random perturbation
        self.V = torch.full((N,), cfg.v_rest, device=self.device)
        # Break symmetry: give tripod A flexors a slight boost
        for leg in TRIPOD_A_INDICES:
            base = leg * cfg.neurons_per_module
            self.V[base:base + cfg.half_module] += 5.0  # mV boost

        # Synaptic currents (excitatory and inhibitory traces)
        self.I_syn = torch.zeros(N, device=self.device)

        # Refractory counters (in timesteps)
        self.refrac_counter = torch.zeros(N, dtype=torch.int32, device=self.device)
        self.refrac_steps = int(cfg.t_refrac / cfg.dt)

        # Spike recording
        self.spike_history: list[torch.Tensor] = []
        self.time_steps: int = 0

    def step(self, I_ext: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance the CPG network by one timestep.

        Args:
            I_ext: Optional external current injection (N,). If None, uses
                   default tonic drive.

        Returns:
            Binary spike vector (N,) for this timestep.
        """
        cfg = self.config
        dt = cfg.dt
        N = self.N

        # Default external drive: tonic current to all neurons
        if I_ext is None:
            I_ext = torch.full((N,), cfg.I_drive, device=self.device)

        # Gaussian noise
        I_noise = torch.randn(N, device=self.device) * cfg.noise_std

        # Total input current
        I_total = self.I_syn + I_ext + I_noise

        # LIF membrane dynamics (Euler integration)
        # tau_m * dV/dt = -(V - V_rest) + R_m * I_total
        not_refractory = (self.refrac_counter == 0).float()
        dV = (-(self.V - cfg.v_rest) + cfg.R_m * I_total) * (dt / cfg.tau_m)
        self.V = self.V + dV * not_refractory

        # Spike detection
        spikes = (self.V >= cfg.v_thresh).float()

        # Reset spiking neurons
        self.V = torch.where(spikes.bool(), torch.tensor(cfg.v_reset, device=self.device), self.V)

        # Set refractory period
        self.refrac_counter = torch.where(
            spikes.bool(),
            torch.tensor(self.refrac_steps, dtype=torch.int32, device=self.device),
            self.refrac_counter,
        )
        self.refrac_counter = torch.clamp(self.refrac_counter - 1, min=0)

        # Update synaptic currents: I_syn decays and receives spikes through W
        # Use a single decay constant (average of exc/inh) for simplicity,
        # but separate weights carry the sign
        tau_avg = (cfg.tau_syn_exc + cfg.tau_syn_inh) / 2.0
        decay = torch.exp(torch.tensor(-dt / tau_avg, device=self.device))
        self.I_syn = self.I_syn * decay + self.W @ spikes

        # Record
        self.spike_history.append(spikes.cpu())
        self.time_steps += 1

        return spikes

    def run(self, duration_ms: float, I_ext: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the CPG for a specified duration.

        Args:
            duration_ms: Simulation duration in milliseconds.
            I_ext: Optional constant external current (N,).

        Returns:
            Spike matrix of shape (T, N) where T = duration_ms / dt.
        """
        n_steps = int(duration_ms / self.config.dt)
        spikes = torch.zeros(n_steps, self.N, device=self.device)

        for t in range(n_steps):
            spikes[t] = self.step(I_ext)

        return spikes

    def get_spike_history(self) -> np.ndarray:
        """Return spike history as a numpy array of shape (T, N)."""
        if not self.spike_history:
            return np.empty((0, self.N))
        return torch.stack(self.spike_history).numpy()

    def get_module_spike_rates(
        self, window_ms: float = 50.0
    ) -> np.ndarray:
        """Compute firing rates per module over a sliding window.

        Returns:
            Array of shape (T, num_legs, 2) where the last dim is
            [flexor_rate, extensor_rate] in Hz.
        """
        spikes = self.get_spike_history()  # (T, N)
        if spikes.shape[0] == 0:
            return np.empty((0, self.config.num_legs, 2))

        cfg = self.config
        n = cfg.neurons_per_module
        half = cfg.half_module
        window_steps = max(1, int(window_ms / cfg.dt))

        T = spikes.shape[0]
        rates = np.zeros((T, cfg.num_legs, 2))

        # Cumulative sum for efficient windowed rate computation
        for leg in range(cfg.num_legs):
            base = leg * n
            flex_spikes = spikes[:, base:base + half]
            ext_spikes = spikes[:, base + half:base + n]

            # Sum spikes per pool per timestep
            flex_sum = flex_spikes.sum(axis=1)
            ext_sum = ext_spikes.sum(axis=1)

            # Sliding window average → firing rate
            for t in range(T):
                t_start = max(0, t - window_steps + 1)
                window_len = t - t_start + 1
                rates[t, leg, 0] = flex_sum[t_start:t + 1].sum() / (window_len * cfg.dt * 1e-3 * half)
                rates[t, leg, 1] = ext_sum[t_start:t + 1].sum() / (window_len * cfg.dt * 1e-3 * half)

        return rates

    def get_module_phases(self) -> np.ndarray:
        """Estimate instantaneous phase of each leg module using flexor/extensor balance.

        Returns:
            Phase array of shape (T, num_legs) in radians [0, 2*pi].
        """
        spikes = self.get_spike_history()
        if spikes.shape[0] == 0:
            return np.empty((0, self.config.num_legs))

        cfg = self.config
        n = cfg.neurons_per_module
        half = cfg.half_module
        T = spikes.shape[0]

        # Smooth spike trains with a Gaussian kernel
        from scipy.ndimage import gaussian_filter1d
        sigma = int(20.0 / cfg.dt)  # 20 ms smoothing

        phases = np.zeros((T, cfg.num_legs))
        for leg in range(cfg.num_legs):
            base = leg * n
            flex_rate = gaussian_filter1d(spikes[:, base:base + half].sum(axis=1).astype(float), sigma)
            ext_rate = gaussian_filter1d(spikes[:, base + half:base + n].sum(axis=1).astype(float), sigma)

            # Phase from arctan2 of smoothed flexor/extensor activity
            phases[:, leg] = np.arctan2(flex_rate - ext_rate, flex_rate + ext_rate + 1e-8) + np.pi

        return phases
