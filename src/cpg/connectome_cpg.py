"""Connectome-inspired CPG wiring based on Drosophila VNC motor circuits.

This module builds CPG weight matrices using connectivity patterns inspired
by the FlyWire connectome (Dorkenwald et al., 2024). Rather than using
hand-tuned weights, the adjacency matrix is constructed from biological
constraints:

  - Neuron type ratios: MNs (motor neurons), excitatory INs, inhibitory INs
  - Connection probabilities derived from observed VNC circuit motifs
  - Synaptic weight distributions based on synapse counts

BIOLOGICAL GROUNDING vs SIMPLIFICATIONS:
  - Grounded: E/I ratios (~35% inhibitory), contralateral inhibition motif,
    local recurrent excitation, motor neuron output layer
  - Simplified: Actual VNC has thousands of neurons per segment; we use 20.
    Real synaptic weights come from synapse counts; we use scaled values.
    Proprioceptive feedback is omitted (open-loop CPG only).
    Neuromodulation (e.g., octopamine for speed control) is not modeled.

References:
  - Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain.
    Nature, 634, 124-138.
  - Azevedo, A. et al. (2024). Connectomic reconstruction of a female-brain
    Drosophila ventral nerve cord. Nature, 631, 360-368.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .spiking_cpg import SpikingCPG, CPGConfig, LEG_NAMES


@dataclass
class ConnectomeConfig:
    """Configuration for connectome-inspired wiring."""

    # Neuron type fractions per module
    mn_fraction: float = 0.3       # Motor neurons
    exc_in_fraction: float = 0.35  # Excitatory interneurons
    inh_in_fraction: float = 0.35  # Inhibitory interneurons

    # Connection probabilities (from VNC circuit motifs)
    p_local: float = 0.4           # Within-module connection probability
    p_contralateral: float = 0.15  # Between contralateral legs
    p_ipsilateral: float = 0.2     # Between ipsilateral neighbors

    # Weight scaling
    weight_scale: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "ConnectomeConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ConnectomeCPG(SpikingCPG):
    """CPG with connectivity derived from Drosophila VNC motor circuit data.

    Extends SpikingCPG by replacing the hand-tuned weight matrix with one
    constructed from biological connectivity rules. The neuron population
    in each module is divided into three types:

      - Motor Neurons (MN): output neurons that drive joints. Receive
        excitatory input from INs. ~30% of module.
      - Excitatory Interneurons (eIN): provide recurrent excitation and
        inter-segment coupling. ~35% of module.
      - Inhibitory Interneurons (iIN): provide local and contralateral
        inhibition. ~35% of module. These form the half-center oscillator
        via mutual inhibition.
    """

    def __init__(
        self,
        cpg_config: Optional[CPGConfig] = None,
        connectome_config: Optional[ConnectomeConfig] = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.connectome_config = connectome_config or ConnectomeConfig()
        self.seed = seed

        # Compute neuron type assignments before parent __init__
        # (parent __init__ calls _build_weight_matrix)
        self._cpg_cfg = cpg_config or CPGConfig()
        self._assign_neuron_types()

        # Now call parent init which will call _build_weight_matrix
        super().__init__(config=self._cpg_cfg, device=device)

    def _assign_neuron_types(self) -> None:
        """Assign each neuron a type: 'mn', 'exc_in', or 'inh_in'."""
        cfg = self._cpg_cfg
        cc = self.connectome_config
        n = cfg.neurons_per_module

        n_mn = max(1, int(n * cc.mn_fraction))
        n_exc = max(1, int(n * cc.exc_in_fraction))
        n_inh = n - n_mn - n_exc  # remainder

        # Type labels for a single module
        module_types = (
            ["mn"] * n_mn
            + ["exc_in"] * n_exc
            + ["inh_in"] * n_inh
        )

        # Repeat for all legs
        self.neuron_types: list[str] = module_types * cfg.num_legs
        self.n_mn = n_mn
        self.n_exc = n_exc
        self.n_inh = n_inh

    def _build_weight_matrix(self) -> torch.Tensor:
        """Build weight matrix from connectome-inspired rules.

        Connection rules (inspired by VNC circuit motifs):
          1. Local circuit (within module):
             - eIN → eIN: recurrent excitation (sustains activity)
             - eIN → MN: excitatory drive to motor output
             - iIN → eIN: inhibition (half-center oscillator)
             - iIN → MN: inhibitory gating of motor output
             - eIN → iIN: excitation of inhibitory pool (delayed inhibition)
          2. Contralateral (left↔right same segment):
             - iIN → contralateral eIN: mutual inhibition for alternation
          3. Ipsilateral (front↔back same side):
             - eIN → neighboring eIN: wave propagation
        """
        rng = np.random.RandomState(self.seed)
        cfg = self.config
        cc = self.connectome_config
        N = self.N
        n = cfg.neurons_per_module

        W = np.zeros((N, N), dtype=np.float32)

        # --- Local connections within each module ---
        for leg in range(cfg.num_legs):
            base = leg * n
            W = self._wire_local_circuit(W, base, rng)

        # --- Contralateral inhibition ---
        contra_pairs = [(0, 3), (1, 4), (2, 5)]  # L1-R1, L2-R2, L3-R3
        for left, right in contra_pairs:
            W = self._wire_contralateral(W, left, right, rng)
            W = self._wire_contralateral(W, right, left, rng)

        # --- Ipsilateral coupling ---
        ipsi_pairs = [(0, 1), (1, 2), (3, 4), (4, 5)]
        for front, back in ipsi_pairs:
            W = self._wire_ipsilateral(W, front, back, rng)
            W = self._wire_ipsilateral(W, back, front, rng, backward=True)

        # --- Tripod synchronization ---
        # Within-tripod excitatory coupling between iIN pools
        # This helps synchronize legs within the same tripod group
        tripod_a = [0, 4, 2]  # L1, R2, L3
        tripod_b = [3, 1, 5]  # R1, L2, R3
        for group in [tripod_a, tripod_b]:
            for i in range(len(group)):
                for j in range(len(group)):
                    if i != j:
                        W = self._wire_tripod_sync(W, group[i], group[j], rng)

        # Scale and convert
        W *= cc.weight_scale
        np.fill_diagonal(W, 0.0)

        return torch.tensor(W, device=self.device)

    def _get_type_indices(self, base: int) -> dict[str, np.ndarray]:
        """Get neuron indices by type within a module starting at `base`."""
        n = self.config.neurons_per_module
        indices = np.arange(base, base + n)
        types = self.neuron_types[base:base + n]
        return {
            "mn": indices[np.array([t == "mn" for t in types])],
            "exc_in": indices[np.array([t == "exc_in" for t in types])],
            "inh_in": indices[np.array([t == "inh_in" for t in types])],
        }

    def _wire_local_circuit(
        self, W: np.ndarray, base: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Wire local circuit within a single leg module."""
        cc = self.connectome_config
        idx = self._get_type_indices(base)

        exc = idx["exc_in"]
        inh = idx["inh_in"]
        mn = idx["mn"]

        # eIN → eIN: recurrent excitation
        mask = rng.random((len(exc), len(exc))) < cc.p_local
        for i, src in enumerate(exc):
            for j, dst in enumerate(exc):
                if mask[i, j] and src != dst:
                    W[dst, src] = rng.uniform(0.2, 0.5)

        # eIN → MN: excitatory drive
        mask = rng.random((len(exc), len(mn))) < cc.p_local * 1.2
        for i, src in enumerate(exc):
            for j, dst in enumerate(mn):
                if mask[i, j]:
                    W[dst, src] = rng.uniform(0.3, 0.8)

        # iIN → eIN: half-center inhibition (strong)
        mask = rng.random((len(inh), len(exc))) < cc.p_local * 1.5
        for i, src in enumerate(inh):
            for j, dst in enumerate(exc):
                if mask[i, j]:
                    W[dst, src] = -rng.uniform(0.5, 1.2)

        # iIN → MN: inhibitory gating
        mask = rng.random((len(inh), len(mn))) < cc.p_local * 0.8
        for i, src in enumerate(inh):
            for j, dst in enumerate(mn):
                if mask[i, j]:
                    W[dst, src] = -rng.uniform(0.3, 0.7)

        # eIN → iIN: excitation of inhibitory pool
        mask = rng.random((len(exc), len(inh))) < cc.p_local
        for i, src in enumerate(exc):
            for j, dst in enumerate(inh):
                if mask[i, j]:
                    W[dst, src] = rng.uniform(0.3, 0.6)

        # iIN → iIN: recurrent inhibition (mutual inhibition within inh pool)
        mask = rng.random((len(inh), len(inh))) < cc.p_local * 0.5
        for i, src in enumerate(inh):
            for j, dst in enumerate(inh):
                if mask[i, j] and src != dst:
                    W[dst, src] = -rng.uniform(0.2, 0.4)

        return W

    def _wire_contralateral(
        self, W: np.ndarray, src_leg: int, dst_leg: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Wire contralateral inhibition between two legs."""
        cc = self.connectome_config
        src_idx = self._get_type_indices(src_leg * self.config.neurons_per_module)
        dst_idx = self._get_type_indices(dst_leg * self.config.neurons_per_module)

        # iIN(src) → eIN(dst): contralateral inhibition
        src_inh = src_idx["inh_in"]
        dst_exc = dst_idx["exc_in"]

        mask = rng.random((len(src_inh), len(dst_exc))) < cc.p_contralateral
        for i, src in enumerate(src_inh):
            for j, dst in enumerate(dst_exc):
                if mask[i, j]:
                    W[dst, src] = -rng.uniform(0.3, 0.6)

        return W

    def _wire_ipsilateral(
        self,
        W: np.ndarray,
        src_leg: int,
        dst_leg: int,
        rng: np.random.RandomState,
        backward: bool = False,
    ) -> np.ndarray:
        """Wire ipsilateral coupling (front→back excitation)."""
        cc = self.connectome_config
        n = self.config.neurons_per_module
        src_idx = self._get_type_indices(src_leg * n)
        dst_idx = self._get_type_indices(dst_leg * n)

        # eIN(src) → eIN(dst): excitatory wave propagation
        src_exc = src_idx["exc_in"]
        dst_exc = dst_idx["exc_in"]

        p = cc.p_ipsilateral * (0.5 if backward else 1.0)
        mask = rng.random((len(src_exc), len(dst_exc))) < p
        for i, src in enumerate(src_exc):
            for j, dst in enumerate(dst_exc):
                if mask[i, j]:
                    weight = rng.uniform(0.1, 0.3) if backward else rng.uniform(0.2, 0.5)
                    W[dst, src] = weight

        return W

    def _wire_tripod_sync(
        self, W: np.ndarray, src_leg: int, dst_leg: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Wire weak synchronizing connections within a tripod group."""
        n = self.config.neurons_per_module
        src_idx = self._get_type_indices(src_leg * n)
        dst_idx = self._get_type_indices(dst_leg * n)

        # eIN(src) → eIN(dst): weak synchronization
        src_exc = src_idx["exc_in"]
        dst_exc = dst_idx["exc_in"]

        mask = rng.random((len(src_exc), len(dst_exc))) < 0.1
        for i, src in enumerate(src_exc):
            for j, dst in enumerate(dst_exc):
                if mask[i, j]:
                    W[dst, src] = rng.uniform(0.05, 0.15)

        return W

    def get_neuron_type_summary(self) -> dict[str, int]:
        """Return count of each neuron type."""
        from collections import Counter
        return dict(Counter(self.neuron_types))

    def get_weight_statistics(self) -> dict[str, float]:
        """Return statistics about the weight matrix."""
        W = self.W.cpu().numpy()
        nonzero = W[W != 0]
        exc = nonzero[nonzero > 0]
        inh = nonzero[nonzero < 0]
        return {
            "total_connections": int(np.count_nonzero(W)),
            "density": float(np.count_nonzero(W)) / (W.shape[0] * W.shape[1]),
            "n_excitatory": len(exc),
            "n_inhibitory": len(inh),
            "ei_ratio": len(exc) / max(1, len(inh)),
            "mean_exc_weight": float(exc.mean()) if len(exc) > 0 else 0.0,
            "mean_inh_weight": float(inh.mean()) if len(inh) > 0 else 0.0,
        }
