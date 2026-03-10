"""Neural controller: maps CPG spike output to joint angle targets.

Uses push-pull decoding: each leg module's flexor and extensor pools
encode opposing joint movements. The difference in firing rates between
the two pools determines the joint angle.

Pipeline:
    Spikes → Sliding window rate → Push-pull difference → Gain + offset → Joint angles

The resting pose offsets are taken from NeuroMechFly v2's "stretch" init pose,
which differs per leg segment (front/mid/hind have different Coxa_roll values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from ..cpg.spiking_cpg import SpikingCPG, CPGConfig
except (ImportError, ValueError):
    from cpg.spiking_cpg import SpikingCPG, CPGConfig


# NeuroMechFly v2 "tripod" standing pose initial joint angles (radians)
# Captured from FlyGym's tripod init_pose — this is a stable standing position.
# Order per leg: Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1
REST_POSE: dict[str, list[float]] = {
    # Left front (L1)
    "LF": [-0.014, 0.0, 0.003, -1.179, -0.002, 0.758, -0.142],
    # Left mid (L2)
    "LM": [0.036, 1.778, 0.219, -1.675, -0.003, 1.764, -0.120],
    # Left hind (L3)
    "LH": [0.021, 2.402, 0.187, -1.560, -0.141, 1.148, -0.106],
    # Right front (R1)
    "RF": [-0.001, 0.001, 0.0, -1.311, 0.0, 0.896, -0.172],
    # Right mid (R2)
    "RM": [0.012, -1.826, -0.106, -1.836, 0.002, 1.728, -0.033],
    # Right hind (R3)
    "RH": [0.280, -2.442, -0.138, -1.316, -0.267, 1.145, -0.384],
}

# Leg order matching CPG module indices: L1, L2, L3, R1, R2, R3
LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]

# Which joints should be driven by the CPG (amplitude > 0)
# and which should stay at rest pose (amplitude = 0)
# Index: Coxa(0), Coxa_roll(1), Coxa_yaw(2), Femur(3), Femur_roll(4), Tibia(5), Tarsus(6)
# For walking: Coxa drives protraction/retraction, Femur drives levation/depression,
# Tibia drives extension/flexion
JOINT_AMPLITUDES = [0.5, 0.0, 0.15, 0.6, 0.0, 0.5, 0.15]

# Phase offsets per joint within a leg cycle (radians)
# Coxa and Femur should be ~90° offset (lift during forward swing)
JOINT_PHASE_OFFSETS = [0.0, 0.0, 0.0, np.pi / 2, 0.0, np.pi / 3, np.pi / 4]


@dataclass
class ControllerConfig:
    """Configuration for the neural controller / spike decoder."""

    decode_window: float = 50.0      # Sliding window for rate decoding (ms)
    joints_per_leg: int = 7          # DoF per leg in FlyGym

    # Gain per joint type (maps normalized rate difference → radians)
    gains: list[float] = field(default_factory=lambda: list(JOINT_AMPLITUDES))

    # Phase offsets per joint within a leg cycle (radians)
    joint_phase_offsets: list[float] = field(default_factory=lambda: list(JOINT_PHASE_OFFSETS))

    @classmethod
    def from_dict(cls, d: dict) -> "ControllerConfig":
        fields = cls.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k in fields:
                kwargs[k] = v
        return cls(**kwargs)


class NeuralController:
    """Decodes CPG spike trains into continuous joint angle targets.

    For each leg module, the controller:
      1. Counts spikes in flexor and extensor pools over a sliding window
      2. Computes the push-pull difference: rate_flex - rate_ext
      3. Normalizes to [-1, 1] range
      4. Applies per-joint amplitudes and phase offsets around the rest pose
    """

    def __init__(
        self,
        cpg: SpikingCPG,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.cpg = cpg
        self.config = config or ControllerConfig()

        cfg = cpg.config
        self.neurons_per_module = cfg.neurons_per_module
        self.half_module = cfg.half_module
        self.num_legs = cfg.num_legs

        # Spike buffer for sliding window
        self.window_steps = max(1, int(self.config.decode_window / cfg.dt))
        self.spike_buffer: list[np.ndarray] = []

        # Total joints
        self.n_joints = self.num_legs * self.config.joints_per_leg

        # Build per-joint rest pose array (42,) from the per-leg rest poses
        self.rest_pose = np.zeros(self.n_joints)
        for leg_idx, leg_name in enumerate(LEG_ORDER):
            base = leg_idx * self.config.joints_per_leg
            self.rest_pose[base:base + self.config.joints_per_leg] = REST_POSE[leg_name]

    def set_rest_pose(self, joint_angles: np.ndarray) -> None:
        """Override rest pose with actual initial joint angles from simulation."""
        assert joint_angles.shape == (self.n_joints,)
        self.rest_pose = joint_angles.copy()

    def reset(self) -> None:
        """Clear the spike buffer."""
        self.spike_buffer.clear()

    def decode_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Decode a single timestep of spikes into joint angle targets.

        Args:
            spikes: Binary spike vector of shape (N,) from the CPG.

        Returns:
            Joint angle targets of shape (42,) in radians.
        """
        # Add to sliding window buffer
        self.spike_buffer.append(spikes.copy())
        if len(self.spike_buffer) > self.window_steps:
            self.spike_buffer.pop(0)

        # Stack buffer into array
        buffer = np.array(self.spike_buffer)  # (window, N)

        # Start with rest pose
        joint_targets = self.rest_pose.copy()

        for leg in range(self.num_legs):
            base_neuron = leg * self.neurons_per_module
            base_joint = leg * self.config.joints_per_leg

            # Flexor and extensor spike counts over the window
            flex_spikes = buffer[:, base_neuron:base_neuron + self.half_module]
            ext_spikes = buffer[:, base_neuron + self.half_module:
                                base_neuron + self.neurons_per_module]

            # Average firing rates
            flex_rate = flex_spikes.mean()
            ext_rate = ext_spikes.mean()

            # Push-pull: normalized difference in [-1, 1]
            total = flex_rate + ext_rate + 1e-8
            push_pull = (flex_rate - ext_rate) / total

            # Map to each joint with individual amplitudes around rest pose
            for j in range(self.config.joints_per_leg):
                joint_idx = base_joint + j
                amplitude = self.config.gains[j]

                if amplitude == 0.0:
                    continue  # keep rest pose

                phase = self.config.joint_phase_offsets[j]

                # Apply phase offset via rotation of the push-pull signal
                if phase != 0.0:
                    signal = push_pull * np.cos(phase) - (1.0 - abs(push_pull)) * np.sin(phase)
                else:
                    signal = push_pull

                joint_targets[joint_idx] += amplitude * signal

        return joint_targets

    def decode_spike_batch(self, spike_sequence: np.ndarray) -> np.ndarray:
        """Decode a batch of spike timesteps.

        Args:
            spike_sequence: Spike matrix of shape (T, N).

        Returns:
            Joint angle targets of shape (T, 42).
        """
        T = spike_sequence.shape[0]
        targets = np.zeros((T, self.n_joints))
        for t in range(T):
            targets[t] = self.decode_spikes(spike_sequence[t])
        return targets


class SineWaveController:
    """Baseline controller using simple sine waves (no neural network).

    Generates coordinated leg movements using sinusoidal oscillators
    with fixed phase relationships for tripod gait.
    """

    def __init__(
        self,
        frequency: float = 10.0,
        dt_ms: float = 0.1,
        joints_per_leg: int = 7,
        num_legs: int = 6,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.frequency = frequency
        self.dt_ms = dt_ms
        self.joints_per_leg = joints_per_leg
        self.num_legs = num_legs
        self.config = config or ControllerConfig()
        self.n_joints = num_legs * joints_per_leg
        self.t = 0.0  # current time in ms

        # Build rest pose
        self.rest_pose = np.zeros(self.n_joints)
        for leg_idx, leg_name in enumerate(LEG_ORDER):
            base = leg_idx * joints_per_leg
            self.rest_pose[base:base + joints_per_leg] = REST_POSE[leg_name]

        # Phase offsets for tripod gait
        self.leg_phases = np.array([
            0.0,      # L1 — tripod A
            np.pi,    # L2 — tripod B
            0.0,      # L3 — tripod A
            np.pi,    # R1 — tripod B
            0.0,      # R2 — tripod A
            np.pi,    # R3 — tripod B
        ])

    def reset(self) -> None:
        self.t = 0.0

    def step(self) -> np.ndarray:
        """Generate one timestep of joint angle targets.

        Returns:
            Joint angle targets of shape (42,).
        """
        omega = 2 * np.pi * self.frequency
        t_sec = self.t * 1e-3

        targets = self.rest_pose.copy()

        for leg in range(self.num_legs):
            phase = omega * t_sec + self.leg_phases[leg]
            base = leg * self.joints_per_leg

            for j in range(self.joints_per_leg):
                amplitude = self.config.gains[j]
                if amplitude == 0.0:
                    continue
                jp = self.config.joint_phase_offsets[j]
                targets[base + j] += amplitude * np.sin(phase + jp)

        self.t += self.dt_ms
        return targets


class RandomController:
    """Baseline controller with random noise (no coordination)."""

    def __init__(
        self,
        amplitude: float = 0.3,
        joints_per_leg: int = 7,
        num_legs: int = 6,
        config: Optional[ControllerConfig] = None,
        seed: int = 123,
    ) -> None:
        self.amplitude = amplitude
        self.joints_per_leg = joints_per_leg
        self.num_legs = num_legs
        self.config = config or ControllerConfig()
        self.n_joints = num_legs * joints_per_leg
        self.rng = np.random.RandomState(seed)

        # Build rest pose
        self.rest_pose = np.zeros(self.n_joints)
        for leg_idx, leg_name in enumerate(LEG_ORDER):
            base = leg_idx * joints_per_leg
            self.rest_pose[base:base + joints_per_leg] = REST_POSE[leg_name]

    def reset(self) -> None:
        pass

    def step(self) -> np.ndarray:
        """Generate random joint angle targets.

        Returns:
            Joint angle targets of shape (42,).
        """
        noise = self.rng.randn(self.n_joints) * self.amplitude
        # Only apply noise to joints that have nonzero amplitude
        for leg in range(self.num_legs):
            base = leg * self.joints_per_leg
            for j in range(self.joints_per_leg):
                if self.config.gains[j] == 0.0:
                    noise[base + j] = 0.0
        return self.rest_pose + noise
