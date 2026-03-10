"""FlyGym environment wrapper for NeuroMechFly v2.

Handles initialization of the fly model, physics stepping, recording of
simulation data, and video rendering. Bridges the gap between neural
simulation timestep (0.1 ms) and physics timestep (0.1 ms default in FlyGym).

Reference:
  Wang-Chen, S. et al. (2024). NeuroMechFly v2: simulating embodied
  sensorimotor control in adult Drosophila. Nature Methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    from flygym import Fly, Camera, SingleFlySimulation
    from flygym.arena import FlatTerrain
    HAS_FLYGYM = True
except ImportError:
    HAS_FLYGYM = False


# All controllable leg joints in FlyGym (7 DoF per leg, 6 legs = 42 joints)
LEG_JOINT_NAMES: list[str] = [
    # Left front (L1)
    "joint_LFCoxa", "joint_LFCoxa_roll", "joint_LFCoxa_yaw",
    "joint_LFFemur", "joint_LFFemur_roll", "joint_LFTibia", "joint_LFTarsus1",
    # Left mid (L2)
    "joint_LMCoxa", "joint_LMCoxa_roll", "joint_LMCoxa_yaw",
    "joint_LMFemur", "joint_LMFemur_roll", "joint_LMTibia", "joint_LMTarsus1",
    # Left hind (L3)
    "joint_LHCoxa", "joint_LHCoxa_roll", "joint_LHCoxa_yaw",
    "joint_LHFemur", "joint_LHFemur_roll", "joint_LHTibia", "joint_LHTarsus1",
    # Right front (R1)
    "joint_RFCoxa", "joint_RFCoxa_roll", "joint_RFCoxa_yaw",
    "joint_RFFemur", "joint_RFFemur_roll", "joint_RFTibia", "joint_RFTarsus1",
    # Right mid (R2)
    "joint_RMCoxa", "joint_RMCoxa_roll", "joint_RMCoxa_yaw",
    "joint_RMFemur", "joint_RMFemur_roll", "joint_RMTibia", "joint_RMTarsus1",
    # Right hind (R3)
    "joint_RHCoxa", "joint_RHCoxa_roll", "joint_RHCoxa_yaw",
    "joint_RHFemur", "joint_RHFemur_roll", "joint_RHTibia", "joint_RHTarsus1",
]

JOINTS_PER_LEG = 7
LEG_LABELS = ["LF", "LM", "LH", "RF", "RM", "RH"]

# Contact sensors: 5 tarsus segments per leg, 6 legs = 30 sensors
CONTACT_SENSORS_PER_LEG = 5


@dataclass
class EnvConfig:
    """Configuration for the FlyGym environment."""

    physics_dt: float = 0.0001          # MuJoCo timestep in seconds (0.1 ms, FlyGym default)
    neural_steps_per_physics: int = 1   # Neural dt (0.1ms) steps per physics step
    sim_duration: float = 2.0           # seconds
    terrain: str = "flat"
    render_mode: str = "saved"          # "saved" for headless video, "none" to skip
    render_fps: int = 30
    render_playback_speed: float = 0.2
    camera: str = "camera_left"

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimulationRecord:
    """Container for recorded simulation data."""

    timestamps: list[float] = field(default_factory=list)
    joint_angles: list[np.ndarray] = field(default_factory=list)
    joint_angle_targets: list[np.ndarray] = field(default_factory=list)
    body_position: list[np.ndarray] = field(default_factory=list)
    body_velocity: list[np.ndarray] = field(default_factory=list)
    foot_contacts: list[np.ndarray] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert all lists to numpy arrays."""
        result = {}
        for name in ["timestamps", "joint_angles", "joint_angle_targets",
                      "body_position", "body_velocity", "foot_contacts"]:
            data = getattr(self, name)
            if data:
                result[name] = np.array(data)
        return result


class FlyEnvironment:
    """Wrapper around FlyGym's NeuroMechFly v2 simulation.

    Provides a simple interface for stepping the physics simulation with
    joint angle targets, recording data, and rendering video.

    FlyGym API (v1.2):
      - obs['joints']: shape (3, 42) — [angles, angular_velocities, torques]
      - obs['fly']: shape (4, 3) — [position, velocity, angular_velocity, orientation]
      - obs['contact_forces']: shape (30, 3) — 5 tarsus sensors × 6 legs
      - Camera(attachment_point, camera_name, ...)
      - action = {'joints': np.ndarray(42,)}
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self.record = SimulationRecord()
        self.sim: Optional[Any] = None
        self.fly: Optional[Any] = None
        self._initialized = False
        self._step_count = 0

    def initialize(self) -> None:
        """Initialize the FlyGym simulation environment."""
        if not HAS_FLYGYM:
            raise ImportError(
                "flygym is not installed. Install with: pip install flygym"
            )

        cfg = self.config

        # Create fly with all leg joints actuated
        # Use tripod init pose (standing) instead of stretch (legs out)
        self.fly = Fly(
            actuated_joints=LEG_JOINT_NAMES,
            enable_adhesion=True,
            init_pose="tripod",
        )

        # Create cameras if rendering
        cameras = []
        if cfg.render_mode != "none":
            cam = Camera(
                self.fly.model.worldbody,
                camera_name=cfg.camera,
                play_speed=cfg.render_playback_speed,
                fps=cfg.render_fps,
            )
            cameras.append(cam)

        # Create simulation
        self.sim = SingleFlySimulation(
            fly=self.fly,
            cameras=cameras,
            timestep=cfg.physics_dt,
            arena=FlatTerrain(),
        )

        self._initialized = True
        self._step_count = 0
        self.record = SimulationRecord()

    def reset(self) -> dict[str, Any]:
        """Reset the simulation and return initial observation."""
        if not self._initialized:
            self.initialize()

        self.record = SimulationRecord()
        self._step_count = 0

        obs, info = self.sim.reset()
        # Store initial joint angles for use as rest pose
        if "joints" in obs:
            self.init_joint_angles = obs["joints"][0].copy()
        return obs

    def step(self, joint_angle_targets: np.ndarray) -> tuple[dict[str, Any], bool]:
        """Step the physics simulation with the given joint angle targets.

        Args:
            joint_angle_targets: Array of shape (42,) with target angles
                for all leg joints in radians.

        Returns:
            Tuple of (observation_dict, terminated).
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Clip targets to reasonable joint ranges
        targets = np.clip(joint_angle_targets, -np.pi, np.pi)

        # Step the simulation (include adhesion signal: all legs adhesive)
        action = {"joints": targets, "adhesion": np.ones(6)}
        obs, reward, terminated, truncated, info = self.sim.step(action)

        # Record data
        t = self._step_count * self.config.physics_dt
        self.record.timestamps.append(t)
        self.record.joint_angle_targets.append(targets.copy())

        # Extract joint angles: obs['joints'] has shape (3, 42)
        # Row 0 = angles, row 1 = angular velocities, row 2 = torques
        if "joints" in obs:
            self.record.joint_angles.append(obs["joints"][0].copy())
        else:
            self.record.joint_angles.append(targets.copy())

        # Extract fly position and velocity: obs['fly'] has shape (4, 3)
        # Row 0 = position, row 1 = velocity, row 2 = angular velocity, row 3 = orientation
        if "fly" in obs:
            fly_state = obs["fly"]
            self.record.body_position.append(fly_state[0].copy())
            self.record.body_velocity.append(fly_state[1].copy())
        else:
            self.record.body_position.append(np.zeros(3))
            self.record.body_velocity.append(np.zeros(3))

        # Extract foot contacts from contact_forces: shape (30, 3)
        # 5 tarsus sensors per leg × 6 legs
        contacts = self._detect_foot_contacts(obs)
        self.record.foot_contacts.append(contacts)

        # Capture frame for video
        if self.config.render_mode != "none" and self.sim:
            rendered = self.sim.render()
            if rendered is not None:
                if isinstance(rendered, list):
                    # Filter out None entries (render returns [None] between frames)
                    for frame in rendered:
                        if frame is not None and isinstance(frame, np.ndarray):
                            self.record.frames.append(frame)
                elif isinstance(rendered, np.ndarray):
                    self.record.frames.append(rendered)

        self._step_count += 1
        return obs, terminated or truncated

    def _detect_foot_contacts(self, obs: dict) -> np.ndarray:
        """Detect which feet are in contact with the ground.

        Contact forces have shape (30, 3): 5 tarsus segments × 6 legs.
        A leg is in contact if any of its 5 tarsus segments has nonzero force.
        """
        contacts = np.zeros(6, dtype=bool)

        if "contact_forces" in obs:
            cf = obs["contact_forces"]  # (30, 3)
            for leg in range(6):
                # 5 sensors per leg
                leg_forces = cf[leg * CONTACT_SENSORS_PER_LEG:
                                (leg + 1) * CONTACT_SENSORS_PER_LEG]
                total_force = np.linalg.norm(leg_forces.sum(axis=0))
                contacts[leg] = total_force > 0.1
        else:
            contacts[:] = True

        return contacts

    def get_frames(self) -> list[np.ndarray]:
        """Return rendered frames."""
        return self.record.frames

    def save_video(self, path: str) -> None:
        """Save rendered frames as an MP4 video."""
        if not self.record.frames:
            print("No frames to save.")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        try:
            import mediapy as media
            media.write_video(path, self.record.frames, fps=self.config.render_fps)
            print(f"Video saved to {path}")
        except ImportError:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FFMpegWriter

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")

            writer = FFMpegWriter(fps=self.config.render_fps)
            with writer.saving(fig, path, dpi=100):
                for frame in self.record.frames:
                    ax.clear()
                    ax.axis("off")
                    ax.imshow(frame)
                    writer.grab_frame()
            plt.close(fig)
            print(f"Video saved to {path}")

    def close(self) -> None:
        """Clean up the simulation."""
        if self.sim is not None:
            self.sim.close()
        self._initialized = False

    @property
    def n_joints(self) -> int:
        """Total number of actuated joints."""
        return len(LEG_JOINT_NAMES)

    @property
    def total_steps(self) -> int:
        """Total number of physics steps for the configured duration."""
        return int(self.config.sim_duration / self.config.physics_dt)
