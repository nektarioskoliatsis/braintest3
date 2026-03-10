"""Embodiment layer: FlyGym environment and neural controller."""

from .fly_env import FlyEnvironment
from .neural_controller import NeuralController

__all__ = ["FlyEnvironment", "NeuralController"]
