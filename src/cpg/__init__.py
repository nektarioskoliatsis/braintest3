"""Central Pattern Generator modules for rhythmic locomotion."""

from .spiking_cpg import SpikingCPG
from .connectome_cpg import ConnectomeCPG

__all__ = ["SpikingCPG", "ConnectomeCPG"]
