"""Tests for spiking CPG and connectome-inspired CPG networks."""

import numpy as np
import pytest
import torch

from cpg.spiking_cpg import SpikingCPG, CPGConfig, LEG_NAMES
from cpg.connectome_cpg import ConnectomeCPG, ConnectomeConfig


class TestSpikingCPG:
    """Tests for the hand-tuned spiking CPG."""

    def test_initialization(self) -> None:
        cpg = SpikingCPG()
        assert cpg.N == 120  # 20 neurons × 6 legs
        assert cpg.W.shape == (120, 120)
        assert cpg.V.shape == (120,)

    def test_custom_config(self) -> None:
        cfg = CPGConfig(neurons_per_module=10, num_legs=6)
        cpg = SpikingCPG(config=cfg)
        assert cpg.N == 60
        assert cpg.W.shape == (60, 60)

    def test_step_produces_spikes(self) -> None:
        cpg = SpikingCPG()
        # Run for enough steps to produce some spikes
        total_spikes = 0
        for _ in range(1000):
            spikes = cpg.step()
            total_spikes += spikes.sum().item()
        assert total_spikes > 0, "CPG should produce spikes within 1000 steps"

    def test_run_duration(self) -> None:
        cfg = CPGConfig(dt=0.1)
        cpg = SpikingCPG(config=cfg)
        duration_ms = 100.0
        spikes = cpg.run(duration_ms)
        expected_steps = int(duration_ms / cfg.dt)
        assert spikes.shape == (expected_steps, cpg.N)

    def test_spike_history(self) -> None:
        cpg = SpikingCPG()
        cpg.run(50.0)
        history = cpg.get_spike_history()
        assert history.shape[0] == 500  # 50ms / 0.1ms
        assert history.shape[1] == 120
        # All values should be 0 or 1
        assert set(np.unique(history)).issubset({0.0, 1.0})

    def test_reset_clears_state(self) -> None:
        cpg = SpikingCPG()
        cpg.run(10.0)
        assert len(cpg.spike_history) > 0

        cpg.reset()
        assert len(cpg.spike_history) == 0
        assert cpg.time_steps == 0

    def test_weight_matrix_structure(self) -> None:
        cpg = SpikingCPG()
        W = cpg.W.numpy()
        # Diagonal should be zero (no self-connections)
        assert np.allclose(np.diag(W), 0.0)
        # Should have both positive and negative weights
        assert (W > 0).any(), "Should have excitatory connections"
        assert (W < 0).any(), "Should have inhibitory connections"

    def test_external_current(self) -> None:
        cpg = SpikingCPG()
        I_ext = torch.zeros(cpg.N)
        # With zero drive, should produce fewer spikes
        spikes_no_drive = cpg.run(100.0, I_ext=I_ext)
        cpg.reset()
        # With strong drive
        I_ext_strong = torch.full((cpg.N,), 10.0)
        spikes_strong = cpg.run(100.0, I_ext=I_ext_strong)
        assert spikes_strong.sum() > spikes_no_drive.sum()

    def test_module_spike_rates(self) -> None:
        cpg = SpikingCPG()
        cpg.run(200.0)
        rates = cpg.get_module_spike_rates(window_ms=50.0)
        assert rates.shape[1] == 6  # 6 legs
        assert rates.shape[2] == 2  # flexor, extensor


class TestConnectomeCPG:
    """Tests for the connectome-inspired CPG."""

    def test_initialization(self) -> None:
        cpg = ConnectomeCPG()
        assert cpg.N == 120
        assert len(cpg.neuron_types) == 120

    def test_neuron_types(self) -> None:
        cpg = ConnectomeCPG()
        types = cpg.get_neuron_type_summary()
        assert "mn" in types
        assert "exc_in" in types
        assert "inh_in" in types
        assert sum(types.values()) == 120

    def test_weight_statistics(self) -> None:
        cpg = ConnectomeCPG()
        stats = cpg.get_weight_statistics()
        assert stats["total_connections"] > 0
        assert stats["n_excitatory"] > 0
        assert stats["n_inhibitory"] > 0
        assert 0 < stats["density"] < 1

    def test_produces_spikes(self) -> None:
        cpg = ConnectomeCPG()
        spikes = cpg.run(200.0)
        assert spikes.sum() > 0

    def test_deterministic_with_seed(self) -> None:
        cpg1 = ConnectomeCPG(seed=42)
        cpg2 = ConnectomeCPG(seed=42)
        W1 = cpg1.W.numpy()
        W2 = cpg2.W.numpy()
        np.testing.assert_array_equal(W1, W2)

    def test_different_seeds_differ(self) -> None:
        cpg1 = ConnectomeCPG(seed=42)
        cpg2 = ConnectomeCPG(seed=99)
        W1 = cpg1.W.numpy()
        W2 = cpg2.W.numpy()
        assert not np.allclose(W1, W2)

    def test_custom_connectome_config(self) -> None:
        cc = ConnectomeConfig(mn_fraction=0.5, exc_in_fraction=0.25, inh_in_fraction=0.25)
        cpg = ConnectomeCPG(connectome_config=cc)
        types = cpg.get_neuron_type_summary()
        # Motor neurons should be the largest group
        assert types["mn"] >= types["exc_in"]

    def test_ei_ratio_biological(self) -> None:
        """E/I ratio should be roughly balanced (biological constraint)."""
        cpg = ConnectomeCPG()
        stats = cpg.get_weight_statistics()
        # Biological E/I ratio is roughly 1-3
        assert 0.3 < stats["ei_ratio"] < 5.0


class TestCPGConfig:
    """Tests for CPG configuration."""

    def test_from_dict(self) -> None:
        d = {"neurons_per_module": 30, "tau_m": 15.0, "dt": 0.05}
        cfg = CPGConfig.from_dict(d)
        assert cfg.neurons_per_module == 30
        assert cfg.tau_m == 15.0
        assert cfg.dt == 0.05

    def test_total_neurons(self) -> None:
        cfg = CPGConfig(neurons_per_module=20, num_legs=6)
        assert cfg.total_neurons == 120

    def test_half_module(self) -> None:
        cfg = CPGConfig(neurons_per_module=20)
        assert cfg.half_module == 10
