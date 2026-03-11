"""Tests for the neural controller and baseline controllers."""

import numpy as np
import pytest

from cpg.spiking_cpg import SpikingCPG, CPGConfig
from embodiment.neural_controller import (
    NeuralController,
    ControllerConfig,
    SineWaveController,
    RandomController,
    REST_POSE,
    LEG_ORDER,
)


class TestNeuralController:
    """Tests for the CPG spike decoder."""

    @pytest.fixture
    def cpg_and_controller(self) -> tuple[SpikingCPG, NeuralController]:
        cpg = SpikingCPG()
        ctrl = NeuralController(cpg)
        return cpg, ctrl

    def test_output_shape(self, cpg_and_controller: tuple) -> None:
        cpg, ctrl = cpg_and_controller
        spikes = cpg.step()
        targets = ctrl.decode_spikes(spikes.numpy())
        assert targets.shape == (42,)  # 7 joints × 6 legs

    def test_output_range(self, cpg_and_controller: tuple) -> None:
        cpg, ctrl = cpg_and_controller
        # Run for a while to get stable output
        for _ in range(500):
            spikes = cpg.step()
        targets = ctrl.decode_spikes(spikes.numpy())
        # Targets should be within reasonable joint angle range
        assert np.all(np.abs(targets) < 5.0), "Joint angles should be bounded"

    def test_batch_decode(self, cpg_and_controller: tuple) -> None:
        cpg, ctrl = cpg_and_controller
        spikes = cpg.run(100.0)  # 1000 steps
        targets = ctrl.decode_spike_batch(spikes.numpy())
        assert targets.shape == (1000, 42)

    def test_reset_clears_buffer(self, cpg_and_controller: tuple) -> None:
        cpg, ctrl = cpg_and_controller
        for _ in range(100):
            spikes = cpg.step()
            ctrl.decode_spikes(spikes.numpy())
        assert len(ctrl.spike_buffer) > 0
        ctrl.reset()
        assert len(ctrl.spike_buffer) == 0

    def test_zero_spikes_near_rest_pose(self) -> None:
        """When no spikes, output should be close to the rest pose."""
        cpg = SpikingCPG()
        ctrl = NeuralController(cpg)
        zero_spikes = np.zeros(cpg.N)
        for _ in range(ctrl.window_steps):
            targets = ctrl.decode_spikes(zero_spikes)
        # Should be close to rest pose (phase offsets cause some deviation
        # on joints with nonzero amplitude)
        assert np.all(np.abs(targets - ctrl.rest_pose) < 1.0), \
            "Joint targets should be near rest pose when no spikes"

    def test_rest_pose_shape(self) -> None:
        """Rest pose should have correct shape."""
        cpg = SpikingCPG()
        ctrl = NeuralController(cpg)
        assert ctrl.rest_pose.shape == (42,)


class TestSineWaveController:
    """Tests for the sine wave baseline controller."""

    def test_output_shape(self) -> None:
        ctrl = SineWaveController()
        targets = ctrl.step()
        assert targets.shape == (42,)

    def test_periodicity(self) -> None:
        ctrl = SineWaveController(frequency=10.0, dt_ms=0.1)
        # One full period = 100ms = 1000 steps
        first = ctrl.step()
        for _ in range(999):
            ctrl.step()
        after_period = ctrl.step()
        # After exactly one period, should be close to the first value
        np.testing.assert_allclose(first, after_period, atol=0.01)

    def test_tripod_phase_offset(self) -> None:
        """L1 and L2 should be in anti-phase (tripod pattern)."""
        ctrl = SineWaveController(frequency=10.0, dt_ms=0.1)
        l1_coxa = []
        l2_coxa = []
        for _ in range(1000):
            t = ctrl.step()
            l1_coxa.append(t[0])  # L1 Coxa
            l2_coxa.append(t[7])  # L2 Coxa

        l1 = np.array(l1_coxa)
        l2 = np.array(l2_coxa)

        # Cross-correlation should be negative (anti-phase)
        correlation = np.corrcoef(l1, l2)[0, 1]
        assert correlation < 0, "L1 and L2 should be anti-correlated (anti-phase)"

    def test_reset(self) -> None:
        ctrl = SineWaveController()
        ctrl.step()
        ctrl.step()
        ctrl.reset()
        assert ctrl.t == 0.0

    def test_rest_pose_used(self) -> None:
        """At t=0, joints with zero amplitude should equal rest pose."""
        ctrl = SineWaveController()
        targets = ctrl.step()
        # Coxa_roll (index 1) has zero amplitude, should match rest pose
        assert targets[1] == ctrl.rest_pose[1]


class TestRandomController:
    """Tests for the random noise baseline controller."""

    def test_output_shape(self) -> None:
        ctrl = RandomController()
        targets = ctrl.step()
        assert targets.shape == (42,)

    def test_randomness(self) -> None:
        ctrl = RandomController()
        t1 = ctrl.step()
        t2 = ctrl.step()
        assert not np.allclose(t1, t2)

    def test_centered_on_rest_pose(self) -> None:
        ctrl = RandomController(amplitude=0.1, seed=42)
        samples = np.array([ctrl.step() for _ in range(1000)])
        # Mean should be close to rest pose
        np.testing.assert_allclose(samples.mean(axis=0), ctrl.rest_pose, atol=0.05)

    def test_zero_amplitude_joints_unchanged(self) -> None:
        """Joints with zero gain should always equal rest pose."""
        ctrl = RandomController(seed=42)
        targets = ctrl.step()
        # Coxa_roll (index 1) has zero amplitude
        assert targets[1] == ctrl.rest_pose[1]


class TestControllerConfig:
    """Tests for controller configuration."""

    def test_default_gains(self) -> None:
        cfg = ControllerConfig()
        assert len(cfg.gains) == 7

    def test_default_phase_offsets(self) -> None:
        cfg = ControllerConfig()
        assert len(cfg.joint_phase_offsets) == 7
