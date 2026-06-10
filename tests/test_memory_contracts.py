"""Contract tests for the GPU model-residency policy (hairport.memory).

All tests are CPU-only: device moves are exercised with a recording dummy
module, never a real GPU.
"""

from __future__ import annotations

import unittest
from importlib.util import find_spec
from unittest.mock import MagicMock


@unittest.skipUnless(
    find_spec("torch") and find_spec("omegaconf"),
    "HairPort runtime dependencies not installed",
)
class MemoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from hairport.config import load_config, reset_config, set_config

        self.reset_config = reset_config
        reset_config()
        set_config(load_config())

    def tearDown(self) -> None:
        self.reset_config()

    def _recording_module(self):
        import torch.nn as nn

        moves: list[str] = []

        class Recorder(nn.Module):
            def to(self, device, *args, **kwargs):  # type: ignore[override]
                moves.append(str(device))
                return self

        return Recorder(), moves

    def test_on_gpu_exclusive_moves_on_and_off(self) -> None:
        from hairport import memory

        module, moves = self._recording_module()
        with memory.on_gpu(module, "cuda:0", exclusive=True):
            self.assertEqual(moves, ["cuda:0"])
        self.assertEqual(moves, ["cuda:0", "cpu"])

    def test_on_gpu_resident_policy_only_onloads(self) -> None:
        from hairport import memory

        module, moves = self._recording_module()
        with memory.on_gpu(module, "cuda:0", exclusive=False):
            pass
        self.assertEqual(moves, ["cuda:0"])  # no offload afterwards

    def test_offload_is_noop_under_resident_policy(self) -> None:
        from hairport import memory
        from hairport.config import load_config, set_config

        set_config(load_config(overrides=["memory.policy=resident"]))
        module, moves = self._recording_module()
        memory.offload(module)
        self.assertEqual(moves, [])

        set_config(load_config(overrides=["memory.policy=exclusive"]))
        memory.offload(module)
        self.assertEqual(moves, ["cpu"])

    def test_move_to_skips_accelerate_hooked_pipelines(self) -> None:
        from hairport import memory

        pipe = MagicMock()
        pipe._all_hooks = [object()]  # as set by enable_model_cpu_offload
        memory.move_to(pipe, "cuda:0")
        pipe.to.assert_not_called()

    def test_apply_offload_mode_dispatch(self) -> None:
        from hairport import memory

        pipe = MagicMock()
        memory.apply_offload_mode(pipe, "none", "cuda:0")
        pipe.to.assert_called_once_with("cuda:0")

        pipe = MagicMock()
        memory.apply_offload_mode(pipe, "model", "cuda:0")
        pipe.enable_model_cpu_offload.assert_called_once_with(device="cuda:0")

        pipe = MagicMock()
        memory.apply_offload_mode(pipe, "sequential", "cuda:0")
        pipe.enable_sequential_cpu_offload.assert_called_once_with(device="cuda:0")

        with self.assertRaises(ValueError):
            memory.apply_offload_mode(MagicMock(), "bogus", "cuda:0")

    def test_config_validates_memory_section(self) -> None:
        from hairport.config import load_config

        cfg = load_config(overrides=["memory.policy=resident"])
        self.assertEqual(cfg.memory.policy, "resident")
        with self.assertRaisesRegex(ValueError, "memory.policy"):
            load_config(overrides=["memory.policy=sometimes"])
        with self.assertRaisesRegex(ValueError, "memory.flux_offload"):
            load_config(overrides=["memory.flux_offload=turbo"])

    def test_default_policy_is_exclusive(self) -> None:
        from hairport import memory

        self.assertEqual(memory.memory_policy(), "exclusive")
        self.assertEqual(memory.flux_offload_mode(), "none")
        self.assertTrue(memory.exclusive_enabled())


if __name__ == "__main__":
    unittest.main()
