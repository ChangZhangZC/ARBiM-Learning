"""Opt-in subclass PPO diagnostics keep upstream ratio records unmodified."""
from types import SimpleNamespace

import torch

from arbim.rl.online.telemetry import RatioDiagnosticsMixin, native_ratio_summary


class FakeNativePPO:
    _enable_ratio_logging = True

    def __init__(self):
        self.args = SimpleNamespace(epsilon=0.2)
        self._ratio_records = []
        self.calls = 0
        self.should_record = True

    def _record_ratio_stats(self, phase, step, ratio, old_lp, new_lp):
        self.calls += 1
        if self.should_record:
            self._ratio_records.append({
                "phase": phase, "denoise_step": step,
                "ratio_mean": ratio.mean().item(),
                "delta_mean": (new_lp - old_lp).mean().item(),
            })


class PPOWithDiagnostics(RatioDiagnosticsMixin, FakeNativePPO):
    pass


def test_ratio_clip_fraction_and_approx_kl_from_native_tensors():
    ppo = PPOWithDiagnostics()
    ratios = torch.tensor([0.7, 1.0, 1.3], dtype=torch.float32)
    old_logprob = torch.zeros(3)
    new_logprob = ratios.log()
    ppo._record_ratio_stats("online", 0, ratios, old_logprob, new_logprob)
    assert ppo.calls == 1
    record = ppo._ratio_records[0]
    assert record["sample_size"] == 3
    assert abs(record["clip_fraction"] - 2 / 3) < 1e-5
    target_kl = ((ratios - 1) - ratios.log()).mean().item()
    assert abs(record["approx_kl"] - target_kl) < 1e-6
    summary = native_ratio_summary(ppo)
    assert abs(summary["clip_fraction"] - 2 / 3) < 1e-5


def test_native_logger_skip_does_not_leave_stale_diagnostics():
    ppo = PPOWithDiagnostics()
    ppo.should_record = False
    ppo._record_ratio_stats("online", 0, torch.ones(2), torch.zeros(2), torch.zeros(2))
    assert ppo._ratio_records == []
    assert ppo.calls == 1
