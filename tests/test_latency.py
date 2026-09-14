"""Stability of the latency harness: two consecutive measurements agree within 5%."""

import torch
import torch.nn as nn

from measure.latency import measure


def tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(8, 4)).eval()


def test_two_measurements_within_5_percent():
    torch.set_num_threads(1)
    m = tiny()
    x = torch.randn(1, 3, 64, 64)

    def fn(t):
        with torch.no_grad():
            return m(t)
    a = measure(fn, x, warmup=50, iters=300, sync=lambda: None)
    b = measure(fn, x, warmup=50, iters=300, sync=lambda: None)
    assert abs(a["p50_ms"] - b["p50_ms"]) / a["p50_ms"] < 0.05, (a["p50_ms"], b["p50_ms"])
    assert a["p95_ms"] >= a["p50_ms"] >= a["min_ms"] > 0


def test_stats_keys_and_iters():
    r = measure(lambda t: t * 2, torch.ones(4), warmup=3, iters=10, sync=lambda: None)
    assert r["iters"] == 10 and r["warmup"] == 3
    assert set(r) >= {"p50_ms", "p95_ms", "mean_ms", "std_ms", "min_ms"}
