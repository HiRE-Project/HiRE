import json
from pathlib import Path
import runpy
import numpy as np
from hire_dice_rl.integration import reward


def test_dice_rl_reward_trace_matches_reference(tmp_path):
    trace = runpy.run_path(str(Path(__file__).with_name("reward_trace.py")))["reward_trace"]
    actual = trace(reward, reward.HiRERewardAdapter, tmp_path)
    expected = json.loads((Path(__file__).parents[1] / "fixtures/dice_rl_reward_trace.json").read_text())
    for got, wanted in zip(actual, expected, strict=True):
        for key in ("reward", "previous"):
            np.testing.assert_allclose(got[key], wanted[key], rtol=1e-7, atol=1e-7)
        for a, b in zip(got["substep"], wanted["substep"], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
        assert got["positive_size"] == wanted["positive_size"]
        assert got["negative_size"] == wanted["negative_size"]
