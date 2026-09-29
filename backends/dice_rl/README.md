# DICE-RL backend

This package provides policy networks, BC pretraining, residual RL, replay, and RoboMimic/MimicGen wrappers for HiRE experiments. It imports `hire` for reward mathematics, visual encoders, and reference-buffer storage.

From the repository root:

```bash
pip install -e '.[vision,dev]' -e './backends/dice_rl[sim]'
```

See [training](../../docs/training.md), [data preparation](../../docs/data.md), and [reward integration](../../docs/integration.md). The upstream MIT notice is retained in [LICENSE](LICENSE).
