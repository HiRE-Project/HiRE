<div align="center">

# HiRE: Hindsight Reward Editing for Policy Finetuning

</div>

## Introduction

**HiRE edits visual rewards using the outcomes of past interactions.** Frozen foundation representations provide visual similarity, while successful and failed trajectories ground that similarity in task progress. HiRE attracts the policy toward successful states, suppresses visually plausible failure states, and incorporates the edited potential through reward shaping without training an additional reward model.

This repository contains the simulation training and evaluation code for HiRE. The experiments use DICE-RL to finetune diffusion and flow-matching policies on four manipulation tasks.

| Task | Benchmark | Base policy | Policy image resolution | Demonstrations | Released checkpoint |
| --- | --- | --- | --- | --- | --- |
| `stack_three` | MimicGen D0 | Diffusion | 96 × 96 | 200 | Epoch 75 |
| `threading` | MimicGen D0 | Diffusion | 96 × 96 | 30 | Epoch 100 |
| `three_piece_assembly` | MimicGen D0 | Diffusion | 96 × 96 | 200 | Epoch 75 |
| `tool_hang` | RoboMimic | Flow matching | 240 × 240 | 20 | Not released |

## Installation

Use Linux with an NVIDIA GPU and EGL offscreen rendering. From the repository root:

```bash
conda create -n hire python=3.10 -y
conda activate hire
```

On NVIDIA Blackwell GPUs, install PyTorch's CUDA 12.8 wheels before the project packages:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

RoboMimic 0.5.0 is available from its official Git tag. Install it alongside the project packages:

```bash
mkdir -p external
git clone --depth 1 --branch v0.5.0 https://github.com/ARISE-Initiative/robomimic.git external/robomimic
python -m pip install -e '.[vision]' -e './backends/dice_rl[sim]' -e external/robomimic
git clone https://github.com/NVlabs/mimicgen.git external/mimicgen
python -m pip install -e external/mimicgen
source scripts/set_path.sh
```

Data and outputs default to `data_dir/` and `log_dir/`. Set `HIRE_DATA_DIR` and `HIRE_LOG_DIR` to use other locations. W&B logging is disabled by default. See [setup](docs/setup.md) for rendering and logging options.

## Training

The following commands run Stack Three. See the [quick start](docs/quickstart.md) for checkpoint locations and the [data guide](docs/data.md) for the other tasks.

### 1. Prepare demonstrations

```bash
python scripts/dataset/prepare_mimicgen_datasets.py \
  --tasks stack_three --max-episodes 200 --jobs 1
```

This downloads the MimicGen demonstrations and prepares normalized pretraining and finetuning datasets. Tool Hang uses a separate data-conversion command with its camera settings, described in the [data guide](docs/data.md#robomimic-tool-hang).

### 2. Download the base policy

The three MimicGen BC policies are available on [Hugging Face](https://huggingface.co/JimmyHan2004/HiRE-release). You can use them directly without BC pretraining:

```bash
python scripts/download_checkpoints.py --tasks stack_three
```

Omit `--tasks` to download all three policies (about 624 MB total). The script saves each checkpoint and its Hydra config under `checkpoints/HiRE-release/`. Set `HIRE_CHECKPOINT_DIR` to use another location.

### 3. Finetune with HiRE

```bash
python scripts/launch.py finetune stack_three --seed 42
```

The launcher selects the released checkpoint for the task. For online W&B logging, append `--wandb online`. The [main-figure run table](docs/training.md#main-figure-runs) lists the recorded seeds and task-specific settings.

On first use, HiRE loads DINOv2 and builds a positive reference buffer from the demonstrations. During finetuning, successful and failed rollouts update the hindsight buffers used for reward computation.

For a sparse-reward comparison from the same base policy:

```bash
python scripts/launch.py finetune stack_three --reward sparse --seed 42
```

Use `threading` or `three_piece_assembly` after preparing that task's data and downloading its checkpoint. Tool Hang requires a separately supplied flow-matching BC checkpoint:

```bash
python scripts/launch.py finetune tool_hang \
  --checkpoint /path/to/bc_run/checkpoint/state_500.pt --seed 42
```

### Optional: pretrain your own policy

```bash
python scripts/launch.py pretrain stack_three --seed 42
python scripts/launch.py finetune stack_three \
  --checkpoint /path/to/bc_run/checkpoint/state_100.pt --seed 42
```

Pretraining saves runs under `log_dir/mimicgen-pretrain/`. Keep `.hydra/config.yaml` with `checkpoint/state_<epoch>.pt`. An explicit `--checkpoint` overrides the released base policy.

## Evaluation

Evaluate the finetuned policy and its base policy on the same seeded initial states:

```bash
python scripts/eval_rl_checkpoint.py \
  --ckpt_path /path/to/finetune_run/model_step_1000.pth \
  --num_eval_episodes 10 --eval_n_envs 10
```

This runs 100 episodes per policy (10 batches × 10 parallel environments) and reports sparse task success, return, and improvement over the base policy. Add `--render` to save rollout videos. Use `--base_policy_path` and `--normalization_path` if the checkpoint bundle or data has moved.

## Experiment settings

Task recipes are in `configs/experiment/`. Reward settings are in `configs/reward/`, and camera/environment settings are in `configs/task/`. For example, change the contrastive weight and negative-buffer capacity:

```bash
python scripts/launch.py finetune threading \
  reward.contrastive_weight=0.3 reward.negative_buffer.buffer_size=512
```

To use SigLIP, install `'.[siglip]'` and add `encoder=siglip`. Use `--config-only` to inspect a run's settings or `--check` to check its local inputs before starting it. See the [training reference](docs/training.md) for defaults, ablations, and implementation details.

## Code structure

```text
src/hire/       HiRE potential, reward shaping, visual encoders, and hindsight buffers
backends/       Policy training, replay, and environment integration
configs/        Task and experiment settings
scripts/        Data preparation, training, and evaluation commands
docs/           Setup, experiment instructions, and implementation reference
tests/          Reward, policy, and integration checks
```

For the method implementation, start with [`potential.py`](src/hire/potential.py) and [`shaping.py`](src/hire/shaping.py). See the [implementation guide](docs/integration.md) for how these components connect to the training loop.

## Citation

If you find HiRE useful in your research, please cite:

```bibtex
@inproceedings{niu2026hire,
  title  = {HiRE: Hindsight Reward Editing for Policy Finetuning},
  author = {Niu, Haoyi and Han, Zhengtao and Ji, Yufeng and Li, Zhongyu and Sreenath, Koushil},
  booktitle = {Conference on Robot Learning (CoRL)},
  year   = {2026},
}
```

## Acknowledgements

Our implementation builds on [DICE-RL](https://github.com/real-stanford/dice-rl), [DPPO](https://github.com/irom-princeton/dppo), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [RoboMimic](https://github.com/ARISE-Initiative/robomimic), and [MimicGen](https://github.com/NVlabs/mimicgen). We thank the authors for sharing their code. Please also cite DICE-RL when using the policy-learning backbone:

```bibtex
@article{sun2026prior,
  title={From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning},
  author={Sun, Zhanyi and Song, Shuran},
  journal={arXiv preprint arXiv:2603.10263},
  year={2026}
}
```

Code is distributed under the [MIT License](LICENSE). External datasets, pretrained encoders, and simulator packages retain their respective licenses.
