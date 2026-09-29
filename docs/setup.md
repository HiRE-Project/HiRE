# Installation

The training target is **Linux, Python 3.10, an NVIDIA GPU, and EGL offscreen rendering**. The package supports Python 3.10–3.12.

## Robotics experiment environment

From the repository root:

```bash
conda create -n hire python=3.10 -y
conda activate hire
```

On NVIDIA Blackwell GPUs, install the CUDA 12.8 PyTorch wheels before the editable packages:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

RoboMimic 0.5.0 is available from its official Git tag:

```bash
mkdir -p external
git clone --depth 1 --branch v0.5.0 https://github.com/ARISE-Initiative/robomimic.git external/robomimic
python -m pip install -e '.[vision,dev]' -e './backends/dice_rl[sim]' -e external/robomimic
git clone https://github.com/NVlabs/mimicgen.git external/mimicgen
python -m pip install -e external/mimicgen
source scripts/set_path.sh
python -m pip check
```

The core and backend `pyproject.toml` files pin PyTorch 2.7.1, torchvision 0.22.1, Gym 0.22.0, RoboMimic 0.5.0, Robosuite 1.4.1, and MuJoCo 2.3.2. Do not replace Gym with Gymnasium: the vector environment uses Gym's older API. MimicGen's [installation guide](https://mimicgen.github.io/docs/introduction/installation.html) documents compatibility with Robosuite 1.4.1 and the MuJoCo version. Kitchen/Hammer-specific task-zoo packages are unnecessary for the three released MimicGen tasks.

Record `git -C external/robomimic rev-parse HEAD`, `git -C external/mimicgen rev-parse HEAD` and `python -m pip freeze` with each experiment so its dependency versions can be reproduced.

MuJoCo rendering also needs the NVIDIA driver and system EGL/OpenGL libraries. On a headless Linux machine use `MUJOCO_GL=egl`; a valid `nvidia-smi` result alone does not verify rendering. CPU checks do not validate the Linux simulator stack.

## Paths and logging

```bash
export HIRE_DATA_DIR=/absolute/path/to/data
export HIRE_LOG_DIR=/absolute/path/to/runs
export HIRE_CHECKPOINT_DIR=/absolute/path/to/HiRE-release
source scripts/set_path.sh
```

The defaults are `data_dir/`, `log_dir/`, and `checkpoints/HiRE-release/` inside this checkout. Sourcing this script only sets the current shell's environment; it does not edit shell startup files. `scripts/launch.py` supplies these defaults independently.

Weights & Biases is optional at runtime. The launcher defaults to `WANDB_MODE=disabled`. To enable it, pass `--wandb online` or set the environment variable:

```bash
export WANDB_MODE=online
export DICE_RL_WANDB_ENTITY=your-team
wandb login
```

## BC checkpoints

```bash
python scripts/download_checkpoints.py --tasks stack_three
python scripts/launch.py finetune stack_three --config-only
```

The public checkpoint downloads require neither an HF login nor the HF CLI. The script uses Python's standard library. Tool Hang requires an explicit checkpoint path. Demonstration datasets and normalizers must still be prepared before training.

## Frozen reward encoders

DINOv2 ViT-S/14 is downloaded by `torch.hub` on the first HiRE run. The source revision is pinned in `src/hire/encoders/vision.py`. To use an existing source checkout, set `DINO_REPO_LOCAL=/absolute/path/to/dinov2`; the model weights must still be present in the Torch Hub cache for fully offline use. Configure `TORCH_HOME` to relocate that cache.

For SigLIP:

```bash
python -m pip install -e '.[siglip]'
python scripts/launch.py finetune stack_three encoder=siglip
```

The default model is `google/siglip-base-patch16-224`; `HIRE_SIGLIP_MODEL` overrides it. Use distinct positive-buffer files for different encoders. The default path includes the encoder name, but does not fingerprint the dataset, camera order, or model revision: rebuild the cache when any of those change.

## Checks

```bash
python scripts/check_release.py
python -m pytest tests/core -q
python -m pytest -q
python scripts/launch.py pretrain stack_three --config-only
```

Before a long run, use the [pipeline check](training.md#execution-checks) with prepared data and a compatible BC checkpoint. Checkpoint camera views, action conventions, and normalization must match the selected task; see [checkpoint bundles](data.md#checkpoint-bundles).
