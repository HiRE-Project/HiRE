# Quick start: Stack Three

This walkthrough downloads the released Stack Three base policy, finetunes it with HiRE, and evaluates the resulting checkpoint. Run commands from the repository root after completing [installation](setup.md#robotics-experiment-environment).

## Prepare data

```bash
source scripts/set_path.sh
python scripts/dataset/prepare_mimicgen_datasets.py \
  --tasks stack_three --max-episodes 200 --jobs 1
```

The processed files are written to:

```text
data_dir/mimicgen/stack_three-img/
├── ph_pretrain/
│   ├── train.npz
│   └── normalization.npz
└── ph_finetune/
    ├── train.npz
    └── normalization.npz
```

## Download the base policy

```bash
python scripts/download_checkpoints.py --tasks stack_three
```

This downloads the selected BC checkpoint and its original configuration from [JimmyHan2004/HiRE-release](https://huggingface.co/JimmyHan2004/HiRE-release):

```text
checkpoints/HiRE-release/stack_three/
├── .hydra/config.yaml
└── checkpoint/state_75.pt
```

`HIRE_CHECKPOINT_DIR` overrides the `checkpoints/HiRE-release/` location. Use the same value for downloading and launching. Existing files are reused on subsequent downloads; use `--force` to download them again. BC pretraining is optional when using these weights.

## Finetune

Check that the checkpoint, data, normalizer, and CUDA device are available:

```bash
python scripts/launch.py finetune stack_three --check
```

Then launch training:

```bash
python scripts/launch.py finetune stack_three --seed 42
```

HiRE builds `dino_positive_buffer.pt` alongside the pretraining data on first use. Finetuning outputs are saved under `log_dir/mimicgen-finetune/`, with policy checkpoints named `model_step_<step>.pth`.

To check a prepared pipeline briefly before a full run, append:

```text
train.num_train_steps=2 env.n_envs=1 run_eval=false train.batch_size=2 train.gradient_steps=1
```

This short run checks execution and checkpoint saving; it does not measure learning performance. To compare sparse rewards, add `--reward sparse` and use the same BC checkpoint and seed. Add `--wandb online` to log a new W&B run. The main Stack Three seeds are 42, 123, and 345; see the [main-figure presets](training.md#main-figure-runs) for other tasks.

## Evaluate

```bash
python scripts/eval_rl_checkpoint.py \
  --ckpt_path /path/to/finetune_run/model_step_1000.pth \
  --num_eval_episodes 10 --eval_n_envs 10
```

The evaluator runs 100 episodes for each policy and reports success rates and returns for the base and finetuned policies. Add `--render` for videos. A short pipeline check instead produces `model_step_2.pth`; use the checkpoint that your run actually saved.

Threading and Three Piece Assembly have released checkpoints selected in the same way. Tool Hang has no released weights and requires `--checkpoint`. For task-specific data preparation, follow the [data guide](data.md). See [training settings](training.md) for configuration overrides and ablations.

## Use your own BC policy

Run `python scripts/launch.py pretrain stack_three --seed 42`, then pass the resulting checkpoint explicitly:

```bash
python scripts/launch.py finetune stack_three \
  --checkpoint /path/to/bc_run/checkpoint/state_100.pt --seed 42
```

Preserve the BC run's `.hydra/config.yaml` alongside its `checkpoint/` directory. Custom checkpoints must match the task's observation/action conventions and demonstration count.
