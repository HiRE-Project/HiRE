# Data preparation

All paths below are relative to the repository root unless stated otherwise. Run `source scripts/set_path.sh` first. Demonstrations and model weights are stored separately from the source code. Released BC policies are available on [Hugging Face](https://huggingface.co/JimmyHan2004/HiRE-release).

## MimicGen

The helper downloads the official core D0 HDF5 datasets through MimicGen's registry, reads stored RGB observations, and creates pretraining and finetuning NPZs:

```bash
python scripts/dataset/prepare_mimicgen_datasets.py --tasks stack_three --max-episodes 200 --jobs 1
python scripts/dataset/prepare_mimicgen_datasets.py --tasks threading --max-episodes 30 --jobs 1
python scripts/dataset/prepare_mimicgen_datasets.py --tasks three_piece_assembly --max-episodes 200 --jobs 1
```

The helper defaults to `external/mimicgen`. Supply `--mimicgen-root /path/to/mimicgen` if installed elsewhere. Use `--dry-run` to inspect download/conversion commands or `--check-only` to inspect existing artifacts. `--force` regenerates processed files; it does not delete raw demonstrations.

Downloading a BC checkpoint skips policy pretraining, but the demonstrations are still needed: `ph_pretrain/train.npz` supplies positive references, `ph_finetune/train.npz` supplies expert replay, and `normalization.npz` defines policy input/action scaling. The checkpoint repository does not bundle those data files.

The camera order is **agentview, robot0_eye_in_hand**, each at 96 × 96. The launcher limits policy and expert replay demonstrations to the counts above. Preparing more demonstrations also exposes those extra demonstrations to the default offline positive-buffer builder, which reads every episode in its NPZ; keep the preprocessing counts fixed when comparing runs.

## RoboMimic Tool Hang

Obtain the Tool Hang proficient-human raw HDF5 from the [RoboMimic datasets](https://robomimic.github.io/docs/datasets/overview.html). Use a file with simulator states, model metadata, actions, rewards, and proprioception. The commands below **render** the images so that the released modified Tool Hang camera XML is used consistently in demonstrations and online rollouts.

```bash
python scripts/dataset/process_robomimic_dataset.py \
  --load_path /absolute/path/to/tool_hang/demo.hdf5 \
  --save_dir "$HIRE_DATA_DIR/robomimic/tool_hang-img/ph_pretrain" \
  --normalize --max_episodes 20 --cameras sideview robot0_eye_in_hand

python scripts/dataset/process_robomimic_dataset.py \
  --load_path /absolute/path/to/tool_hang/demo.hdf5 \
  --save_dir "$HIRE_DATA_DIR/robomimic/tool_hang-img/ph_finetune" \
  --normalize --max_episodes 20 --cameras sideview robot0_eye_in_hand --truncate
```

Keep `tool_hang` in the raw file path: the converter uses that path to select the modified camera model. Camera order is **sideview, robot0_eye_in_hand**, at 240 × 240 per view. The XML's `robosuite/` asset markers are resolved by RoboMimic against the installed Robosuite package. Do not use `--use_hdf5_images` unless the stored cameras exactly match this setup.

## Processed layout

```text
data_dir/
  mimicgen/stack_three-img/
    ph_pretrain/
      train.npz
      normalization.npz
      dino_positive_buffer.pt       # created on first HiRE run
    ph_finetune/
      train.npz
      normalization.npz
```

The other tasks follow the same structure under their benchmark. Finetuning trajectories are truncated at the first successful transition. Policy normalization comes from `ph_pretrain/normalization.npz`; keep the pretraining and finetuning processing inputs, episode limit, camera order, and action convention identical. Use delta actions with seven action dimensions for the released configs.

`train.npz` stores concatenated trajectories:

| Key | Content |
| --- | --- |
| `states` | Normalized 9-D proprioception: end-effector position, quaternion, gripper state |
| `actions` | Normalized 7-D action sequence |
| `rewards` | Environment rewards |
| `traj_lengths` | Number of transitions per trajectory; their sum is the concatenated length |
| `images` | RGB views concatenated in the configured camera order, HWC layout, six channels |

The normalizer contains per-dimension observation/action minima and maxima. Image inputs to the frozen reward encoder use raw pixel values in the 0–255 range, not policy-normalized tensors. Optional `val.npz` can be generated with the converter's validation-split arguments; it is not required by the quick start.

## Positive buffer

HiRE automatically builds and caches the positive buffer from `ph_pretrain/train.npz`. To prepare it explicitly:

```bash
python scripts/dataset/build_robomimic_dino_positive_buffer.py \
  --dataset-path "$HIRE_DATA_DIR/mimicgen/stack_three-img/ph_pretrain/train.npz" \
  --output-path "$HIRE_DATA_DIR/mimicgen/stack_three-img/ph_pretrain/dino_positive_buffer.pt" \
  --camera-keys agentview_image robot0_eye_in_hand_image \
  --frame-stride 5 --device cuda:0
```

For Tool Hang use `sideview_image robot0_eye_in_hand_image`. For SigLIP add `--encoder-kind siglip` and use `siglip_positive_buffer.pt`.

## Checkpoint bundles

Download all released MimicGen bundles, or select tasks:

```bash
python scripts/download_checkpoints.py
python scripts/download_checkpoints.py --tasks threading three_piece_assembly
```

The default location is `checkpoints/HiRE-release/`; `HIRE_CHECKPOINT_DIR` changes it. `configs/checkpoints.json` lists the repository and per-task filenames. The downloader reuses existing files; `--force` downloads them again. Interrupted transfers are kept separate from completed files. Keep customized BC files separately and select them with `--checkpoint`.

A BC checkpoint needs its saved model configuration. Preserve:

```text
bc_run/
  .hydra/config.yaml
  checkpoint/state_100.pt
```

Finetuning instantiates the BC architecture from that saved config and loads the checkpoint's `model` weights. See [checkpoint weight selection](training.md#base-policy-and-checkpoint-weights) for details. The config must describe the same cameras, observation shape, action convention, and normalization as finetuning. Diffusion and flow checkpoints are not interchangeable. The filename's number is a BC epoch; `model_step_<step>.pth` uses the finetuning loop counter.
