# DICE-RL experiment reference

## Base checkpoints

The launcher defaults to the released epoch-75 Stack Three, epoch-100 Threading, and epoch-75 Three Piece Assembly checkpoints from [JimmyHan2004/HiRE-release](https://huggingface.co/JimmyHan2004/HiRE-release). Download them with `python scripts/download_checkpoints.py`. Their saved architectures and weights are loaded from `checkpoints/HiRE-release/`, or from `HIRE_CHECKPOINT_DIR`.

`--checkpoint` selects a custom BC bundle. Tool Hang has no published checkpoint and always requires this argument. Configuration previews (`--config-only` and `--dry-run`) do not require files or perform downloads. Training checks the selected checkpoint/config, required datasets, normalizer, and CUDA device before starting environments or W&B. Use `--check` to run these input checks separately.

## Configuration

`scripts/launch.py` combines task YAMLs in `configs/` with the settings below. The HiRE and sparse-reward recipes share the selected base policy, demonstration count, optimizer, and evaluation settings.

| Setting | Value |
| --- | --- |
| Pretraining / expert demonstration count | 200 / 30 / 200 / 20 for Stack Three / Threading / Three Piece Assembly / Tool Hang |
| BC loss weight during finetuning | 50 for MimicGen; 100 for Tool Hang |
| Evaluation interval | 200 finetuning loop steps |
| Dense coefficient for HiRE | `1 - recent_success_rate` (`reward.schedule.maximum=1`, minimum=0, exponent=1) |
| Successful online replay | Keep shaped rewards for MimicGen; sparse relabeling enabled for Tool Hang |
| Expert replay | Sparse environment reward only |
| Sparse comparison | Disable DINO computation and dense shaping |


Use `--config-only` to inspect the resolved configuration or `--dry-run` to print the command. Additional Hydra `key=value` arguments take precedence over the recipe. The saved `.hydra/config.yaml` records the configuration for each run.

Reward and encoder settings live in `configs/reward/` and `configs/encoder/`; DICE-RL replay and optimizer settings live in `configs/backend/`. Task experiment compositions live in `configs/experiment/`.

## Main-figure runs

The finetuning presets match the logged settings of the HiRE runs below. Each command starts a new run; it does not resume or overwrite these W&B records.

| Task | Seeds | BC checkpoint | Contrastive weight | Negative buffer | Successful replay |
| --- | --- | --- | --- | --- | --- |
| Stack Three | 42, 123, 345 | Epoch 75 | 0.9 | 128 | Shaped |
| Threading | 42, 123, 456 | Epoch 100 | 0.9 | 128 | Shaped |
| Three Piece Assembly | 42, 123, 456 | Epoch 75 | 0.9 | 128 | Shaped |
| Tool Hang | 42, 62, 82 | Epoch 500, supplied separately | 0.3 | 1024 | Sparse relabeling |

Threading seed 42 automatically selects `threading-img-256.json`, matching its recorded 256x256 environment rendering. The wrapper resizes policy observations to 96x96, so it uses the same released BC policy as the other seeds. The MimicGen presets enable evaluation videos and reward-buffer snapshots every 200 training iterations; Tool Hang keeps both disabled, as recorded.

With data and checkpoints prepared, launch one of the recorded configurations:

```bash
python scripts/launch.py finetune stack_three --seed 345 --wandb online
python scripts/launch.py finetune threading --seed 42 --wandb online
python scripts/launch.py finetune three_piece_assembly --seed 456 --wandb online
python scripts/launch.py finetune tool_hang --seed 62 --wandb online \
  --checkpoint /path/to/tool_hang_bc/checkpoint/state_500.pt
```

Repeat with the other seeds in the table for three-seed comparisons. `--wandb online` enables logging even if the shell defaults to disabled; `DICE_RL_WANDB_ENTITY` selects your W&B account/team. Run `wandb login` before using online logging. Each new run name includes its seed.

Recorded W&B runs:

- Stack Three: [seed 42](https://wandb.ai/qizhi_1/mimicgen-stack_three-distill-residual-diffusion-img/runs/mnue6bke), [seed 123](https://wandb.ai/qizhi_1/mimicgen-stack_three-distill-residual-diffusion-img/runs/10z8yupk), [seed 345](https://wandb.ai/qizhi_1/mimicgen-stack_three-distill-residual-diffusion-img/runs/lc06sfv8)
- Threading: [seed 42](https://wandb.ai/qizhi_1/mimicgen-threading-distill-residual-diffusion-img/runs/rzxf9fy9), [seed 123](https://wandb.ai/qizhi_1/mimicgen-threading-distill-residual-diffusion-img/runs/7i4gf0vt), [seed 456](https://wandb.ai/qizhi_1/mimicgen-threading-distill-residual-diffusion-img/runs/gxaucg5k)
- Three Piece Assembly: [seed 42](https://wandb.ai/qizhi_1/mimicgen-three_piece_assembly-distill-residual-diffusion-img/runs/0arlunev), [seed 123](https://wandb.ai/qizhi_1/mimicgen-three_piece_assembly-distill-residual-diffusion-img/runs/r12y79fp), [seed 456](https://wandb.ai/qizhi_1/mimicgen-three_piece_assembly-distill-residual-diffusion-img/runs/owhy5zir)
- Tool Hang: [seed 42](https://wandb.ai/t6-thu/robomimic-tool_hang-distill-residual-flow-img/runs/u64arocv), [seed 62](https://wandb.ai/t6-thu/robomimic-tool_hang-distill-residual-flow-img/runs/o6ckyb2o), [seed 82](https://wandb.ai/t6-thu/robomimic-tool_hang-distill-residual-flow-img/runs/hvv082q8)

The `--reward sparse` option disables HiRE using the same task/seed preset. It is a controlled comparison from that configuration, not an automatic selection of historical sparse-baseline runs.

## Implementation map

| Paper component | Code |
| --- | --- |
| Frozen DINOv2 / SigLIP visual representation | `src/hire/encoders/vision.py` |
| Offline success references and online FIFO buffers | `backends/dice_rl/src/hire_dice_rl/util/dino_prompt_buffer.py`, `src/hire/buffers/fifo.py` |
| Patch cosine similarity, contrastive potential, PBRS | `src/hire/potential.py`, `src/hire/shaping.py` |
| Reward integration into online rollouts | `backends/dice_rl/src/hire_dice_rl/agent/finetune/train_distill_residual_flow_img_agent.py` |
| Sparse expert replay and online transition sampling | `backends/dice_rl/src/hire_dice_rl/agent/dataset/sequence.py`, `backends/dice_rl/src/hire_dice_rl/util/hybrid_replay_buffer.py` |
| Residual actor and ensemble critic | `backends/dice_rl/src/hire_dice_rl/model/rl/distill_residual_rl.py`, `backends/dice_rl/src/hire_dice_rl/model/rl/distill_residual_rl_img.py` |
| Action chunks, terminal observations, camera images | `backends/dice_rl/src/hire_dice_rl/env/gym_utils/wrapper/` |
| Shared reward and replay settings | `configs/reward/` |

For each camera, the shaper compares corresponding normalized patch tokens and averages cosine similarity across patches. It aggregates sampled positive and negative references, subtracts the negative score scaled by `reward.contrastive_weight`, averages across cameras, and applies temporal potential differences. Successful and failed terminal observations update separate FIFO buffers. Expert demonstrations initialize the offline positive buffer and supply sparse expert replay.

## Implementation details

### Reward aggregation

The implementation computes `logsumexp(beta * similarities) / beta` over sampled references, with `reward.temperature=10` shared by both buffers and `reward.contrastive_weight=0.9` for MimicGen and `0.3` for Tool Hang. This differs from the paper's log-expectation formulation with separate concentration parameters: the code does not subtract `log(K) / beta`. These offsets can affect shaping when discounting or sampled support changes. The optional `kappa_ratio` mode uses a ratio of aggregate scores, not separate KDE temperatures.

The DICE-RL adapter carries previous potentials across action chunks, resets episode state, and updates reference buffers online. See the [reward computation flow](integration.md#reward-computation). These adaptive operations should be distinguished from the fixed-potential assumptions of the standard policy-invariance result.

### Base policy and checkpoint weights

Use diffusion checkpoints for the MimicGen tasks and flow-matching checkpoints for Tool Hang. The loader selects the checkpoint's `model` weights and freezes the pretrained policy, including its visual encoder, during residual finetuning. BC pretraining also maintains EMA weights with a warmup schedule. The paper describes EMA initialization and an end-to-end trainable visual policy; those settings differ from this implementation.

### Step accounting

`train.num_train_steps` and `train.eval_freq` count vector action-chunk iterations. The logger multiplies this counter by `horizon_steps * n_envs` for environment-step plots. Early episode termination can shorten executed chunks.

For reproducible comparisons, retain the resolved configuration, base checkpoint, normalizer, demonstration subset, random seed, and dependency versions with each run.

## Ablations

Append these overrides to a HiRE finetuning command:

| Experiment | Override |
| --- | --- |
| SigLIP encoder | `encoder=siglip` (install `[siglip]`) |
| Positive references only | `reward.mode=positive` |
| Larger negative buffer | `reward.negative_buffer.buffer_size=512` |
| Contrastive weight | `reward.contrastive_weight=0.3` |
| Direct potential addition | `reward.shaping.pbrs=false` |
| Buffer snapshots | `dino_buffer_snapshot.enabled=true` |

Keep reward scaling, replay settings, and evaluation seeds fixed when comparing ablations. Buffer snapshots store diagnostic reward references; they are separate from policy checkpoints and do not contain complete training-resume state.

## Execution checks

Check input files and the compute device before launching training:

```bash
python scripts/launch.py finetune stack_three --check
python scripts/launch.py finetune threading --seed 42 --config-only
```

`--check` does not start a simulator or write a W&B run. It reports missing data, normalizers, BC files, and unavailable CUDA devices. Passing it is an input check, not a simulator rollout test.

Check source structure and run the CPU test suite:

```bash
python scripts/check_release.py
python -m pytest -q
```

The CPU suite compares all 12 task/seed presets against their exported W&B settings and covers checkpoint loading, policy loss/backpropagation and sampling, FIFO buffers, and reward mathematics. It uses synthetic observations without downloading reward-encoder weights or running simulators.

For a short GPU pipeline check with prepared data and a compatible base checkpoint:

```bash
python scripts/launch.py finetune stack_three \
  train.num_train_steps=2 env.n_envs=1 run_eval=false \
  train.batch_size=2 train.gradient_steps=1
```

This checks execution; meaningful learning and success-rate comparisons require a full training and evaluation run.
