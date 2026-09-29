# Reward computation in the training loop

This guide follows the implementation used by the four simulation tasks. Training commands and task settings are described in the [quick start](quickstart.md) and [training reference](training.md).

## Code locations

| Component | Implementation |
| --- | --- |
| DINOv2 and SigLIP encoders | `src/hire/encoders/vision.py` |
| Patch similarity and contrastive potential | `src/hire/potential.py` |
| Potential-based reward difference | `src/hire/shaping.py` |
| Online reference storage | `src/hire/buffers/fifo.py` |
| Offline references and positive sampling | `backends/dice_rl/src/hire_dice_rl/util/dino_prompt_buffer.py` |
| Online reward computation and episode bookkeeping | `backends/dice_rl/src/hire_dice_rl/integration/reward.py` |
| Configuration mapping | `backends/dice_rl/src/hire_dice_rl/integration/config.py` |

## Reward computation

1. The environment wrappers collect camera frames and sparse rewards for each executed action substep.
2. The training process encodes those frames with the frozen visual encoder and samples successful and failed reference embeddings.
3. The shaper computes patch-averaged cosine similarity and the contrastive potential, then applies potential differences and the configured dense coefficient.
4. Shaped rewards are written to both the returned chunk reward and the substep rewards in `info["full_trajectory"]`. The replay buffer uses those substep rewards when constructing training targets.
5. Completed episodes update the online success/failure reference buffers. The task's sparse success signal identifies successful episodes; timeouts without success supply failure references.

The reward shaper carries previous potentials between chunks and resets this state at episode boundaries. The experiment's action-chunk normalization and training-update schedule are retained. Numerical regression tests cover chunk aggregation, successful and failed episodes, and FIFO replacement.

## Reference buffers

Offline positives come from the pretraining NPZ, sampled with the configured frame stride. Online positives and negatives retain terminal observations from successful and failed rollouts. The online positive sampler mixes recent successes with offline demonstrations.

Actual terminal frames must be captured before an environment resets. Using the next episode's initial image would put the wrong observation in the hindsight buffer.

Optional buffer snapshots export reference embeddings and diagnostic information. They are separate from policy checkpoints and do not contain complete training-resume state.

## Configuration and checkpoints

The launcher reads `reward.*` and `encoder.*` settings; the DICE-RL integration maps them to the environment wrapper and reward shaper. Actor/critic optimization and replay construction remain in the training code.

The checkpoint loader translates saved `agent.*`, `model.*`, `env.*`, and `util.*` Hydra targets to `hire_dice_rl.*`. Model parameter names are unchanged. Keep `.hydra/config.yaml` with each checkpoint; old `DICE_RL_DATA_DIR` and `DICE_RL_LOG_DIR` variables remain accepted by the launch scripts.
