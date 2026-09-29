"""Run a configured pretraining or finetuning experiment."""
import os
from pathlib import Path
import hydra

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
for new, old, default in (("HIRE_DATA_DIR", "DICE_RL_DATA_DIR", "data_dir"),
                          ("HIRE_LOG_DIR", "DICE_RL_LOG_DIR", "log_dir")):
    os.environ.setdefault(new, os.environ.get(old, str(ROOT / default)))
    os.environ.setdefault(old, os.environ[new])


@hydra.main(version_base=None, config_path=str(ROOT / "configs"))
def main(cfg):
    try:
        from hire_dice_rl.runner import run
    except ImportError as exc:
        raise ImportError("Install the experiment backend: pip install -e './backends/dice_rl[sim]'") from exc
    run(cfg)


if __name__ == "__main__":
    main()
