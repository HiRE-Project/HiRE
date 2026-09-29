"""Training entry point for the optional DICE-RL backend."""
import hydra
from .integration.config import configure_reward
from .util.wandb_utils import finish_wandb
from .check_inputs import check_inputs


def run(cfg):
    cfg = configure_reward(cfg)
    paths = check_inputs(cfg)
    if cfg.get("check_inputs_only", False):
        for label, path in paths.items():
            print(f"OK {label}: {path}")
        print("Input files and compute device are available; no simulator was started.")
        return
    cls = hydra.utils.get_class(cfg._target_)
    agent = None
    exit_code = 0
    try:
        agent = cls(cfg)
        agent.run()
    except KeyboardInterrupt:
        exit_code = 130
        raise
    except Exception:
        exit_code = 1
        raise
    finally:
        if agent is not None and hasattr(agent, "close"):
            agent.close()
        finish_wandb(exit_code=exit_code)
