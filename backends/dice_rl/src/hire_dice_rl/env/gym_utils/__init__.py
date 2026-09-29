"""Vectorized RoboMimic and MimicGen environments for image policies."""
import os
import json


def make_async(
    id, num_envs=1, asynchronous=True, wrappers=None, render=False,
    obs_dim=9, action_dim=7, env_type=None, max_episode_steps=None,
    robomimic_env_cfg_path=None, use_image_obs=False,
    render_offscreen=False, reward_shaping=False, shape_meta=None, **kwargs,
):
    """Create independent simulator workers and apply the configured wrappers."""
    if robomimic_env_cfg_path is None:
        raise ValueError("A RoboMimic or MimicGen environment metadata JSON is required.")
    if env_type not in (None, "robomimic", "mimicgen"):
        raise ValueError(f"Unsupported environment type: {env_type}")

    # avoid import error due incompatible gym versions
    from gym import spaces
    from hire_dice_rl.env.gym_utils.async_vector_env import AsyncVectorEnv
    from hire_dice_rl.env.gym_utils.sync_vector_env import SyncVectorEnv
    from hire_dice_rl.env.gym_utils.wrapper import wrapper_dict

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    def _make_env():
        if robomimic_env_cfg_path is not None:
            obs_modality_dict = {
                "low_dim": (
                    wrappers.robomimic_image.low_dim_keys
                    if "robomimic_image" in wrappers
                    else wrappers.robomimic_lowdim.low_dim_keys
                ),
                "rgb": (
                    wrappers.robomimic_image.image_keys
                    if "robomimic_image" in wrappers
                    else None
                ),
            }
            if obs_modality_dict["rgb"] is None:
                obs_modality_dict.pop("rgb")
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            os.environ.setdefault("MUJOCO_GL", "egl")
            if render_offscreen or use_image_obs:
                # Ensure EGL uses the correct GPU device
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    # Map CUDA device to EGL device
                    cuda_device = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]
                    os.environ["EGL_DEVICE_ID"] = cuda_device
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = cuda_device
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            if env_meta["env_name"] in {"StackThree_D0", "Threading_D0", "ThreePieceAssembly_D0"}:
                import mimicgen  # noqa: F401: register custom Robosuite tasks
            env_meta["env_kwargs"]["reward_shaping"] = reward_shaping
            
            # Check if we should use absolute actions (from kwargs)
            if kwargs.get("abs_action", False):
                env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
                # render_gpu_device_id=0,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            env.env.hard_reset = False

        # add wrappers
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    def dummy_env_fn():
        """Create observation metadata without a main-process OpenGL context."""
        import gym
        import numpy as np
        from hire_dice_rl.env.gym_utils.wrapper.multi_step import MultiStep
        from hire_dice_rl.env.gym_utils.wrapper.multi_step_full import MultiStepFull

        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpose is to provide
        # obs/action spaces and metadata.
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:  # rn only for images
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space["state"] = gym.spaces.Box(
                -1,
                1,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.int64)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        # Handle both multi_step and multi_step_full (for upsampling)
        if 'multi_step' in wrappers:
            return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)
        elif 'multi_step_full' in wrappers:
            # Use MultiStep for dummy env even when MultiStepFull is configured
            # This is fine since dummy env is only for metadata
            return MultiStepFull(env=env, n_obs_steps=wrappers.multi_step_full.n_obs_steps)
        else:
            return env

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(
                dummy_env_fn if render or render_offscreen or use_image_obs else None
            ),
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )
