"""
Multi-step wrapper. Allow executing multiple environmnt steps. Returns stacked observation and optionally stacked previous action.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

"""

import gym
from typing import Optional
from gym import spaces
import numpy as np
from collections import defaultdict, deque


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max", gamma=0.99):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    """Apply padding"""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStep(gym.Wrapper):

    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        reward_agg_method="sum",  # never use other types
        prev_action=True,
        reset_within_step=False,
        pass_full_observations=False,
        verbose=False,
        gamma=0.99,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose
        self.gamma = gamma

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: dict = {},
    ):
        """Resets the environment."""
        obs = self.env.reset(
            seed=seed,
            options=options,
            return_info=return_info,
        )
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps)

        self.cnt = 0
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        if action.ndim == 1:  # in case action_steps = 1
            action = action[None]
        truncated = False
        terminated = False
        # Collect per-substep raw images / env rewards out-of-band so they
        # bypass dict_take_last_n aggregation (which assumes np.array-able).
        dino_chunk_images = []
        dino_chunk_env_frame_indices = []
        env_step_rewards = []
        for act_step, act in enumerate(action):
            self.cnt += 1
            if terminated or truncated:
                break

            # done does not differentiate terminal and truncation
            observation, reward, done, info = self.env.step(act)

            self.obs.append(observation)
            self.action.append(act)
            self.reward.append(reward)

            # Pop out-of-band fields before they go into the deque info.
            step_imgs = info.pop("dino_step_images", None)
            if step_imgs is not None:
                dino_chunk_images.append(step_imgs)
                dino_chunk_env_frame_indices.append(int(self.cnt))
            step_env_reward = info.pop("env_step_reward", None)
            if step_env_reward is not None:
                env_step_rewards.append(float(step_env_reward))

            # in gym, timelimit wrapper is automatically used given env._spec.max_episode_steps
            if "TimeLimit.truncated" not in info:
                if done:
                    terminated = True
                elif (
                    self.max_episode_steps is not None
                ) and self.cnt >= self.max_episode_steps:
                    truncated = True
            else:
                truncated = info["TimeLimit.truncated"]
                terminated = done
            done = truncated or terminated
            self.done.append(done)
            self._add_info(info)
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method, self.gamma)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if dino_chunk_images:
            info["dino_chunk_images"] = dino_chunk_images
            info["dino_chunk_env_frame_indices"] = dino_chunk_env_frame_indices
        if env_step_rewards:
            info["env_reward_chunk_sum"] = float(sum(env_step_rewards))
            info["success"] = float(max(env_step_rewards)) >= 1.0
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1)

        # In mujoco case, done can happen within the loop above
        if self.reset_within_step and self.done[-1]:

            # need to save old observation in the case of truncation only, for bootstrapping
            if truncated:
                info["final_obs"] = observation

            # reset
            observation = (
                self.reset()
            )  # TODO: arguments? this cannot handle video recording right now since needs to pass in options
            self.verbose and print("Reset env within wrapper.")

        # reset reward and done for next step
        self.reward = list()
        self.done = list()
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs)
