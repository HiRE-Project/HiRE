"""Convert RoboMimic-format HDF5 demonstrations to normalized NPZ datasets.

Adapted from RoboMimic dataset utilities. See docs/data.md for camera ordering,
pretraining and finetuning preparation commands, and normalization semantics.
"""


import numpy as np
from tqdm import tqdm
import h5py
import os
import random
from copy import deepcopy
import logging
import json

# Import robomimic utilities for environment creation and image rendering
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def render_trajectory_images(env, initial_state, states, camera_names, load_path):
    """
    Render images for a trajectory by setting environment states and capturing camera views.
    Based on playback_trajectory_with_env function from robomimic.
    
    Args:
        env: robomimic environment instance
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        camera_names (list): list of camera names to render
        
    Returns:
        dict: dictionary with camera_name -> images array (T, H, W, 3)
    """
    # Check if this is tool_hang and we need to modify camera
    if 'tool_hang' in load_path and 'robot0_eye_in_hand' in camera_names:
        # Load modified XML for tool_hang camera
        modified_xml_path = "configs/task/robomimic/tool_hang_model_modified.xml"
        if os.path.exists(modified_xml_path):
            logging.info(f"Using modified camera XML for tool_hang from {modified_xml_path}")
            with open(modified_xml_path, "r") as f:
                modified_xml = f.read()
            # Add modified XML to initial state
            initial_state["model"] = modified_xml
    
    # Reset environment to initial state (with potentially modified camera)
    env.reset_to(initial_state)
    
    traj_len = states.shape[0]
    rendered_images = {cam: [] for cam in camera_names}
    
    for i in range(traj_len):
        print(f"Rendering frame {i+1}/{traj_len}", end="\r")
        # Set environment to the state at timestep i
        env.reset_to({"states": states[i]})
        
        # Render each camera view
        for cam_name in camera_names:
            # Render camera view (returns RGB array as uint8)
            if 'transport' in load_path or 'square' in load_path or 'can' in load_path:
                img = env.render(mode="rgb_array", height=96, width=96, camera_name=cam_name)
            elif 'tool_hang' in load_path:
                img = env.render(mode="rgb_array", height=240, width=240, camera_name=cam_name)
                
            rendered_images[cam_name].append(img)
    
    # Convert lists to numpy arrays
    for cam in camera_names:
        rendered_images[cam] = np.array(rendered_images[cam])  # Shape: (T, H, W, 3)
    
    return rendered_images


def extract_images_from_hdf5(f, ep, cameras, traj_length, image_size=96):
    """
    Read RGB observations stored in the HDF5 and resize to the training resolution.
    Expects keys like ``agentview_image`` (no ``_image`` suffix in --cameras arg).
    """
    import cv2

    concatenated_images = []
    for t in range(traj_length):
        frame_cameras = []
        for cam in cameras:
            obs_key = cam if cam.endswith("_image") else f"{cam}_image"
            img = f[f"data/{ep}/obs/{obs_key}"][t]  # (H, W, 3) uint8
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frame_cameras.append(img)
        frame_concat = np.concatenate(frame_cameras, axis=2)  # (H, W, C)
        frame_concat = np.transpose(frame_concat, (2, 0, 1))  # (C, H, W)
        concatenated_images.append(frame_concat)
    return np.array(concatenated_images)


def convert_actions_to_6d(actions, rot_transformer):
    """Convert 7D actions with axis-angle to 10D actions with 6D rotation."""
    # actions shape: (T, 7) where 7 = [pos(3), axis_angle(3), gripper(1)]
    pos = actions[:, :3]
    axis_angle = actions[:, 3:6]
    gripper = actions[:, 6:7]
    
    # Convert axis-angle to 6D rotation
    rot6d = rot_transformer.forward(axis_angle)
    
    # Concatenate to form 10D action
    return np.concatenate([pos, rot6d, gripper], axis=-1)


def _episode_split_indices(num_traj, val_split, split_mode, num_val_episodes=None):
    """Return (train_indices, val_indices) as sets of episode indices 0..num_traj-1."""
    if num_val_episodes is not None:
        if num_val_episodes <= 0:
            return set(range(num_traj)), set()
        num_val = min(int(num_val_episodes), num_traj)
    elif val_split <= 0:
        return set(range(num_traj)), set()
    else:
        num_val = int(num_traj * val_split)
    if split_mode == "last":
        if num_val == 0 and num_traj > 0:
            num_val = 1
        num_train = num_traj - num_val
        val_indices = set(range(num_train, num_traj))
        train_indices = set(range(num_train))
    else:
        num_train = int(num_traj * (1 - val_split))
        train_indices = set(random.sample(range(num_traj), k=num_train))
        val_indices = set(range(num_traj)) - train_indices
    return train_indices, val_indices


def make_dataset(load_path, save_dir, save_name_prefix, val_split, normalize, use_6d_rot=False, is_abs_action=False):
    # Validate 6D rotation requirements
    if use_6d_rot and not is_abs_action:
        raise ValueError("6D rotation representation requires absolute actions. Please use --is_abs_action with --use_6d_rot")
    
    if use_6d_rot:
        logging.info("6D rotation conversion enabled - will convert axis-angle to 6D representation")
        logging.info("Note: Expecting absolute actions (not deltas)")
    
    # Initialize environment for image rendering if cameras are specified
    env = None
    is_robosuite_env = False
    if args.cameras is not None and not args.use_hdf5_images:
        logging.info(f"Initializing environment for image rendering with cameras: {args.cameras}")
        
        # Set up observation specs for robomimic
        dummy_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
        
        # Create environment from dataset metadata
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=load_path)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
        logging.info(f"Environment initialized: {env_meta['env_name']}")
    elif args.cameras is not None and args.use_hdf5_images:
        logging.info(
            f"Reading images from HDF5 (no sim re-render) with cameras: {args.cameras}"
        )

    # Load hdf5 file from load_path
    with h5py.File(load_path, "r") as f:
        # Sort demonstrations in increasing episode order
        demos = sorted(list(f["data"].keys()))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        if args.max_episodes > 0:
            demos = demos[: args.max_episodes]

        # Default low-dimensional observation keys
        low_dim_obs_names = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        if "transport" in load_path:
            low_dim_obs_names += [
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ]
        if args.cameras is None:
            low_dim_obs_names.append("object")

        # Calculate dimensions for observations and actions
        obs_dim = 0
        for low_dim_obs_name in low_dim_obs_names:
            dim = f[f"data/demo_0/obs/{low_dim_obs_name}"].shape[1]
            obs_dim += dim
            logging.info(f"Using {low_dim_obs_name} with dim {dim} for observation")

        original_action_dim = f["data/demo_0/actions"].shape[1]
        
        # Check if we need to adjust action dimension for 6D rotation
        if use_6d_rot:
            if original_action_dim != 7:
                raise ValueError(f"Expected 7D actions for 6D rotation conversion, got {original_action_dim}D")
            action_dim = 10  # pos(3) + rot6d(6) + gripper(1)
            logging.info(f"Converting from {original_action_dim}D (axis-angle) to {action_dim}D (6D rotation)")
        else:
            action_dim = original_action_dim
            
        logging.info(f"Total low-dim observation dim: {obs_dim}")
        logging.info(f"Original action dim: {original_action_dim}")
        logging.info(f"Final action dim: {action_dim}")

        # Initialize variables for tracking trajectory statistics
        traj_lengths = []
        obs_min = np.zeros((obs_dim))
        obs_max = np.zeros((obs_dim))
        action_min = np.zeros((action_dim))
        action_max = np.zeros((action_dim))

        # Import rotation transformer if needed
        if use_6d_rot:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from hire_dice_rl.util.rotation_conversion import RotationTransformer
            rot_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
        
        val_only = getattr(args, "val_only", False)
        split_mode = getattr(args, "split_mode", "random")

        num_val_episodes = getattr(args, "num_val_episodes", None)
        if num_val_episodes is not None and num_val_episodes < 0:
            num_val_episodes = None

        if val_only and val_split <= 0 and not num_val_episodes:
            raise ValueError(
                "--val_only requires --val_split > 0 or --num_val_episodes > 0"
            )

        normalization_save_path = os.path.join(
            save_dir, save_name_prefix + "normalization.npz"
        )
        if val_only and normalize:
            if not os.path.isfile(normalization_save_path):
                raise FileNotFoundError(
                    f"--val_only with --normalize requires existing {normalization_save_path} "
                    "(from the original train.npz processing run)."
                )
            norm = np.load(normalization_save_path)
            obs_min = norm["obs_min"]
            obs_max = norm["obs_max"]
            action_min = norm["action_min"]
            action_max = norm["action_max"]
            logging.info(f"Loaded normalization stats from {normalization_save_path}")
        else:
            # Scan all demos for min/max stats (also collects traj_lengths).
            for ep in demos:
                traj_lengths.append(f[f"data/{ep}/actions"].shape[0])
                obs = np.hstack(
                    [
                        f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                        for low_dim_obs_name in low_dim_obs_names
                    ]
                )
                raw_actions = f[f"data/{ep}/actions"][()]

                if use_6d_rot:
                    actions = convert_actions_to_6d(raw_actions, rot_transformer)
                else:
                    actions = raw_actions

                obs_min = np.minimum(obs_min, np.min(obs, axis=0))
                obs_max = np.maximum(obs_max, np.max(obs, axis=0))
                action_min = np.minimum(action_min, np.min(actions, axis=0))
                action_max = np.maximum(action_max, np.max(actions, axis=0))

        if val_only:
            traj_lengths = np.array(
                [f[f"data/{ep}/actions"].shape[0] for ep in demos]
            )
        else:
            traj_lengths = np.array(traj_lengths)

        # Report statistics
        logging.info("===== Basic stats =====")
        logging.info(f"Total transitions: {np.sum(traj_lengths)}")
        logging.info(f"Total trajectories: {len(traj_lengths)}")
        logging.info(
            f"Traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
        )
        logging.info(
            f"Traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
        )
        logging.info(f"obs min: {obs_min}")
        logging.info(f"obs max: {obs_max}")
        logging.info(f"action min: {action_min}")
        logging.info(f"action max: {action_max}")

        # Split indices into train and validation sets
        num_traj = len(traj_lengths)
        train_indices, val_indices = _episode_split_indices(
            num_traj, val_split, split_mode, num_val_episodes=num_val_episodes
        )
        if val_only:
            train_indices = set()
        if split_mode == "last" and len(val_indices) > 0:
            logging.info(
                f"Split mode 'last': val episodes are indices "
                f"{min(val_indices)}..{max(val_indices)} "
                f"({len(val_indices)} trajectories)"
            )

        # Initialize output dictionaries for train and val sets
        out_train = {"states": [], "actions": [], "rewards": [], "traj_lengths": []}
        out_val = deepcopy(out_train)
        
        # Add images if cameras are specified
        if args.cameras is not None:
            out_train["images"] = []
            out_val["images"] = []

        # Track truncation statistics
        original_total_steps = 0
        truncated_total_steps = 0
        
        # Process each demo
        for i in tqdm(range(len(demos))):
        # for i in tqdm(range(5)):
            if i not in train_indices and i not in val_indices:
                continue
            ep = demos[i]
            out = out_train if i in train_indices else out_val

            # Get trajectory data
            original_traj_length = f[f"data/{ep}"].attrs["num_samples"]
            traj_length = original_traj_length
            
            raw_actions = f[f"data/{ep}/actions"][()]
            rewards = f[f"data/{ep}/rewards"][()]
            raw_obs = np.hstack(
                [
                    f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                    for low_dim_obs_name in low_dim_obs_names
                ]
            )
            
            # Track original steps
            original_total_steps += original_traj_length
            
            # Debug: check reward distribution
            non_zero_rewards = (rewards != 0).sum()
            # Optionally truncate trajectory after cumulative reward reaches 1
            if args.truncate:
                cumsum_rewards = np.cumsum(rewards)
                truncate_idx = np.where(cumsum_rewards >= 1)[0]
                
                if len(truncate_idx) > 0:
                    # Truncate at first point where cumulative reward >= 1
                    truncate_at = truncate_idx[0] + 1  # +1 to include the step that reached threshold
                    traj_length = truncate_at
                    raw_actions = raw_actions[:truncate_at]
                    rewards = rewards[:truncate_at]
                    raw_obs = raw_obs[:truncate_at]
            
            # Track truncated steps    
            truncated_total_steps += traj_length
            
            # Log and assert cumulative reward
            final_cumulative_reward = np.sum(rewards)
            if i < 100:  # Log first 10 trajectories for visibility
                logging.info(f"Trajectory {ep}: cumulative reward = {final_cumulative_reward:.1f} (length: {original_traj_length} -> {traj_length})")
            # assert final_cumulative_reward <= 5.0, f"Trajectory {ep} has cumulative reward {final_cumulative_reward} > 5!"
                
            out["traj_lengths"].append(traj_length)
            
            # Convert actions to 6D rotation if needed (before normalization)
            if use_6d_rot:
                raw_actions = convert_actions_to_6d(raw_actions, rot_transformer)

            # Normalize if specified
            if normalize:
                obs = 2 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
                actions = (
                    2 * (raw_actions - action_min) / (action_max - action_min + 1e-6)
                    - 1
                )
            else:
                obs = raw_obs
                actions = raw_actions

            # Load or render images if cameras are specified
            if args.cameras is not None:
                if args.use_hdf5_images:
                    concatenated_images = extract_images_from_hdf5(
                        f, ep, args.cameras, traj_length
                    )
                    logging.info(
                        f"Loaded HDF5 images for {ep}: shape {concatenated_images.shape}"
                    )
                else:
                    # Get states for this trajectory
                    states = f[f"data/{ep}/states"][()]

                    # Truncate states to match truncated trajectory length (if truncation is enabled)
                    if args.truncate:
                        states = states[:traj_length]

                    # Prepare initial state for robosuite environment
                    initial_state = dict(states=states[0])
                    if is_robosuite_env:
                        initial_state["model"] = f[f"data/{ep}"].attrs["model_file"]
                        initial_state["ep_meta"] = f[f"data/{ep}"].attrs.get("ep_meta", None)

                    # Render images for all camera views
                    rendered_images = render_trajectory_images(
                        env, initial_state, states, args.cameras, load_path
                    )

                    # Concatenate camera views: convert from dict to single array
                    concatenated_images = []
                    for t in range(traj_length):
                        frame_cameras = []
                        for cam in args.cameras:
                            frame_cameras.append(rendered_images[cam][t])  # (H, W, 3)
                        frame_concat = np.concatenate(frame_cameras, axis=2)
                        frame_concat = np.transpose(frame_concat, (2, 0, 1))
                        concatenated_images.append(frame_concat)

                    concatenated_images = np.array(concatenated_images)
                    logging.info(
                        f"Rendered images for {ep}: shape {concatenated_images.shape}"
                    )
                out["images"].append(concatenated_images)

            # Store trajectories in output dictionary
            out["states"].append(obs)
            out["actions"].append(actions)
            out["rewards"].append(rewards)

        # Concatenate trajectories (no padding)
        keys_to_concat = ["states", "actions", "rewards"]
        if args.cameras is not None:
            keys_to_concat.append("images")
            
        if not val_only:
            for key in keys_to_concat:
                out_train[key] = np.concatenate(out_train[key], axis=0)

        if len(val_indices) > 0:
            for key in keys_to_concat:
                out_val[key] = np.concatenate(out_val[key], axis=0)

        # Save datasets as npz files
        if not val_only:
            train_save_path = os.path.join(save_dir, save_name_prefix + "train.npz")
            train_data = {
                "states": np.array(out_train["states"]),
                "actions": np.array(out_train["actions"]),
                "rewards": np.array(out_train["rewards"]),
                "terminals": np.array([False] * len(out_train["states"])),
                "traj_lengths": np.array(out_train["traj_lengths"]),
            }
            if args.cameras is not None:
                train_data["images"] = np.array(out_train["images"])
            np.savez_compressed(train_save_path, **train_data)

        if len(val_indices) > 0:
            val_save_path = os.path.join(save_dir, save_name_prefix + "val.npz")
            val_data = {
                "states": np.array(out_val["states"]),
                "actions": np.array(out_val["actions"]),
                "rewards": np.array(out_val["rewards"]),
                "terminals": np.array([False] * len(out_val["states"])),
                "traj_lengths": np.array(out_val["traj_lengths"]),
            }
            if args.cameras is not None:
                val_data["images"] = np.array(out_val["images"])
            np.savez_compressed(val_save_path, **val_data)
            logging.info(f"Saved validation set to {val_save_path}")

        # Save normalization stats if required (full run only)
        if normalize and not val_only:
            np.savez_compressed(
                normalization_save_path,
                obs_min=obs_min,
                obs_max=obs_max,
                action_min=action_min,
                action_max=action_max,
            )

        # Logging final information
        if args.truncate:
            logging.info("===== Truncation Statistics =====")
            logging.info(f"Original total steps: {original_total_steps}")
            logging.info(f"Truncated total steps: {truncated_total_steps}")
            logging.info(f"Reduction: {original_total_steps - truncated_total_steps} steps ({100 * (1 - truncated_total_steps/original_total_steps):.1f}%)")
        else:
            logging.info("===== No Truncation Applied =====")
            logging.info(f"Total steps: {original_total_steps}")
        
        if not val_only:
            logging.info(
                f"Train - Trajectories: {len(out_train['traj_lengths'])}, "
                f"Transitions: {np.sum(out_train['traj_lengths'])}"
            )
        if len(val_indices) > 0:
            logging.info(
                f"Val - Trajectories: {len(out_val['traj_lengths'])}, "
                f"Transitions: {np.sum(out_val['traj_lengths'])}"
            )

        # Log image statistics if cameras were used
        if args.cameras is not None and len(val_indices) > 0:
            val_images_shape = out_val["images"].shape
            logging.info(f"Val images shape: {val_images_shape}")
            logging.info(
                f"Val image value range: [{out_val['images'].min()}, {out_val['images'].max()}]"
            )
            logging.info(f"Cameras used: {args.cameras}")

            num_cameras = len(args.cameras)
            channels_per_camera = 3
            total_channels = val_images_shape[1]
            expected_channels = num_cameras * channels_per_camera
            if total_channels == expected_channels:
                logging.info(
                    f"Image channels match expected: {total_channels} = "
                    f"{num_cameras} cameras × {channels_per_camera} RGB"
                )
            else:
                logging.warning(
                    f"Image channels mismatch: got {total_channels}, "
                    f"expected {expected_channels}"
                )
        elif args.cameras is not None and not val_only:
            train_images_shape = out_train["images"].shape
            logging.info(f"Train images shape: {train_images_shape}")
            logging.info(
                f"Image value range: [{out_train['images'].min()}, {out_train['images'].max()}]"
            )
            logging.info(f"Cameras used: {args.cameras}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default="0")
    parser.add_argument(
        "--num_val_episodes",
        type=int,
        default=-1,
        help=(
            "Hold out this many trailing demos for val (overrides --val_split). "
            "Use with --split_mode last for deterministic holdout."
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "last"],
        help="'last' assigns the final val demos to val (deterministic).",
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="Only write val.npz (reuse existing normalization.npz). Does not touch train.npz.",
    )
    parser.add_argument("--max_episodes", type=int, default="-1")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cameras", nargs="*", default=None) # no "_image" suffix after camera name
    parser.add_argument("--use_6d_rot", action="store_true", 
                        help="Convert axis-angle to 6D rotation representation (requires absolute actions)")
    parser.add_argument("--is_abs_action", action="store_true",
                        help="Whether the dataset contains absolute actions (not deltas)")
    parser.add_argument("--truncate", action="store_true",
                        help="Truncate trajectories after first positive reward")
    parser.add_argument(
        "--use_hdf5_images",
        action="store_true",
        help="Read camera images from HDF5 obs keys instead of re-rendering in sim (faster for MimicGen datasets)",
    )
    args = parser.parse_args()

    import datetime

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    make_dataset(
        args.load_path,
        args.save_dir,
        args.save_name_prefix,
        args.val_split,
        args.normalize,
        args.use_6d_rot,
        args.is_abs_action,
    )
