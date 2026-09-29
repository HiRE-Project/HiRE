#!/usr/bin/env bash
# Source from the repository root; existing settings are preserved.
export HIRE_DATA_DIR="${HIRE_DATA_DIR:-${DICE_RL_DATA_DIR:-${PWD}/data_dir}}"
export HIRE_LOG_DIR="${HIRE_LOG_DIR:-${DICE_RL_LOG_DIR:-${PWD}/log_dir}}"
export DICE_RL_DATA_DIR="${DICE_RL_DATA_DIR:-${HIRE_DATA_DIR}}"
export DICE_RL_LOG_DIR="${DICE_RL_LOG_DIR:-${HIRE_LOG_DIR}}"
export HIRE_CHECKPOINT_DIR="${HIRE_CHECKPOINT_DIR:-${PWD}/checkpoints/HiRE-release}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
