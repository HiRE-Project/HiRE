import argparse
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hire_dice_rl.util.dino_prompt_buffer import RobomimicNpzDinoPositiveBufferBuilder
from hire_dice_rl.util.similarity_encoder import build_similarity_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a visual positive prompt buffer from a processed Robomimic NPZ. "
            "Supports DINOv2 (default), LIV, and SigLIP via --encoder-kind."
        )
    )
    parser.add_argument(
        "--encoder-kind",
        default="dino",
        choices=["dino", "liv", "siglip"],
        help="Similarity encoder: dino | liv | siglip (default: dino).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Processed DICE-RL Robomimic NPZ, e.g. data_dir/mimicgen/stack_three-img/ph_pretrain/train.npz",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output .pt path for the DINO positive buffer.",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=["agentview_image", "robot0_eye_in_hand_image"],
        help="Camera keys in the same order as image channels in the NPZ.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for DINO encoding.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Use one frame every N frames inside each trajectory.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Limit number of trajectories (0 means all).",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=0,
        help="Limit sampled frames per trajectory (0 means all).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Batch size for DINO encoding.",
    )
    parser.add_argument(
        "--no-save-images-in-buffer",
        action="store_true",
        help="Omit raw HWC frames from the .pt (embeddings only). Default saves images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s build_robomimic_dino_positive_buffer] %(message)s",
        datefmt="%H:%M:%S",
    )
    encoder = build_similarity_encoder(args.encoder_kind, device=args.device)
    builder = RobomimicNpzDinoPositiveBufferBuilder(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        camera_keys=args.camera_keys,
        encoder=encoder,
        device=str(encoder.device),
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        encode_batch_size=args.encode_batch_size,
        encoder_kind=args.encoder_kind,
        save_images_in_buffer=not args.no_save_images_in_buffer,
    )
    builder.build()


if __name__ == "__main__":
    main()

