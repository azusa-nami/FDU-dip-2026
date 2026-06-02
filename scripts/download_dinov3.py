"""Download DINOv3 ViT-B/16 for offline use.

Model: facebook/dinov3-vitb16-pretrain-lvd1689m
Hidden size: 768 (matches ERRNet code)
Parameters:   86M

Prerequisite:
  1. Accept the license at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
  2. huggingface-cli login  (or set HF_TOKEN env var)

Usage:
  python scripts/download_dinov3.py                          # default path
  python scripts/download_dinov3.py --save-dir /path/to/dinov3
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 ViT-B/16")
    parser.add_argument(
        "--save-dir",
        default="/oldhome/zengyuqi/model/dinov3",
        help="local directory to save the model",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Downloading {args.model_id} ...")
    print(f"Save to: {args.save_dir}")
    print()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        # use_auth_token=True,  # if needed for gated model
    )
    model.save_pretrained(args.save_dir)
    print(f"Done. Model saved to {args.save_dir}")
    print(f"Files: {os.listdir(args.save_dir)}")


if __name__ == "__main__":
    main()
