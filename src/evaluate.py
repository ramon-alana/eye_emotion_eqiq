from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torchvision import transforms

from data import EyeIQDataset
from models import EyeIQNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--num-emotions", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = EyeIQDataset(metadata_csv=args.metadata, image_root=args.image_root)
    model = EyeIQNet(num_emotions=args.num_emotions, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            outputs = model(image)
            emotion = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy().tolist()[0]
            print(
                {
                    "image": str(sample.get("image_path", "N/A")),
                    "emotion_probs": emotion,
                    "valence": float(outputs["valence"].item()),
                    "arousal": float(outputs["arousal"].item()),
                    "iq_proxy": float(outputs["iq_proxy"].item()),
                    "eq_proxy": float(outputs["eq_proxy"].item()),
                }
            )


if __name__ == "__main__":
    main()



