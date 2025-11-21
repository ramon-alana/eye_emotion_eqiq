from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_absolute_error
from tqdm import tqdm

from data import EyeIQDataset, collate_with_padding
from models import EyeIQNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train eye-based emotion/IQ model")
    parser.add_argument("--metadata", type=str, required=True, help="CSV with labels")
    parser.add_argument("--image-root", type=str, required=True, help="Base image dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-emotions", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = EyeIQDataset(
        metadata_csv=args.metadata,
        image_root=args.image_root,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_padding,
    )


def compute_losses(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], criterion_ce: nn.Module
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    if "emotion_label" in batch:
        losses.append(criterion_ce(outputs["emotion_logits"], batch["emotion_label"]))
    regression_targets = ["valence", "arousal", "iq_proxy", "eq_proxy"]
    for key in regression_targets:
        if key in batch:
            losses.append(nn.functional.mse_loss(outputs[key], batch[key]))
    return sum(losses) / len(losses)


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader = build_dataloader(args)
    model = EyeIQNet(num_emotions=args.num_emotions).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion_ce = nn.CrossEntropyLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        all_preds: List[int] = []
        all_labels: List[int] = []
        reg_metrics: Dict[str, List[float]] = {key: [] for key in ["valence", "arousal", "iq_proxy", "eq_proxy"]}

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["image"])
            loss = compute_losses(outputs, batch, criterion_ce)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if "emotion_label" in batch:
                preds = outputs["emotion_logits"].argmax(dim=1).detach().cpu()
                labels = batch["emotion_label"].cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

            for key in reg_metrics.keys():
                if key in batch:
                    metric = mean_absolute_error(outputs[key], batch[key]).item()
                    reg_metrics[key].append(metric)

        stats = {
            "loss": sum(epoch_losses) / len(epoch_losses),
        }
        if all_labels:
            stats["acc"] = accuracy_score(all_labels, all_preds)
            stats["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
        for key, values in reg_metrics.items():
            if values:
                stats[f"{key}_mae"] = sum(values) / len(values)

        print(f"Epoch {epoch+1}: {stats}")

        ckpt = checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict()}, ckpt)
        print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    train()



