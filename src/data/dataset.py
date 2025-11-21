from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


@dataclass
class EyeRecord:
    image_path: Path
    emotion_label: Optional[int]
    valence: Optional[float]
    arousal: Optional[float]
    iq_proxy: Optional[float]
    eq_proxy: Optional[float]


def _build_default_transform(image_size: int) -> A.BasicTransform:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


class EyeIQDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        image_root: str | Path,
        image_size: int = 224,
        transform: Optional[A.BasicTransform] = None,
    ) -> None:
        self.metadata = self._load_metadata(Path(metadata_csv))
        self.image_root = Path(image_root)
        self.transform = transform or _build_default_transform(image_size)

    @staticmethod
    def _load_metadata(csv_path: Path) -> List[EyeRecord]:
        df = pd.read_csv(csv_path)
        required_cols = {"image_path"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"metadata missing columns: {missing}")

        records: List[EyeRecord] = []
        for row in df.itertuples(index=False):
            records.append(
                EyeRecord(
                    image_path=Path(getattr(row, "image_path")),
                    emotion_label=getattr(row, "emotion_label", None),
                    valence=getattr(row, "valence", None),
                    arousal=getattr(row, "arousal", None),
                    iq_proxy=getattr(row, "iq_proxy", None),
                    eq_proxy=getattr(row, "eq_proxy", None),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.metadata[idx]
        image_fp = (self.image_root / record.image_path).resolve()
        if not image_fp.exists():
            raise FileNotFoundError(f"missing image: {image_fp}")

        image = cv2.imread(str(image_fp))
        if image is None:
            raise ValueError(f"cv2 failed to read {image_fp}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        tensor_img = augmented["image"]

        sample: Dict[str, Any] = {"image": tensor_img, "image_path": str(record.image_path)}
        if record.emotion_label is not None and not np.isnan(record.emotion_label):
            sample["emotion_label"] = torch.tensor(
                int(record.emotion_label), dtype=torch.long
            )
        if record.valence is not None and not np.isnan(record.valence):
            sample["valence"] = torch.tensor(record.valence, dtype=torch.float32)
        if record.arousal is not None and not np.isnan(record.arousal):
            sample["arousal"] = torch.tensor(record.arousal, dtype=torch.float32)
        if record.iq_proxy is not None and not np.isnan(record.iq_proxy):
            sample["iq_proxy"] = torch.tensor(record.iq_proxy, dtype=torch.float32)
        if record.eq_proxy is not None and not np.isnan(record.eq_proxy):
            sample["eq_proxy"] = torch.tensor(record.eq_proxy, dtype=torch.float32)
        return sample


def collate_with_padding(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    collated: Dict[str, Any] = {"image": images}

    def maybe_stack(key: str, dtype: torch.dtype) -> None:
        values = [item[key] for item in batch if key in item]
        if values:
            collated[key] = torch.stack(
                [v.to(dtype=dtype) if v.dtype != dtype else v for v in values]
            )

    maybe_stack("emotion_label", torch.long)
    for regression_key in ("valence", "arousal", "iq_proxy", "eq_proxy"):
        values = [item[regression_key] for item in batch if regression_key in item]
        if values:
            collated[regression_key] = torch.stack(values)

    return collated

