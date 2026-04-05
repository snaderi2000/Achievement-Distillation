import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def _load_pt_or_npz(path: str) -> Dict[str, np.ndarray]:
    if path.endswith(".pt"):
        shard = th.load(path, map_location="cpu")
        return {k: v.cpu().numpy() if isinstance(v, th.Tensor) else np.asarray(v) for k, v in shard.items()}
    if path.endswith(".npz"):
        with np.load(path) as shard:
            return {k: shard[k] for k in shard.files}
    raise ValueError(f"Unsupported shard format: {path}")


class CrafterShardDataset(Dataset):
    def __init__(self, dataset_dir: str):
        manifest_path = os.path.join(dataset_dir, "metadata.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        self.dataset_dir = dataset_dir
        self.shards: List[Dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            header = json.loads(next(f))
            self.run_info = header.get("run_info", {})
            for line in f:
                record = json.loads(line)
                if "shard_path" not in record:
                    continue
                self.shards.append(record)

        if not self.shards:
            raise ValueError("No shard entries found in manifest. Collect rollout frames with --save_format pt or npz first.")

        self.starts: List[int] = []
        total = 0
        for shard in self.shards:
            self.starts.append(total)
            total += int(shard["num_frames"])
        self.total_frames = total

        self._cached_shard_index = -1
        self._cached_shard = None

    def __len__(self) -> int:
        return self.total_frames

    def _find_shard(self, index: int) -> Tuple[int, int]:
        shard_idx = max(i for i, start in enumerate(self.starts) if start <= index)
        local_idx = index - self.starts[shard_idx]
        return shard_idx, local_idx

    def _get_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._cached_shard_index != shard_idx:
            shard_path = os.path.join(self.dataset_dir, self.shards[shard_idx]["shard_path"])
            self._cached_shard = _load_pt_or_npz(shard_path)
            self._cached_shard_index = shard_idx
        return self._cached_shard

    def __getitem__(self, index: int) -> np.ndarray:
        shard_idx, local_idx = self._find_shard(index)
        shard = self._get_shard(shard_idx)
        return shard["observations"][local_idx]


class CrafterTwoViewTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.mean = th.tensor([0.485, 0.456, 0.406], dtype=th.float32).view(3, 1, 1)
        self.std = th.tensor([0.229, 0.224, 0.225], dtype=th.float32).view(3, 1, 1)

    def __call__(self, frame: np.ndarray) -> Tuple[th.Tensor, th.Tensor]:
        image = Image.fromarray(frame)
        return self._augment(image), self._augment(image)

    def _augment(self, image: Image.Image) -> th.Tensor:
        image = self._random_resized_crop(image)
        image = self._color_jitter(image)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = th.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _random_resized_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(0.7, 1.0)
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        max_x = max(width - crop_w, 0)
        max_y = max(height - crop_h, 0)
        left = random.randint(0, max_x) if max_x > 0 else 0
        top = random.randint(0, max_y) if max_y > 0 else 0
        image = image.crop((left, top, left + crop_w, top + crop_h))
        return image.resize((self.image_size, self.image_size), Image.BICUBIC)

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        return image


class TwoViewDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform: CrafterTwoViewTransform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        frame = self.base_dataset[index]
        return self.transform(frame)


class VICRegProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


def off_diagonal(x: th.Tensor) -> th.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: th.Tensor,
    z2: th.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[th.Tensor, Dict[str, float]]:
    repr_loss = F.mse_loss(z1, z2)

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std_z1 = th.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std_z2 = th.sqrt(z2.var(dim=0, unbiased=False) + eps)
    std_loss = 0.5 * (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

    cov_z1 = (z1.T @ z1) / max(z1.shape[0] - 1, 1)
    cov_z2 = (z2.T @ z2) / max(z2.shape[0] - 1, 1)
    cov_loss = off_diagonal(cov_z1).pow(2).mean() + off_diagonal(cov_z2).pow(2).mean()

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    stats = {
        "repr_loss": float(repr_loss.item()),
        "std_loss": float(std_loss.item()),
        "cov_loss": float(cov_loss.item()),
        "loss": float(loss.item()),
    }
    return loss, stats


def get_backbone_blocks(model: nn.Module):
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layer"):
        return model.vision_model.encoder.layer
    raise AttributeError("Could not find transformer blocks on this DINO model.")


def configure_trainable_backbone(model: nn.Module, unfreeze_last_n_blocks: int):
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_last_n_blocks <= 0:
        return

    blocks = get_backbone_blocks(model)
    for block in blocks[-unfreeze_last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    if hasattr(model, "layernorm") and isinstance(model.layernorm, nn.Module):
        for param in model.layernorm.parameters():
            param.requires_grad = True
    if hasattr(model, "post_layernorm") and isinstance(model.post_layernorm, nn.Module):
        for param in model.post_layernorm.parameters():
            param.requires_grad = True


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    model_name_or_path: str
    image_size: int
    batch_size: int
    epochs: int
    num_workers: int
    lr_backbone: float
    lr_head: float
    weight_decay: float
    proj_dim: int
    unfreeze_last_n_blocks: int
    log_every_steps: int
    seed: int


def main():
    parser = argparse.ArgumentParser(description="Lightweight Crafter-domain adaptation for pretrained DINOv3 using VICReg.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=1)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    os.makedirs(config.output_dir, exist_ok=True)

    set_seed(config.seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)

    base_dataset = CrafterShardDataset(config.dataset_dir)
    dataset = TwoViewDataset(base_dataset, CrafterTwoViewTransform(config.image_size))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=th.cuda.is_available(),
        drop_last=True,
    )

    model = AutoModel.from_pretrained(config.model_name_or_path)
    hidden_size = int(model.config.hidden_size)
    configure_trainable_backbone(model, config.unfreeze_last_n_blocks)
    projector = VICRegProjector(hidden_size, config.proj_dim)

    model = model.to(device)
    projector = projector.to(device)

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": projector.parameters(), "lr": config.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.lr_backbone})

    optimizer = th.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = len(loader) * config.epochs
    print(f"Using device: {device}")
    print(f"Frames in dataset: {len(base_dataset)}")
    print(f"Steps per epoch: {len(loader)}")
    print(f"Total train steps: {total_steps}")
    print(f"Trainable backbone params: {sum(p.numel() for p in backbone_params):,}")

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        running = {"loss": 0.0, "repr_loss": 0.0, "std_loss": 0.0, "cov_loss": 0.0}
        for step, (view1, view2) in enumerate(loader, start=1):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            out1 = model(pixel_values=view1).last_hidden_state[:, 0]
            out2 = model(pixel_values=view2).last_hidden_state[:, 0]
            z1 = projector(out1)
            z2 = projector(out2)

            loss, stats = vicreg_loss(z1, z2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            for key in running:
                running[key] += stats[key]

            if config.log_every_steps > 0 and step % config.log_every_steps == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"[epoch {epoch:02d} step {step:04d}/{len(loader)}] "
                    f"loss={stats['loss']:.4f} repr={stats['repr_loss']:.4f} "
                    f"std={stats['std_loss']:.4f} cov={stats['cov_loss']:.4f} "
                    f"elapsed={elapsed/60:.1f}m"
                )

        epoch_time = time.time() - epoch_start
        epoch_stats = {key: value / max(len(loader), 1) for key, value in running.items()}
        print(
            f"epoch {epoch} done: loss={epoch_stats['loss']:.4f} repr={epoch_stats['repr_loss']:.4f} "
            f"std={epoch_stats['std_loss']:.4f} cov={epoch_stats['cov_loss']:.4f} time={epoch_time/60:.1f}m"
        )

        ckpt_dir = os.path.join(config.output_dir, f"epoch-{epoch:03d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(os.path.join(ckpt_dir, "backbone"))
        th.save(projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
        with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        with open(os.path.join(ckpt_dir, "epoch_stats.json"), "w", encoding="utf-8") as f:
            json.dump(epoch_stats, f, indent=2)

    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(os.path.join(final_dir, "backbone"))
    th.save(projector.state_dict(), os.path.join(final_dir, "projector.pt"))
    with open(os.path.join(final_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


if __name__ == "__main__":
    main()
