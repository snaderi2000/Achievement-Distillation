import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class TransitionDataset(Dataset):
    def __init__(self, data: Dict[str, th.Tensor]):
        self.obs = data["observations"]
        self.next_obs = data["next_observations"]
        self.actions = data["actions"].long()

    def __len__(self):
        return int(self.actions.shape[0])

    def __getitem__(self, idx):
        obs = self.obs[idx].float().permute(2, 0, 1) / 255.0
        next_obs = self.next_obs[idx].float().permute(2, 0, 1) / 255.0
        delta = next_obs - obs
        x = th.cat([obs, next_obs, delta], dim=0)
        return x, self.actions[idx]


class InverseDynamicsCNN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with th.no_grad():
            dummy = th.zeros(1, in_channels, 64, 64)
            out_dim = self.net(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.head(self.net(x))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with th.no_grad():
        for x, actions in loader:
            x = x.to(device)
            actions = actions.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, actions, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=-1) == actions).sum().item())
            total += int(actions.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def main(args):
    set_seed(args.seed)
    device = th.device("cuda:0" if th.cuda.is_available() and not args.cpu else "cpu")

    data = th.load(args.dataset_path, map_location="cpu")
    dataset = TransitionDataset(data)
    num_actions = len(data.get("metadata", {}).get("action_names", [])) or int(data["actions"].max().item() + 1)
    train_size = int(len(dataset) * args.train_frac)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=th.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample_x, _ = dataset[0]
    model = InverseDynamicsCNN(sample_x.shape[0], num_actions).to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "inverse_model_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, actions in train_loader:
            x = x.to(device)
            actions = actions.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * int(actions.numel())
            total_correct += int((logits.argmax(dim=-1) == actions).sum().item())
            total += int(actions.numel())

        train_stats = {
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
        }
        val_stats = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row))

        if val_stats["accuracy"] > best_acc:
            best_acc = val_stats["accuracy"]
            th.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": sample_x.shape[0],
                    "num_actions": num_actions,
                    "metadata": data.get("metadata", {}),
                    "history": history,
                },
                best_path,
            )

    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved best model to {best_path}")
    print(f"random baseline accuracy: {1.0 / num_actions:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="affordance_transitions.pt")
    parser.add_argument("--output_dir", type=str, default="affordance_probe")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
