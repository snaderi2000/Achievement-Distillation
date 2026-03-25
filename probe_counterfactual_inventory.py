import argparse
import os
from typing import Dict, List

import torch as th

from collect_value_map import load_model, set_seed


def parse_donor_steps(text: str) -> List[int]:
    steps = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        steps.append(int(item))
    if not steps:
        raise ValueError("Expected at least one donor step.")
    return steps


def find_dataset_index(dataset: Dict[str, th.Tensor], episode_id: int, step_id: int) -> int:
    episode_ids = dataset["episode_ids"]
    step_ids = dataset["step_ids"]
    matches = ((episode_ids == episode_id) & (step_ids == step_id)).nonzero(as_tuple=False).view(-1)
    if len(matches) == 0:
        raise ValueError(f"No state found for episode={episode_id}, step={step_id}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple states found for episode={episode_id}, step={step_id}.")
    return int(matches.item())


def evaluate_value(model, observation: th.Tensor, device: th.device) -> float:
    with th.no_grad():
        outputs = model.act(observation.unsqueeze(0).to(device))
    return float(outputs["vpreds"].item())


def swap_inventory_rows(base_obs: th.Tensor, donor_obs: th.Tensor, inventory_rows: int) -> th.Tensor:
    if inventory_rows <= 0:
        raise ValueError("--inventory_rows must be positive.")
    if base_obs.shape != donor_obs.shape:
        raise ValueError(f"Observation shape mismatch: {tuple(base_obs.shape)} vs {tuple(donor_obs.shape)}")
    if inventory_rows > base_obs.shape[1]:
        raise ValueError(
            f"inventory_rows={inventory_rows} exceeds observation height {base_obs.shape[1]}"
        )

    hybrid = base_obs.clone()
    hybrid[:, -inventory_rows:, :] = donor_obs[:, -inventory_rows:, :]
    return hybrid


def save_probe_figure(
    base_obs: th.Tensor,
    donor_obs: th.Tensor,
    hybrid_obs: th.Tensor,
    output_path: str,
    base_step: int,
    donor_step: int,
    base_value: float,
    donor_value: float,
    hybrid_value: float,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    def to_hwc(obs: th.Tensor):
        return obs.detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    panels = [
        (base_obs, f"Base step {base_step}\nvalue={base_value:.3f}"),
        (donor_obs, f"Donor step {donor_step}\nvalue={donor_value:.3f}"),
        (hybrid_obs, f"Hybrid\nvalue={hybrid_value:.3f}"),
    ]
    for ax, (obs, title) in zip(axes, panels):
        ax.imshow(to_hwc(obs))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Counterfactual inventory swap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Swap the inventory HUD from one rollout state into another and evaluate the trained value function."
    )
    parser.add_argument("--dataset_path", type=str, default="value_dataset.pt")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--base_step", type=int, required=True)
    parser.add_argument(
        "--donor_steps",
        type=str,
        required=True,
        help="Comma-separated donor steps, e.g. 100,200,300,389",
    )
    parser.add_argument(
        "--inventory_rows",
        type=int,
        default=16,
        help="How many rows from the bottom of the image to swap.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default=None,
        help="Optional directory for side-by-side base/donor/hybrid figures.",
    )
    args = parser.parse_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)
    set_seed(0)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    dataset = th.load(args.dataset_path, map_location="cpu")
    donor_steps = parse_donor_steps(args.donor_steps)

    model, _, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Using device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")

    base_idx = find_dataset_index(dataset, args.episode_id, args.base_step)
    base_obs = dataset["observations"][base_idx]
    base_value = evaluate_value(model, base_obs, device)
    print(f"Base state: episode={args.episode_id}, step={args.base_step}, value={base_value:.4f}")

    if args.figure_dir:
        os.makedirs(args.figure_dir, exist_ok=True)

    print("")
    print("donor_step\t donor_value\t hybrid_value\t delta")
    for donor_step in donor_steps:
        donor_idx = find_dataset_index(dataset, args.episode_id, donor_step)
        donor_obs = dataset["observations"][donor_idx]
        donor_value = evaluate_value(model, donor_obs, device)

        hybrid_obs = swap_inventory_rows(base_obs, donor_obs, inventory_rows=args.inventory_rows)
        hybrid_value = evaluate_value(model, hybrid_obs, device)
        delta = hybrid_value - base_value

        print(f"{donor_step:9d}\t {donor_value:11.4f}\t {hybrid_value:11.4f}\t {delta:+.4f}")

        if args.figure_dir:
            output_path = os.path.join(
                args.figure_dir,
                f"base-{args.base_step:04d}_donor-{donor_step:04d}.png",
            )
            save_probe_figure(
                base_obs=base_obs,
                donor_obs=donor_obs,
                hybrid_obs=hybrid_obs,
                output_path=output_path,
                base_step=args.base_step,
                donor_step=donor_step,
                base_value=base_value,
                donor_value=donor_value,
                hybrid_value=hybrid_value,
            )
            print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
