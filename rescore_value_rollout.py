import argparse
import copy
import os
from typing import Dict, Optional

import torch as th

from collect_value_map import load_model, save_value_graph_viewer, set_seed


def swap_component_weights(
    receiver_model: th.nn.Module,
    donor_model: th.nn.Module,
    component_name: str,
):
    receiver_state = receiver_model.state_dict()
    donor_state = donor_model.state_dict()
    prefix = f"{component_name}."
    matched = 0
    for key, value in donor_state.items():
        if key.startswith(prefix):
            if key not in receiver_state:
                raise KeyError(f"Component key '{key}' not found in receiver model.")
            if receiver_state[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': receiver {tuple(receiver_state[key].shape)} vs donor {tuple(value.shape)}"
                )
            receiver_state[key] = value.clone()
            matched += 1
    if matched == 0:
        raise ValueError(f"No parameters found for component '{component_name}'.")
    receiver_model.load_state_dict(receiver_state)


def rescore_dataset(
    model: th.nn.Module,
    dataset: Dict[str, th.Tensor],
    device: th.device,
    batch_size: int,
) -> Dict[str, th.Tensor]:
    observations = dataset["observations"]
    states = dataset.get("states")
    rnn_states = dataset.get("rnn_states")
    rescored_values = []
    rescored_latents = []

    model.eval()
    with th.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start : start + batch_size].to(device)
            kwargs = {}
            if states is not None:
                kwargs["states"] = states[start : start + batch_size].to(device)
            if rnn_states is not None:
                kwargs["rnn_states"] = rnn_states[start : start + batch_size].to(device)
            outputs = model.forward(batch, **kwargs)
            rescored_values.append(model.vf_head.denormalize(outputs["vpreds"]).view(-1).cpu())
            rescored_latents.append(outputs["latents"].cpu())

    rescored_dataset = copy.deepcopy(dataset)
    rescored_dataset["values"] = th.cat(rescored_values, dim=0)
    rescored_dataset["latents"] = th.cat(rescored_latents, dim=0)
    return rescored_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Re-score a saved rollout dataset with another checkpoint or a component-swapped checkpoint, then export a comparable HTML value graph."
    )
    parser.add_argument("--rollout_dataset_path", type=str, required=True)
    parser.add_argument("--base_exp_name", type=str, required=True)
    parser.add_argument("--base_timestamp", type=str, required=True)
    parser.add_argument("--base_train_seed", type=int, required=True)
    parser.add_argument("--base_ckpt_epoch", type=int, required=True)
    parser.add_argument("--output_html_path", type=str, required=True)
    parser.add_argument("--output_dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--value_graph_max_states", type=int, default=400)
    parser.add_argument("--value_graph_num_neighbors", type=int, default=4)
    parser.add_argument("--value_graph_value_threshold", type=float, default=None)
    parser.add_argument("--eval_seed", type=int, default=123)

    parser.add_argument("--donor_exp_name", type=str, default=None)
    parser.add_argument("--donor_timestamp", type=str, default=None)
    parser.add_argument("--donor_train_seed", type=int, default=None)
    parser.add_argument("--donor_ckpt_epoch", type=int, default=None)
    parser.add_argument(
        "--swap_components",
        type=str,
        default="",
        help="Comma-separated components to swap from donor into base model. Supported names are enc, linear, pi_head, vf_head.",
    )
    args = parser.parse_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)
    set_seed(args.eval_seed)

    if not os.path.exists(args.rollout_dataset_path):
        raise FileNotFoundError(f"Rollout dataset not found at {args.rollout_dataset_path}")
    dataset = th.load(args.rollout_dataset_path, map_location="cpu")

    base_model, _, base_ckpt = load_model(
        args.base_exp_name,
        args.base_timestamp,
        args.base_train_seed,
        args.base_ckpt_epoch,
        device,
    )
    print(f"Loaded base scorer: {base_ckpt}")

    swap_components = [item.strip() for item in args.swap_components.split(",") if item.strip()]
    donor_model: Optional[th.nn.Module] = None
    if swap_components:
        required = [args.donor_exp_name, args.donor_timestamp, args.donor_train_seed, args.donor_ckpt_epoch]
        if any(value is None for value in required):
            raise ValueError("Donor checkpoint args are required when --swap_components is used.")
        donor_model, _, donor_ckpt = load_model(
            args.donor_exp_name,
            args.donor_timestamp,
            int(args.donor_train_seed),
            int(args.donor_ckpt_epoch),
            device,
        )
        print(f"Loaded donor scorer: {donor_ckpt}")
        for component_name in swap_components:
            swap_component_weights(base_model, donor_model, component_name)
            print(f"Swapped component '{component_name}' from donor into base scorer.")

    rescored_dataset = rescore_dataset(
        model=base_model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
    )
    print(
        f"Rescored {len(rescored_dataset['values'])} states: "
        f"mean={rescored_dataset['values'].mean().item():.4f}, "
        f"std={rescored_dataset['values'].std().item():.4f}, "
        f"min={rescored_dataset['values'].min().item():.4f}, "
        f"max={rescored_dataset['values'].max().item():.4f}"
    )

    if args.output_dataset_path:
        th.save(rescored_dataset, args.output_dataset_path)
        print(f"Saved rescored dataset to {args.output_dataset_path}")

    save_value_graph_viewer(
        dataset=rescored_dataset,
        output_path=args.output_html_path,
        max_states=args.value_graph_max_states,
        num_neighbors=args.value_graph_num_neighbors,
        value_threshold=args.value_graph_value_threshold,
    )
    print(f"Saved rescored value graph to {args.output_html_path}")


if __name__ == "__main__":
    main()
