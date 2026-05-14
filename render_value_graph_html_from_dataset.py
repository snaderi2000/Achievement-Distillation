import argparse
import os

import torch as th

from collect_value_map import save_value_graph_viewer


def main():
    parser = argparse.ArgumentParser(description="Render a value-graph HTML viewer directly from a saved .pt dataset.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_html_path", type=str, required=True)
    parser.add_argument("--value_graph_max_states", type=int, default=0, help="0 or negative means include all states.")
    parser.add_argument("--value_graph_num_neighbors", type=int, default=4)
    parser.add_argument("--value_graph_value_threshold", type=float, default=None)
    args = parser.parse_args()

    dataset = th.load(args.dataset_path, map_location="cpu")
    max_states = None if args.value_graph_max_states <= 0 else args.value_graph_max_states
    output_dir = os.path.dirname(args.output_html_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_value_graph_viewer(
        dataset,
        output_path=args.output_html_path,
        max_states=max_states,
        num_neighbors=args.value_graph_num_neighbors,
        value_threshold=args.value_graph_value_threshold,
    )
    print(
        f"Rendered {args.output_html_path} from {args.dataset_path} "
        f"with {len(dataset['values'])} dataset states.",
        flush=True,
    )


if __name__ == "__main__":
    main()
