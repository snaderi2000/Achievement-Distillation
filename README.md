# Achievement Distillation

This is the code for the paper [Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning](https://arxiv.org/abs/2307.03486) accepted to NeurIPS 2023.

<img src="figures/overview.png" width="800">
<img src="figures/result.png" width="600">

## Installation

```
conda create --name ad-crafter python=3.10
conda activate ad-crafter

pip install --upgrade "setuptools==65.7.0" "wheel==0.38.4" "pip==24.0"
pip install -r requirements.txt
pip install -e .
```

## Training

If you are working in a Bash environment, you can use the following commands. The system will set the timestamp to `debug` by default.

PPO
```
python train.py --exp_name ppo --log_stats --save_ckpt
```

PPO + Achievement Distillation (ours)
```
python train.py --exp_name ppo_ad --log_stats --save_ckpt
```

If you are working in an environment that utilizes the Slurm Workload Manager, you can submit your job using `submit.py`. In this setup, the system automatically assigns a timestamp that corresponds to the actual start time of your job.


## Evaluation

If you want to evaluate an agent on a new environment, you can use the following command. This will create a video displaying the agent's behavior.

```
python eval.py --exp_name [your exp name] --timestamp [your timestamp]
```


## Value dataset export

To roll out a trained policy and save per-state value estimates for downstream representation learning, run:

```bash
python collect_value_map.py \
  --exp_name ppo \
  --timestamp <timestamp> \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --num_episodes 10 \
  --output_dataset_path value_dataset.pt \
  --value_map_path value_map.png
```

This exports rollout observations, latent vectors, predicted values, actions, rewards, done flags, episode/step ids, and achievement vectors. If `--value_map_path` is set, the script also saves a 2D PCA visualization of the latent states colored by predicted value and an accompanying `*_embedding.npz` file.

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
For a first-pass interactive explorer, you can also export a self-contained HTML value graph from a single rollout episode:

```bash
python collect_value_map.py \
  --exp_name ppo \
  --timestamp <timestamp> \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --num_episodes 1 \
  --output_dataset_path value_dataset.pt \
  --value_graph_html_path value_graph.html \
  --value_graph_num_neighbors 4 \
  --value_graph_value_threshold 0.25 \
  --episode_video_dir videos/value-graph-demo
```

Open `value_graph.html` in a browser to pan, zoom, hover over nodes, and click a state to pin and enlarge its image. The viewer also shows the value-neighbor settings you used to build the graph, can ring states with non-zero rewards, highlights first-time achievement unlock steps, shows human-readable action names, and lets you highlight a step range like `181-187` from the sidebar. If `--episode_video_dir` is set, the rollout is recorded to an `.mp4` in that directory so you can pair the graph with the full episode video.

=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
## Citation

If you find this code useful, please cite this work.

```
@inproceedings{moon2023ad,
    title={Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning},
    author={Seungyong Moon and Junyoung Yeom and Bumsoo Park and Hyun Oh Song},
    booktitle={Neural Information Processing Systems},
    year={2023}
}
```

## Credit
- https://github.com/openai/Video-Pre-Training
- https://github.com/snu-mllab/DCPG
