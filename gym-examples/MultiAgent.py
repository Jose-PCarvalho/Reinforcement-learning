import gym
from gym.spaces import Discrete, MultiDiscrete, Dict, MultiBinary
from ipywidgets import Output
from IPython import display
import numpy as np
import pygame
import os
from starlette.requests import Request
import time

# Ray imports.
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray import serve
from ray import tune
from Environments.MAGrid import MultiAgentArena as env

ray.init()

tune.register_env('myEnv', env)
TRAINER_CFG = {
    # Using our environment class defined above.
    "env": 'myEnv',
    # Use `framework=torch` here for PyTorch.
    "framework": "tf",

    # Run on 1 GPU on the "learner".
    "num_gpus": 1,
    # Use 15 ray-parallelized environment workers,
    # which collect samples to learn from. Each worker gets assigned
    # 1 CPU.
    "num_workers": 10,
    # Each of the 15 workers has 10 environment copies ("vectorization")
    # for faster (batched) forward passes.
    "num_envs_per_worker": 10,

    "horizon": 60,

    # Multi-agent setup: 2 policies.
    "multiagent": {
        "policies": {"policy1", "policy2"},
        "policy_mapping_fn": lambda agent_id: "policy1" if agent_id == "agent1" else "policy2"
    },

}

results = tune.run(
    # RLlib Trainer class (we use the "PPO" algorithm today).
    PPOTrainer,
    # Give our experiment a name (we will find results/checkpoints
    # under this name on the server's `~ray_results/` dir).
    name=f"CUJ-RL",
    # The RLlib config (defined in a cell above).
    config=TRAINER_CFG,
    # Take a snapshot every 2 iterations.
    checkpoint_freq=2,
    # Plus one at the very end of training.
    checkpoint_at_end=True,
    # Run for exactly 30 training iterations.
    stop={"training_iteration": 20},
    # Define what we are comparing for, when we search for the
    # "best" checkpoint at the end.
    metric="episode_reward_mean",
    mode="max")

print("Best checkpoint: ", results.best_checkpoint)
cpu_config = TRAINER_CFG.copy()
cpu_config["num_gpus"] = 0
cpu_config["num_workers"] = 0

new_trainer = PPOTrainer(config=cpu_config)
new_trainer.restore(results.best_checkpoint)

env = env(config={"render": True})
while True:
    obs = env.reset()
    env.render()

    while True:
        a1 = new_trainer.compute_single_action(obs["agent1"], policy_id="policy1", explore=False)
        a2 = new_trainer.compute_single_action(obs["agent2"], policy_id="policy2", explore=False)

        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})

        env.render()

        if dones["agent1"] is True:
            break
