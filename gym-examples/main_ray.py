import gym
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback

import gym_examples
from gym.wrappers import FlattenObservation
from gym.wrappers import TimeLimit
import gym_examples
from gym_examples.envs import grid_world_coverage
from ray import tune
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print


def env_creator(env_name):
    if env_name == 'CustomEnv-v0':
        from gym_examples.envs.grid_world_coverage import GridWorldCoverageEnv as env
    else:
        raise NotImplementedError
    return env


ray.init()
env = env_creator('CustomEnv-v0')
tune.register_env('myEnv', env)

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["env"]='myEnv'
algo = ppo.PPO(config=config)

# Can optionally call algo.restore(path) to load a checkpoint.

for i in range(200):
    # Perform one iteration of training the policy with PPO
    result = algo.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = algo.save()
        print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Algorithm's Policy's ModelV2
# (tf or torch) by doing:
algo.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch policies/models.
