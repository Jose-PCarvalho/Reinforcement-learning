import gym
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback

import gym_examples
from gym.wrappers import FlattenObservation
import stable_baselines3
import gym_examples
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

eval_env = gym.make('gym_examples/GridWorldTarget-v0',render_mode="human",size=5)
eval_env = FlattenObservation(eval_env)
eval_env=Monitor(eval_env)
model = DQN.load("DQN_BEST", env=eval_env)
mean_reward, std_reward =evaluate_policy(model, eval_env, n_eval_episodes=10)
print(mean_reward,std_reward)