import gym
import gym_examples
from gym.wrappers import FlattenObservation
import stable_baselines3
import gym_examples
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
env = gym.make('gym_examples/GridWorld-v0')
env.action_space.sample()

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

#env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1,cuda='cpu')
check_env(env)

model.learn(total_timesteps=10000)


evaluate_policy(model, env, n_eval_episodes=10, render=True)

