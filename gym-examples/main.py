import gym
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback

import gym_examples
from gym.wrappers import FlattenObservation
from gym.wrappers import TimeLimit
import stable_baselines3
import gym_examples
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
env = gym.make('gym_examples/GridWorldCoverage-v0',size=5)
env.action_space.sample()

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:

        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))



#env = FlattenObservation(env)
env=TimeLimit(env,max_episode_steps=5*5*5*5*1000)
check_env(env)
env = DummyVecEnv([lambda: env])
eval_env = gym.make('gym_examples/GridWorldCoverage-v0',size=5)
eval_env=TimeLimit(eval_env,max_episode_steps=5*5*5*5*1000)
#eval_env = FlattenObservation(eval_env)
eval_env=Monitor(eval_env)
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=15, min_evals=5, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=10000000000, callback_after_eval=stop_train_callback, verbose=1)

model = PPO('MultiInputPolicy', env, verbose = 1, device='cpu',learning_rate=0.0002,gamma=0.995)

model.learn(total_timesteps=10000000, callback=eval_callback)
model.save('PPO-Coverage')
env = gym.make('gym_examples/GridWorldCoverage-v0',render_mode="human",size=5)
env=TimeLimit(env,max_episode_steps=5*5*5*5*100)
#env = FlattenObservation(env)
#env=Monitor(env)
#mean_reward, std_reward =evaluate_policy(model, env, n_eval_episodes=10000)
#print(mean_reward,std_reward)

