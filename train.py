from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py

hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99

env = gym.make('ALE/Pong-v5', render_mode='rgb_array')

env = ResizeObservation(env, shape=(64, 64))

env = GrayscaleObservation(env, keep_dim=True)

agent = Agent(env, 
              hidden_layer=hidden_layer,
                learning_rate=learning_rate,
                step_repeat=step_repeat,
                gamma=gamma)

