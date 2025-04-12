from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2


class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        
        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        
        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate


    def process_observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        return obs


    def test(self):

        self.model.load_the_model()

        obs, info = self.env.reset()

        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        episode_reward = 0

        while not done:

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                action = torch.argmax(q_values, dim=-1).item()
            
            reward = 0

            for i in range(self.step_repeat):
                reward_temp = 0

                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                reward += reward_temp

                frame = self.env.env.env.render() 

                resized_frame = cv2.resize(frame, (500, 400))

                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Pong AI", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)

                if(done):
                    break
            
            obs = self.process_observation(next_obs)

            episode_reward += reward


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        for episode in range(episodes):

            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()
                
                reward = 0

                for i in range(self.step_repeat):
                    reward_temp = 0

                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                    reward += reward_temp

                    if(done):
                        break
                
                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):

                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()

                    # Current Q-Values from both models
                    q_values = self.model(observations)
                    actions = actions.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, actions)

                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)

                    next_q_values = self.target_model(next_observations).gather(1, next_actions)

                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    loss = F.mse_loss(qsa_batch, target_b.detach())

                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)
                    
            
            self.model.save_the_model()

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")