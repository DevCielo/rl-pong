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
        self.env = env # The OpenAI gym environment

        # Number of times to repeat an action (called frame-skipping)
        # This means agent isn't forced to compute and choose an action at every single frame increasing efficiency, stability and speed.
        self.step_repeat = step_repeat 
        self.gamma = gamma # Discount factor used to value future rewards

        obs, info = self.env.reset() # Resets the environment and returns an observation
        obs = self.process_observation(obs) # Observation is processed to convert into a suitable format.

        # Ensures GPU is available
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Loaded model on device {self.device}')

        ''' 
        Creates a replay buffer with 500,000 transitions
        Replay buffers serve as memory banks that store past experiences (or transitions) from agent's interaction with environment.
        A transition represents single record of agent's experience at one step in environment.
        It contains a:
        1. State (observation) is what the agent perceives in the environment at time t
        2. Action is the decision made by the agent while in that state.
        3. Reward is the reward received by the agent for taking the action in the state.
        4. Next state is the state the agent transitions to after taking the action in the current state.
        5. Done is a boolean that indicates whether the episode has ended.
        '''
        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)

        '''
        In Deep Q-Learning two neural network models are used:
        The model is a neural network that we train directly with data. Takes current state (observation) and outputs Q-values (expected future rewards)
        The Q values represent model's current estimate of long-term benefit of taking each action in a given state.

        The target model is a seperate copy of neural network used to provide a stable set of Q-value targets during training. It's params
        are updated less frequently. This creates a "lag" to help prevent learning process from becoming unstable.
        '''
        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

        # Target network's parameters are initially set to match those of main network to ensure initial training stability.
        self.target_model.load_state_dict(self.model.state_dict())

        # Adam optimizer is used to update model's parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate


    def process_observation(self, obs):
        # Converts the observation from the environment into a PyTorch tensor so it can be passed to the NN
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        return obs


    def test(self):
        # Loads the saved weights
        self.model.load_the_model()

        # Resets the environment and processes observation
        # resets done to false to indicate episode isn't finished
        # initializes the total reward counter for the episode.
        obs, info = self.env.reset()
        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        episode_reward = 0

        # Main testing loop
        while not done:
            # With 5% chance agent selects random action (exploration)
            if random.random() < 0.05:
                action = self.env.action_space.sample()
            
            # Otherwise it uses the network to predict Q-values and selects the action with the highest Q-value
            else:
                # Adds batch dimension and returns the Q-values
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                # Finds action with highest Q-value and converts tensor to a python scalar
                action = torch.argmax(q_values, dim=-1).item()
            
            reward = 0

            # Action is executed step_repeat times to simulate fram-skipping.
            for i in range(self.step_repeat):
                reward_temp = 0
                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                reward += reward_temp
                
                # Visualization with OpenCV
                frame = self.env.env.env.render() 
                resized_frame = cv2.resize(frame, (500, 400))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Pong AI", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.05)
                if(done):
                    break
            
            # Updates the observation to the next observation
            obs = self.process_observation(next_obs)
            episode_reward += reward

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        # Sets up Tensorboard log directory
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)
        if not os.path.exists('models'):
            os.makedirs('models')

        # Starts main training loop
        total_steps = 0
        for episode in range(episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            episode_steps = 0
            episode_start_time = time.time()

            # Loops within each episode and continues until either episode is finished or max allowed steps for episode is reached.
            while not done and episode_steps < max_episode_steps:
                # Uses epsilon-greedy policy to select an action
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()
                
                # Executes the chosen action with step_repeat times to simulate frame skipping
                # During each step the next observation, a temporary reward and the done flag are returned
                reward = 0
                for i in range(self.step_repeat):
                    reward_temp = 0
                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                    reward += reward_temp
                    if(done):
                        break
                
                # Stores the transition in a replay buffer
                next_obs = self.process_observation(next_obs) # Converts to PyTorch tensor
                self.memory.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Performs learning from the Replay Buffer
                if self.memory.can_sample(batch_size):
                    # Checks whether there are enough samples in replay buffer to create batch and samples
                    # a mini-batch of transitions and reshapes done flags to proper dimensions.
                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)
                    dones = dones.unsqueeze(1).float()

                    # Calculates current Q-Values from both models
                    q_values = self.model(observations)
                    actions = actions.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, actions)

                    # Computes the target Q-Values (Double DQN)
                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                    next_q_values = self.target_model(next_observations).gather(1, next_actions)
                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    # Computes MSE between predicted Q-values and target Q-values
                    loss = F.mse_loss(qsa_batch, target_b.detach())
                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    # Backpropagation
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Soft updates the target network every 4 steps to prevent overfitting
                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)
                    
            # Ends the episode and saves the current model.
            self.model.save_the_model()
            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            # Gradually decays the epsilon 
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            episode_time = time.time() - episode_start_time
            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")