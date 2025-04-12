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

        # TODO - PROCESS OBSERVATION

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print('Device: ', self.device)
