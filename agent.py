import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN, device

LR = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
BATCH_SIZE = 256
MEMORY_SIZE = 100_000
TARGET_UPDATE = 10
ACTION_SIZE = 4096

class Agent:
    def __init__(self):
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.episode = 0
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)[0]
        mask = torch.full((ACTION_SIZE,), -1e9, device=device)
        mask[legal_actions] = 0.0
        return (q_vals + mask).argmax().item()

    def remember(self, state, action, reward, next_state, done, next_legal_actions):
        self.memory.append((state, action, reward, next_state, done, next_legal_actions))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones, next_legal_actions = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            masks = torch.full((BATCH_SIZE, ACTION_SIZE), -1e9, device=device)
            for i, la in enumerate(next_legal_actions):
                if la: masks[i, la] = 0.0
            max_next_q = (next_q + masks).max(dim=1, keepdim=True).values
            target_q = rewards + GAMMA * max_next_q * (1 - dones)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self):
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
