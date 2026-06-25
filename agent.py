import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN, device

LR = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.2
EPSILON_DECAY = 0.9999
BATCH_SIZE = 256
MEMORY_SIZE = 100_000
TARGET_UPDATE = 10
ACTION_SIZE = 4096
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 100_000


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-5

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, search_depth=4):
        self.epsilon = EPSILON_START
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE, PER_ALPHA)
        self.episode = 0
        self.frame = 0
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.search_depth = search_depth

    def _beta(self):
        frac = min(1.0, self.frame / PER_BETA_FRAMES)
        return PER_BETA_START + frac * (1.0 - PER_BETA_START)

    def choose_action(self, boards, extras, legal_actions, game, clock):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        from search import search_move
        move = search_move(game, self.policy_net, timer=clock, max_depth=self.search_depth)
        if move is not None:
            action = ((move >> 6) & 63) * 64 + (move & 63)
            if action in legal_actions:
                return action
        boards_t = boards.unsqueeze(0)
        extras_t = extras.unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(boards_t, extras_t)[0]
        mask = torch.full((ACTION_SIZE,), -1e9, device=device)
        mask[legal_actions] = 0.0
        return (q_vals + mask).argmax().item()

    def remember(self, boards, extras, action, reward, next_boards, next_extras, done, next_legal_actions):
        self.memory.add((
            boards.cpu(), extras.cpu(),
            action, reward,
            next_boards.cpu(), next_extras.cpu(),
            done, next_legal_actions
        ))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.frame += 1
        beta = self._beta()
        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta)
        boards_b, extras_b, actions, rewards, next_boards_b, next_extras_b, dones, next_legal_actions = zip(*batch)

        boards_t = torch.stack(boards_b).to(device)
        extras_t = torch.stack(extras_b).to(device)
        next_boards_t = torch.stack(next_boards_b).to(device)
        next_extras_t = torch.stack(next_extras_b).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        current_q = self.policy_net(boards_t, extras_t).gather(1, actions_t)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_boards_t, next_extras_t)
            masks = torch.full((BATCH_SIZE, ACTION_SIZE), -1e9, device=device)
            for i, la in enumerate(next_legal_actions):
                if la:
                    masks[i, la] = 0.0
            best_actions = (next_q_policy + masks).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_boards_t, next_extras_t)
            max_next_q = next_q_target.gather(1, best_actions)
            target_q = rewards_t + GAMMA * max_next_q * (1 - dones_t)

        td_errors = (current_q - target_q).abs().squeeze(1).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss_per = self.loss_fn(current_q, target_q).squeeze(1)
        loss = (loss_per * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def end_episode(self):
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
