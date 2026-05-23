import torch
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, input_size=834, hidden_size=1024, output_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path='model.pth'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        self.load_state_dict(torch.load(path, map_location='cpu'))

def state_to_tensor(state):
    tensor = []
    for bb in state[:12]:
        bb = int(bb)
        for sq in range(64):
            tensor.append((bb >> sq) & 1)
    tensor.append(state[14])
    tensor.append(state[15] / 15.0)
    ep = int(state[16])
    for sq in range(64):
        tensor.append(1 if sq == ep else 0)
    return torch.tensor(tensor, dtype=torch.float32, device=device)
