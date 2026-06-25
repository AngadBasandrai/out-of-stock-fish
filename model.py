import torch
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, output_size=4096):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.extra_fc = nn.Linear(66, 128)
        self.head = nn.Sequential(
            nn.Linear(128 * 64 + 128, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, output_size),
        )

    def forward(self, boards, extras):
        b = boards.view(-1, 12, 8, 8)
        b = self.conv(b)
        b = b.view(b.size(0), -1)
        e = self.extra_fc(extras)
        x = torch.cat([b, e], dim=1)
        return self.head(x)

    def save(self, path='model.pth'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def state_to_tensor(state):
    boards = torch.zeros(12, 64, dtype=torch.float32, device=device)
    for i in range(12):
        bb = int(state[i])
        for sq in range(64):
            boards[i, sq] = (bb >> sq) & 1
    boards = boards.view(12, 8, 8)

    extras = torch.zeros(66, dtype=torch.float32, device=device)
    extras[0] = float(state[14])
    extras[1] = float(state[15]) / 15.0
    ep = int(state[16])
    if 0 <= ep < 64:
        extras[2 + ep] = 1.0

    return boards, extras
