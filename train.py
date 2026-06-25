import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100000)
parser.add_argument('--model', type=str, default='saved_model.pth')
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--time', type=float, default=300.0)
parser.add_argument('--increment', type=float, default=0.0)
args = parser.parse_args()

from game import ChessGame
from agent import Agent
from model import state_to_tensor, device
from timer import GameClock

print(f'Using device: {device}')

EPISODES = args.episodes
MODEL_PATH = args.model


def move_to_action(move):
    return ((move >> 6) & 63) * 64 + (move & 63)


def load_model(agent):
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict):
                agent.policy_net.load_state_dict(checkpoint)
                agent.target_net.load_state_dict(checkpoint)
            else:
                agent.policy_net = checkpoint.to(device)
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.policy_net.eval()
            agent.target_net.eval()
            print(f'Loaded existing model: {MODEL_PATH}')
        except Exception as e:
            print(f'Failed to load model: {e}')
            print('Starting fresh.')
    else:
        print('No saved model found. Starting fresh.')


def train():
    game = ChessGame()
    agent = Agent(search_depth=args.depth)

    load_model(agent)

    best_reward = float('-inf')

    print(f'Starting training for {EPISODES} episodes...')
    print(f'Model path: {MODEL_PATH}')
    print(f'Search depth: {args.depth} | Time: {args.time}s | Increment: {args.increment}s')

    for episode in range(1, EPISODES + 1):
        raw_state = game.reset()
        boards, extras = state_to_tensor(raw_state)

        clock = GameClock(args.time, args.increment)

        done = False
        total_reward = 0

        while not done:
            legal_moves = game.legal_moves()

            if not legal_moves:
                break

            legal_actions = list({move_to_action(m) for m in legal_moves})

            clock.start_move()
            action = agent.choose_action(boards, extras, legal_actions, game, clock)
            clock.end_move()

            target_from = action // 64
            target_to = action % 64

            move = None
            for m in legal_moves:
                if ((m >> 6) & 63) == target_from and (m & 63) == target_to:
                    move = m
                    break

            if move is None:
                break

            next_raw_state, reward, done = game.make_move(move)
            next_boards, next_extras = state_to_tensor(next_raw_state)

            next_legal_moves = game.legal_moves()
            next_legal_actions = list({move_to_action(m) for m in next_legal_moves})

            agent.remember(
                boards, extras,
                action, reward,
                next_boards, next_extras,
                done, next_legal_actions
            )

            agent.train()

            boards, extras = next_boards, next_extras
            total_reward += reward

        agent.end_episode()

        if total_reward > best_reward:
            best_reward = total_reward
            try:
                agent.policy_net.save(MODEL_PATH)
            except Exception:
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)
            print(f'New best model saved! Reward: {total_reward:.2f}')

        if episode % 100 == 0:
            try:
                agent.policy_net.save(MODEL_PATH)
            except Exception:
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)
            print(f'Autosaved model at episode {episode}')

        if episode % 10 == 0:
            print(f'Episode {episode} | Reward {total_reward:.2f} | Best {best_reward:.2f} | Epsilon {agent.epsilon:.4f}')

    try:
        agent.policy_net.save(MODEL_PATH)
    except Exception:
        torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    print(f'Training done! Best reward: {best_reward:.2f}')


if __name__ == '__main__':
    train()
