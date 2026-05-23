import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=10000)
parser.add_argument('--model', type=str, default='saved_model.pth')
args = parser.parse_args()

from game import ChessGame
from agent import Agent
from model import state_to_tensor, device

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
    agent = Agent()

    load_model(agent)

    best_reward = float('-inf')

    print(f'Starting training for {EPISODES} episodes...')
    print(f'Model path: {MODEL_PATH}')

    for episode in range(1, EPISODES + 1):
        raw_state = game.reset()
        state = state_to_tensor(raw_state).numpy()

        done = False
        total_reward = 0

        while not done:
            legal_moves = game.generate_moves()

            if not legal_moves:
                break

            legal_actions = list({move_to_action(m) for m in legal_moves})

            action = agent.choose_action(state, legal_actions)

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
            next_state = state_to_tensor(next_raw_state).numpy()

            next_legal_moves = game.generate_moves()
            next_legal_actions = list({move_to_action(m) for m in next_legal_moves})

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                next_legal_actions
            )

            agent.train()

            state = next_state
            total_reward += reward

        agent.end_episode()

        if total_reward > best_reward:
            best_reward = total_reward

            try:
                agent.policy_net.save(MODEL_PATH)

            except:
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)

            print(f'New best model saved! Reward: {total_reward:.2f}')

        if episode % 100 == 0:
            try:
                agent.policy_net.save(MODEL_PATH)

            except:
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)

            print(f'Autosaved model at episode {episode}')

        if episode % 10 == 0:
            print(
                f'Episode {episode} | '
                f'Reward {total_reward:.2f} | '
                f'Best {best_reward:.2f} | '
                f'Epsilon {agent.epsilon:.4f}'
            )

    try:
        agent.policy_net.save(MODEL_PATH)

    except:
        torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    print(f'\nTraining done! Best reward: {best_reward:.2f}')


if __name__ == '__main__':
    train()