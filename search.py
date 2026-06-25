import torch
from model import state_to_tensor, device
from game import WHITE, BLACK

INF = 1_000_000


def _dqn_eval(game, policy_net):
    with torch.no_grad():
        boards, extras = state_to_tensor(game.get_state())
        q_vals = policy_net(boards.unsqueeze(0), extras.unsqueeze(0))[0]
    legal = game.legal_moves()
    if not legal:
        in_check = game._king_in_check(game.side)
        return -INF if in_check else 0
    best = -float('inf')
    for m in legal:
        action = ((m >> 6) & 63) * 64 + (m & 63)
        val = q_vals[action].item() if action < len(q_vals) else -1e9
        if val > best:
            best = val
    return best


def _negamax(game, depth, alpha, beta, color, policy_net, timer):
    if timer and timer.move_time_up():
        return None
    legal = game.legal_moves()
    if not legal:
        in_check = game._king_in_check(game.side)
        return color * (-INF if in_check else 0)
    if depth == 0:
        return color * _dqn_eval(game, policy_net)
    value = -float('inf')
    for move in legal:
        prev_castle = game.castle
        prev_halfmove = game.halfmove
        undo = game._apply_move(move)
        if undo is None:
            continue
        game.side ^= 1
        score = _negamax(game, depth - 1, -beta, -alpha, -color, policy_net, timer)
        game.side ^= 1
        game._undo_move(move, *undo, prev_castle, prev_halfmove)
        if score is None:
            return None
        value = max(value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return value


def search_move(game, policy_net, timer=None, max_depth=4):
    legal = game.legal_moves()
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    best_move = legal[0]
    color = 1 if game.side == WHITE else -1
    for depth in range(1, max_depth + 1):
        if timer and timer.move_time_up():
            break
        current_best = best_move
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in legal:
            prev_castle = game.castle
            prev_halfmove = game.halfmove
            undo = game._apply_move(move)
            if undo is None:
                continue
            game.side ^= 1
            value = _negamax(game, depth - 1, -beta, -alpha, -color, policy_net, timer)
            game.side ^= 1
            game._undo_move(move, *undo, prev_castle, prev_halfmove)
            if value is None:
                return best_move
            if value > best_value:
                best_value = value
                current_best = move
            alpha = max(alpha, value)
        best_move = current_best
    return best_move
