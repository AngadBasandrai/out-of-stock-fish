import time
import numpy as np
import subprocess

EMPTY = 0

WP = 0
WN = 1
WB = 2
WR = 3
WQ = 4
WK = 5
BP = 6
BN = 7
BB = 8
BR = 9  
BQ = 10
BK = 11

WHITE = 0
BLACK = 1

KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
PAWN_ATTACKS = [[0] * 64, [0] * 64]

def set_bit(bb, sq):
    return bb | (1 << sq)

def pop_bit(bb, sq):
    return bb & ~(1 << sq)

def get_bit(bb, sq):
    return (bb >> sq) & 1

def pop_lsb(bb):
    lsb = bb & -bb
    return bb ^ lsb, lsb.bit_length() - 1

def init_leapers():
    for sq in range(64):
        r = sq >> 3
        f = sq & 7
        katk = 0; natk = 0; wp_atk = 0; bp_atk = 0
        for dr, df in ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)):
            rr = r + dr; ff = f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                natk |= 1 << (rr * 8 + ff)
        KNIGHT_ATTACKS[sq] = natk
        for dr, df in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            rr = r + dr; ff = f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                katk |= 1 << (rr * 8 + ff)
        KING_ATTACKS[sq] = katk
        if r < 7:
            if f > 0: wp_atk |= 1 << ((r+1)*8 + f - 1)
            if f < 7: wp_atk |= 1 << ((r+1)*8 + f + 1)
        PAWN_ATTACKS[WHITE][sq] = wp_atk
        if r > 0:
            if f > 0: bp_atk |= 1 << ((r-1)*8 + f - 1)
            if f < 7: bp_atk |= 1 << ((r-1)*8 + f + 1)
        PAWN_ATTACKS[BLACK][sq] = bp_atk

def rook_attacks(sq, occ):
    attacks = 0
    r = sq >> 3; f = sq & 7
    for rr in range(r + 1, 8):
        s = rr * 8 + f; attacks |= 1 << s
        if (occ >> s) & 1: break
    for rr in range(r - 1, -1, -1):
        s = rr * 8 + f; attacks |= 1 << s
        if (occ >> s) & 1: break
    for ff in range(f + 1, 8):
        s = r * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
    for ff in range(f - 1, -1, -1):
        s = r * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
    return attacks

def bishop_attacks(sq, occ):
    attacks = 0
    r = sq >> 3; f = sq & 7
    rr = r + 1; ff = f + 1
    while rr < 8 and ff < 8:
        s = rr * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
        rr += 1; ff += 1
    rr = r + 1; ff = f - 1
    while rr < 8 and ff >= 0:
        s = rr * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
        rr += 1; ff -= 1
    rr = r - 1; ff = f + 1
    while rr >= 0 and ff < 8:
        s = rr * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
        rr -= 1; ff += 1
    rr = r - 1; ff = f - 1
    while rr >= 0 and ff >= 0:
        s = rr * 8 + ff; attacks |= 1 << s
        if (occ >> s) & 1: break
        rr -= 1; ff -= 1
    return attacks

FLAG_NORMAL = 0
FLAG_EP = 1
FLAG_CASTLE = 2
FLAG_PROMO = 3

def encode_move(from_sq, to_sq, flag=FLAG_NORMAL):
    return (flag << 12) | (from_sq << 6) | to_sq

def decode_move(move):
    to_sq = move & 63
    from_sq = (move >> 6) & 63
    flag = (move >> 12) & 3
    return from_sq, to_sq, flag

class ChessGame:
    def __init__(self):
        init_leapers()

        self.engine = subprocess.Popen(
            ['elixir.exe'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

        self.send_command('uci')
        self.wait_for('uciok')

        self.send_command('isready')
        self.wait_for('readyok')

        self.reset()

    def send_command(self,cmd):
        self.engine.stdin.write(cmd + '\n')
        self.engine.stdin.flush()

    def read_line(self):
        return self.engine.stdout.readline().strip()

    def wait_for(self,token):
        while True:
            line = self.read_line()

            if token in line:
                return

    def evaluate_position(self):
        fen = self.to_fen()

        self.send_command(f'position fen {fen}')
        self.send_command('go depth 8')

        score = 0

        while True:
            line = self.read_line()

            if line.startswith('info depth'):
                parts = line.split()

                if 'score' in parts:
                    idx = parts.index('score')

                    if parts[idx + 1] == 'cp':
                        score = int(parts[idx + 2])

                    elif parts[idx + 1] == 'mate':
                        mate = int(parts[idx + 2])

                        if mate > 0:
                            score = 10000
                        else:
                            score = -10000

            elif line.startswith('bestmove'):
                break

        return score

    def reset(self):
        self.bitboards = [0] * 12
        self.bitboards[WP] = 0x000000000000FF00
        self.bitboards[WN] = 0x0000000000000042
        self.bitboards[WB] = 0x0000000000000024
        self.bitboards[WR] = 0x0000000000000081
        self.bitboards[WQ] = 0x0000000000000008
        self.bitboards[WK] = 0x0000000000000010
        self.bitboards[BP] = 0x00FF000000000000
        self.bitboards[BN] = 0x4200000000000000
        self.bitboards[BB] = 0x2400000000000000
        self.bitboards[BR] = 0x8100000000000000
        self.bitboards[BQ] = 0x0800000000000000
        self.bitboards[BK] = 0x1000000000000000
        self.side = WHITE
        self.castle = 15
        self.ep = -1
        self.halfmove = 0
        self.white_time = 300.0
        self.black_time = 300.0
        self.last_time = time.perf_counter()
        return self.get_state()

    def occupancies(self):
        bb = self.bitboards
        white = bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]
        black = bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11]
        return white, black, white | black

    def get_state(self):
        white, black, _ = self.occupancies()
        return np.array([
            self.bitboards[WP], self.bitboards[WN], self.bitboards[WB],
            self.bitboards[WR], self.bitboards[WQ], self.bitboards[WK],
            self.bitboards[BP], self.bitboards[BN], self.bitboards[BB],
            self.bitboards[BR], self.bitboards[BQ], self.bitboards[BK],
            white, black, self.side, self.castle, self.ep,
            self.white_time, self.black_time
        ], dtype=np.float64)

    def to_fen(self):
        board = [''] * 64

        piece_map = {
            WP:'P',WN:'N',WB:'B',WR:'R',WQ:'Q',WK:'K',
            BP:'p',BN:'n',BB:'b',BR:'r',BQ:'q',BK:'k'
        }

        for piece_idx in range(12):
            bb = self.bitboards[piece_idx]

            while bb:
                bb,sq = pop_lsb(bb)
                board[sq] = piece_map[piece_idx]

        rows = []

        for rank in range(7,-1,-1):
            row = ''
            empty = 0

            for file in range(8):
                sq = rank * 8 + file
                piece = board[sq]

                if piece == '':
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0

                    row += piece

            if empty:
                row += str(empty)

            rows.append(row)

        placement = '/'.join(rows)

        side = 'w' if self.side == WHITE else 'b'

        castle = ''

        if self.castle & 1: castle += 'K'
        if self.castle & 2: castle += 'Q'
        if self.castle & 4: castle += 'k'
        if self.castle & 8: castle += 'q'

        if castle == '':
            castle = '-'

        if self.ep >= 0:
            ep_file = chr(ord('a') + (self.ep & 7))
            ep_rank = str((self.ep >> 3) + 1)
            ep = ep_file + ep_rank
        else:
            ep = '-'

        return f'{placement} {side} {castle} {ep} {self.halfmove} 1'

    def square_attacked(self, sq, by_side):
        bb = self.bitboards
        white = bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]
        black = bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11]
        occ = white | black
        sq_bit = 1 << sq
        if by_side == WHITE:
            if PAWN_ATTACKS[BLACK][sq] & bb[WP]: return True
            knights = bb[WN]
            while knights:
                knights, s = pop_lsb(knights)
                if KNIGHT_ATTACKS[s] & sq_bit: return True
            diag = bb[WB] | bb[WQ]
            while diag:
                diag, s = pop_lsb(diag)
                if bishop_attacks(s, occ) & sq_bit: return True
            orth = bb[WR] | bb[WQ]
            while orth:
                orth, s = pop_lsb(orth)
                if rook_attacks(s, occ) & sq_bit: return True
            ks = bb[WK].bit_length() - 1
            if KING_ATTACKS[ks] & sq_bit: return True
        else:
            if PAWN_ATTACKS[WHITE][sq] & bb[BP]: return True
            knights = bb[BN]
            while knights:
                knights, s = pop_lsb(knights)
                if KNIGHT_ATTACKS[s] & sq_bit: return True
            diag = bb[BB] | bb[BQ]
            while diag:
                diag, s = pop_lsb(diag)
                if bishop_attacks(s, occ) & sq_bit: return True
            orth = bb[BR] | bb[BQ]
            while orth:
                orth, s = pop_lsb(orth)
                if rook_attacks(s, occ) & sq_bit: return True
            ks = bb[BK].bit_length() - 1
            if KING_ATTACKS[ks] & sq_bit: return True
        return False

    def _king_in_check(self, side):
        king_bb = self.bitboards[WK] if side == WHITE else self.bitboards[BK]
        if king_bb == 0: return True
        ks = king_bb.bit_length() - 1
        return self.square_attacked(ks, side ^ 1)

    def _apply_move(self, move):
        from_sq = (move >> 6) & 63
        to_sq = move & 63
        flag = (move >> 12) & 3
        bb = self.bitboards
        start = 0 if self.side == WHITE else 6
        end = 6 if self.side == WHITE else 12
        enemy_start = 6 if self.side == WHITE else 0
        enemy_end = 12 if self.side == WHITE else 6
        moved = -1
        from_bit = 1 << from_sq
        for i in range(start, end):
            if bb[i] & from_bit:
                moved = i; break
        if moved == -1: return False
        is_pawn = (moved == WP or moved == BP)
        is_capture = False
        to_bit = 1 << to_sq
        for i in range(enemy_start, enemy_end):
            if bb[i] & to_bit:
                bb[i] &= ~to_bit; is_capture = True; break
        bb[moved] = (bb[moved] & ~from_bit) | to_bit
        if flag == FLAG_EP:
            ep_cap = to_sq - 8 if self.side == WHITE else to_sq + 8
            cap_pawn = BP if self.side == WHITE else WP
            bb[cap_pawn] &= ~(1 << ep_cap); is_capture = True
        elif flag == FLAG_CASTLE:
            if to_sq == 6:   bb[WR] = (bb[WR] & ~(1 << 7)) | (1 << 5)
            elif to_sq == 2: bb[WR] = (bb[WR] & ~(1 << 0)) | (1 << 3)
            elif to_sq == 62: bb[BR] = (bb[BR] & ~(1 << 63)) | (1 << 61)
            elif to_sq == 58: bb[BR] = (bb[BR] & ~(1 << 56)) | (1 << 59)
        elif flag == FLAG_PROMO:
            bb[moved] &= ~to_bit
            queen = WQ if self.side == WHITE else BQ
            bb[queen] |= to_bit
        if moved == WK: self.castle &= ~3
        elif moved == BK: self.castle &= ~12
        elif moved == WR:
            if from_sq == 0: self.castle &= ~2
            elif from_sq == 7: self.castle &= ~1
        elif moved == BR:
            if from_sq == 56: self.castle &= ~8
            elif from_sq == 63: self.castle &= ~4
        if to_sq == 0: self.castle &= ~2
        elif to_sq == 7: self.castle &= ~1
        elif to_sq == 56: self.castle &= ~8
        elif to_sq == 63: self.castle &= ~4
        self.ep = -1
        if is_pawn and abs(to_sq - from_sq) == 16:
            self.ep = (from_sq + to_sq) >> 1
        if is_pawn or is_capture: self.halfmove = 0
        else: self.halfmove += 1
        return True

    def generate_pseudo_moves(self):
        moves = []
        bb = self.bitboards
        white = bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]
        black = bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11]
        occ = white | black
        if self.side == WHITE:
            own = white; enemy = black
            pawns = bb[WP]
            while pawns:
                pawns, sq = pop_lsb(pawns)
                r = sq >> 3; fwd = sq + 8
                if fwd < 64 and not (occ >> fwd) & 1:
                    if r == 6: moves.append(encode_move(sq, fwd, FLAG_PROMO))
                    else:
                        moves.append(encode_move(sq, fwd))
                        if r == 1:
                            fwd2 = sq + 16
                            if not (occ >> fwd2) & 1: moves.append(encode_move(sq, fwd2))
                atk = PAWN_ATTACKS[WHITE][sq]
                cap = atk & enemy
                while cap:
                    cap, to = pop_lsb(cap)
                    moves.append(encode_move(sq, to, FLAG_PROMO) if r == 6 else encode_move(sq, to))
                if self.ep >= 0 and atk & (1 << self.ep):
                    moves.append(encode_move(sq, self.ep, FLAG_EP))
            knights = bb[WN]
            while knights:
                knights, sq = pop_lsb(knights)
                attacks = KNIGHT_ATTACKS[sq] & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            bishops = bb[WB]
            while bishops:
                bishops, sq = pop_lsb(bishops)
                attacks = bishop_attacks(sq, occ) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            rooks = bb[WR]
            while rooks:
                rooks, sq = pop_lsb(rooks)
                attacks = rook_attacks(sq, occ) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            queens = bb[WQ]
            while queens:
                queens, sq = pop_lsb(queens)
                attacks = (bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            king = bb[WK]
            if king:
                ks = king.bit_length() - 1
                attacks = KING_ATTACKS[ks] & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(ks, to))
                if (self.castle & 1) and not (occ>>5)&1 and not (occ>>6)&1:
                    if not self.square_attacked(4,BLACK) and not self.square_attacked(5,BLACK) and not self.square_attacked(6,BLACK):
                        moves.append(encode_move(4, 6, FLAG_CASTLE))
                if (self.castle & 2) and not (occ>>3)&1 and not (occ>>2)&1 and not (occ>>1)&1:
                    if not self.square_attacked(4,BLACK) and not self.square_attacked(3,BLACK) and not self.square_attacked(2,BLACK):
                        moves.append(encode_move(4, 2, FLAG_CASTLE))
        else:
            own = black; enemy = white
            pawns = bb[BP]
            while pawns:
                pawns, sq = pop_lsb(pawns)
                r = sq >> 3; fwd = sq - 8
                if fwd >= 0 and not (occ >> fwd) & 1:
                    if r == 1: moves.append(encode_move(sq, fwd, FLAG_PROMO))
                    else:
                        moves.append(encode_move(sq, fwd))
                        if r == 6:
                            fwd2 = sq - 16
                            if not (occ >> fwd2) & 1: moves.append(encode_move(sq, fwd2))
                atk = PAWN_ATTACKS[BLACK][sq]
                cap = atk & enemy
                while cap:
                    cap, to = pop_lsb(cap)
                    moves.append(encode_move(sq, to, FLAG_PROMO) if r == 1 else encode_move(sq, to))
                if self.ep >= 0 and atk & (1 << self.ep):
                    moves.append(encode_move(sq, self.ep, FLAG_EP))
            knights = bb[BN]
            while knights:
                knights, sq = pop_lsb(knights)
                attacks = KNIGHT_ATTACKS[sq] & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            bishops = bb[BB]
            while bishops:
                bishops, sq = pop_lsb(bishops)
                attacks = bishop_attacks(sq, occ) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            rooks = bb[BR]
            while rooks:
                rooks, sq = pop_lsb(rooks)
                attacks = rook_attacks(sq, occ) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            queens = bb[BQ]
            while queens:
                queens, sq = pop_lsb(queens)
                attacks = (bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(sq, to))
            king = bb[BK]
            if king:
                ks = king.bit_length() - 1
                attacks = KING_ATTACKS[ks] & ~own
                while attacks:
                    attacks, to = pop_lsb(attacks); moves.append(encode_move(ks, to))
                if (self.castle & 4) and not (occ>>61)&1 and not (occ>>62)&1:
                    if not self.square_attacked(60,WHITE) and not self.square_attacked(61,WHITE) and not self.square_attacked(62,WHITE):
                        moves.append(encode_move(60, 62, FLAG_CASTLE))
                if (self.castle & 8) and not (occ>>59)&1 and not (occ>>58)&1 and not (occ>>57)&1:
                    if not self.square_attacked(60,WHITE) and not self.square_attacked(59,WHITE) and not self.square_attacked(58,WHITE):
                        moves.append(encode_move(60, 58, FLAG_CASTLE))
        return moves

    def generate_moves(self):
        legal = []
        saved_bbs = self.bitboards[:]
        saved_castle = self.castle
        saved_ep = self.ep
        saved_hm = self.halfmove
        saved_side = self.side
        for move in self.generate_pseudo_moves():
            self._apply_move(move)
            in_check = self._king_in_check(saved_side)
            self.bitboards = saved_bbs[:]
            self.castle = saved_castle
            self.ep = saved_ep
            self.halfmove = saved_hm
            self.side = saved_side
            if not in_check: legal.append(move)
        return legal

    def make_move(self, move):
        now = time.perf_counter()
        elapsed = now - self.last_time

        if self.side == WHITE:
            self.white_time -= elapsed
        else:
            self.black_time -= elapsed

        self.last_time = now

        eval_before = self.evaluate_position()

        reward = 0.0

        from_sq = (move >> 6) & 63
        to_sq = move & 63

        enemy_start = 6 if self.side == WHITE else 0
        enemy_end = 12 if self.side == WHITE else 6

        captured_piece = None

        for i in range(enemy_start,enemy_end):
            if self.bitboards[i] & (1 << to_sq):
                captured_piece = i
                break

        piece_rewards = {
            WP:0.1,WN:0.3,WB:0.3,WR:0.5,WQ:0.9,
            BP:0.1,BN:0.3,BB:0.3,BR:0.5,BQ:0.9
        }

        if captured_piece is not None:
            reward += piece_rewards.get(captured_piece,0.0)

        self._apply_move(move)

        eval_after = self.evaluate_position()

        delta = eval_after - eval_before

        if self.side == BLACK:
            delta = -delta

        reward += delta / 100.0

        self.side ^= 1

        legal = self.generate_moves()
        in_check = self._king_in_check(self.side)

        if not legal:
            if in_check:
                reward += 100.0
            else:
                reward -= 10.0

            self.reset()

            return self.get_state(),reward,True

        if self.halfmove >= 100:
            self.reset()
            return self.get_state(),-5.0,True

        return self.get_state(),reward,False
