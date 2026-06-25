import numpy as np

EMPTY = 0
WP = 0; WN = 1; WB = 2; WR = 3; WQ = 4; WK = 5
BP = 6; BN = 7; BB = 8; BR = 9; BQ = 10; BK = 11
WHITE = 0; BLACK = 1

KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
PAWN_ATTACKS = [[0] * 64, [0] * 64]

_FLAG_NORMAL = 0
_FLAG_EP = 1
_FLAG_CASTLE = 2
_FLAG_PROMO = 3


def _pop_lsb(bb):
    lsb = bb & -bb
    return bb ^ lsb, lsb.bit_length() - 1


def _init_leapers():
    for sq in range(64):
        r = sq >> 3; f = sq & 7
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


def _rook_attacks(sq, occ):
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


def _bishop_attacks(sq, occ):
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


def encode_move(from_sq, to_sq, flag=_FLAG_NORMAL):
    return (flag << 12) | (from_sq << 6) | to_sq


def decode_move(move):
    return (move >> 6) & 63, move & 63, (move >> 12) & 3


class ChessGame:
    __slots__ = ('bitboards', 'side', 'castle', 'ep', 'halfmove')

    def __init__(self):
        _init_leapers()
        self.reset()

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
            0.0, 0.0
        ], dtype=np.float64)

    def square_attacked(self, sq, by_side):
        bb = self.bitboards
        occ = bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]|bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11]
        sq_bit = 1 << sq
        if by_side == WHITE:
            if PAWN_ATTACKS[BLACK][sq] & bb[WP]: return True
            n = bb[WN]
            while n:
                n, s = _pop_lsb(n)
                if KNIGHT_ATTACKS[s] & sq_bit: return True
            d = bb[WB] | bb[WQ]
            while d:
                d, s = _pop_lsb(d)
                if _bishop_attacks(s, occ) & sq_bit: return True
            o = bb[WR] | bb[WQ]
            while o:
                o, s = _pop_lsb(o)
                if _rook_attacks(s, occ) & sq_bit: return True
            ks = bb[WK].bit_length() - 1
            if KING_ATTACKS[ks] & sq_bit: return True
        else:
            if PAWN_ATTACKS[WHITE][sq] & bb[BP]: return True
            n = bb[BN]
            while n:
                n, s = _pop_lsb(n)
                if KNIGHT_ATTACKS[s] & sq_bit: return True
            d = bb[BB] | bb[BQ]
            while d:
                d, s = _pop_lsb(d)
                if _bishop_attacks(s, occ) & sq_bit: return True
            o = bb[BR] | bb[BQ]
            while o:
                o, s = _pop_lsb(o)
                if _rook_attacks(s, occ) & sq_bit: return True
            ks = bb[BK].bit_length() - 1
            if KING_ATTACKS[ks] & sq_bit: return True
        return False

    def _king_in_check(self, side):
        king_bb = self.bitboards[WK] if side == WHITE else self.bitboards[BK]
        if king_bb == 0: return True
        return self.square_attacked(king_bb.bit_length() - 1, side ^ 1)

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
        if moved == -1: return None
        is_pawn = moved == WP or moved == BP
        to_bit = 1 << to_sq
        captured = None
        for i in range(enemy_start, enemy_end):
            if bb[i] & to_bit:
                captured = i; bb[i] &= ~to_bit; break
        bb[moved] = (bb[moved] & ~from_bit) | to_bit
        if flag == _FLAG_EP:
            ep_sq = to_sq - 8 if self.side == WHITE else to_sq + 8
            cap = BP if self.side == WHITE else WP
            bb[cap] &= ~(1 << ep_sq)
        elif flag == _FLAG_CASTLE:
            if to_sq == 6: bb[WR] = (bb[WR] & ~(1 << 7)) | (1 << 5)
            elif to_sq == 2: bb[WR] = (bb[WR] & ~(1 << 0)) | (1 << 3)
            elif to_sq == 62: bb[BR] = (bb[BR] & ~(1 << 63)) | (1 << 61)
            elif to_sq == 58: bb[BR] = (bb[BR] & ~(1 << 56)) | (1 << 59)
        elif flag == _FLAG_PROMO:
            bb[moved] &= ~to_bit
            bb[WQ if self.side == WHITE else BQ] |= to_bit
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
        prev_ep = self.ep
        self.ep = -1
        if is_pawn and abs(to_sq - from_sq) == 16:
            self.ep = (from_sq + to_sq) >> 1
        if is_pawn or captured is not None:
            self.halfmove = 0
        else:
            self.halfmove += 1
        return moved, captured, prev_ep

    def _undo_move(self, move, moved, captured, prev_ep, prev_castle, prev_hm):
        from_sq = (move >> 6) & 63
        to_sq = move & 63
        flag = (move >> 12) & 3
        bb = self.bitboards
        from_bit = 1 << from_sq
        to_bit = 1 << to_sq
        if flag == _FLAG_PROMO:
            bb[moved] |= from_bit
            bb[WQ if moved < 6 else BQ] &= ~to_bit
        else:
            bb[moved] = (bb[moved] & ~to_bit) | from_bit
        if captured is not None:
            bb[captured] |= to_bit
        if flag == _FLAG_EP:
            ep_sq = to_sq - 8 if self.side == BLACK else to_sq + 8
            bb[BP if self.side == BLACK else WP] |= 1 << ep_sq
        elif flag == _FLAG_CASTLE:
            if to_sq == 6: bb[WR] = (bb[WR] & ~(1 << 5)) | (1 << 7)
            elif to_sq == 2: bb[WR] = (bb[WR] & ~(1 << 3)) | (1 << 0)
            elif to_sq == 62: bb[BR] = (bb[BR] & ~(1 << 61)) | (1 << 63)
            elif to_sq == 58: bb[BR] = (bb[BR] & ~(1 << 59)) | (1 << 56)
        self.castle = prev_castle
        self.ep = prev_ep
        self.halfmove = prev_hm

    def _generate_pseudo_moves(self):
        moves = []
        bb = self.bitboards
        white = bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]
        black = bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11]
        occ = white | black
        if self.side == WHITE:
            own = white; enemy = black
            p = bb[WP]
            while p:
                p, sq = _pop_lsb(p)
                r = sq >> 3; fwd = sq + 8
                if fwd < 64 and not (occ >> fwd) & 1:
                    if r == 6: moves.append(encode_move(sq, fwd, _FLAG_PROMO))
                    else:
                        moves.append(encode_move(sq, fwd))
                        if r == 1 and not (occ >> (sq + 16)) & 1:
                            moves.append(encode_move(sq, sq + 16))
                a = PAWN_ATTACKS[WHITE][sq]
                c = a & enemy
                while c:
                    c, to = _pop_lsb(c)
                    moves.append(encode_move(sq, to, _FLAG_PROMO) if r == 6 else encode_move(sq, to))
                if self.ep >= 0 and a & (1 << self.ep):
                    moves.append(encode_move(sq, self.ep, _FLAG_EP))
            n = bb[WN]
            while n:
                n, sq = _pop_lsb(n)
                a = KNIGHT_ATTACKS[sq] & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            b = bb[WB]
            while b:
                b, sq = _pop_lsb(b)
                a = _bishop_attacks(sq, occ) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            r = bb[WR]
            while r:
                r, sq = _pop_lsb(r)
                a = _rook_attacks(sq, occ) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            q = bb[WQ]
            while q:
                q, sq = _pop_lsb(q)
                a = (_bishop_attacks(sq, occ) | _rook_attacks(sq, occ)) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            k = bb[WK]
            if k:
                ks = k.bit_length() - 1
                a = KING_ATTACKS[ks] & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(ks, to))
                if (self.castle & 1) and not (occ >> 5) & 1 and not (occ >> 6) & 1:
                    if not self.square_attacked(4, BLACK) and not self.square_attacked(5, BLACK) and not self.square_attacked(6, BLACK):
                        moves.append(encode_move(4, 6, _FLAG_CASTLE))
                if (self.castle & 2) and not (occ >> 3) & 1 and not (occ >> 2) & 1 and not (occ >> 1) & 1:
                    if not self.square_attacked(4, BLACK) and not self.square_attacked(3, BLACK) and not self.square_attacked(2, BLACK):
                        moves.append(encode_move(4, 2, _FLAG_CASTLE))
        else:
            own = black; enemy = white
            p = bb[BP]
            while p:
                p, sq = _pop_lsb(p)
                r = sq >> 3; fwd = sq - 8
                if fwd >= 0 and not (occ >> fwd) & 1:
                    if r == 1: moves.append(encode_move(sq, fwd, _FLAG_PROMO))
                    else:
                        moves.append(encode_move(sq, fwd))
                        if r == 6 and not (occ >> (sq - 16)) & 1:
                            moves.append(encode_move(sq, sq - 16))
                a = PAWN_ATTACKS[BLACK][sq]
                c = a & enemy
                while c:
                    c, to = _pop_lsb(c)
                    moves.append(encode_move(sq, to, _FLAG_PROMO) if r == 1 else encode_move(sq, to))
                if self.ep >= 0 and a & (1 << self.ep):
                    moves.append(encode_move(sq, self.ep, _FLAG_EP))
            n = bb[BN]
            while n:
                n, sq = _pop_lsb(n)
                a = KNIGHT_ATTACKS[sq] & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            b = bb[BB]
            while b:
                b, sq = _pop_lsb(b)
                a = _bishop_attacks(sq, occ) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            r = bb[BR]
            while r:
                r, sq = _pop_lsb(r)
                a = _rook_attacks(sq, occ) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            q = bb[BQ]
            while q:
                q, sq = _pop_lsb(q)
                a = (_bishop_attacks(sq, occ) | _rook_attacks(sq, occ)) & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(sq, to))
            k = bb[BK]
            if k:
                ks = k.bit_length() - 1
                a = KING_ATTACKS[ks] & ~own
                while a:
                    a, to = _pop_lsb(a); moves.append(encode_move(ks, to))
                if (self.castle & 4) and not (occ >> 61) & 1 and not (occ >> 62) & 1:
                    if not self.square_attacked(60, WHITE) and not self.square_attacked(61, WHITE) and not self.square_attacked(62, WHITE):
                        moves.append(encode_move(60, 62, _FLAG_CASTLE))
                if (self.castle & 8) and not (occ >> 59) & 1 and not (occ >> 58) & 1 and not (occ >> 57) & 1:
                    if not self.square_attacked(60, WHITE) and not self.square_attacked(59, WHITE) and not self.square_attacked(58, WHITE):
                        moves.append(encode_move(60, 58, _FLAG_CASTLE))
        return moves

    def legal_moves(self):
        pseudo = self._generate_pseudo_moves()
        legal = []
        saved_side = self.side
        saved_castle = self.castle
        saved_ep = self.ep
        saved_hm = self.halfmove
        for move in pseudo:
            undo_info = self._apply_move(move)
            if undo_info is None:
                continue
            self.side ^= 1
            in_check = self._king_in_check(saved_side)
            if not in_check:
                legal.append(move)
            self.side = saved_side
            self._undo_move(move, *undo_info, saved_castle, saved_hm)
            self.castle = saved_castle
            self.ep = saved_ep
            self.halfmove = saved_hm
        return legal

    def make_move(self, move):
        self._apply_move(move)
        self.side ^= 1

        legal = self.legal_moves()
        in_check = self._king_in_check(self.side)

        if not legal:
            if in_check:
                return self.get_state(), 1.0, True
            return self.get_state(), 0.0, True

        if self.halfmove >= 100:
            return self.get_state(), 0.0, True

        return self.get_state(), 0.0, False
