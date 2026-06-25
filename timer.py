import time


class GameClock:
    def __init__(self, total_seconds, increment=0):
        self.total = total_seconds
        self.increment = increment
        self.remaining = total_seconds
        self.move_start = None
        self.moves_made = 0

    def start_move(self):
        self.move_start = time.monotonic()

    def end_move(self):
        if self.move_start is not None:
            elapsed = time.monotonic() - self.move_start
            self.remaining -= elapsed
            self.remaining += self.increment
            if self.remaining < 0:
                self.remaining = 0
            self.move_start = None
        self.moves_made += 1

    def move_alloc(self):
        if self.remaining <= 0:
            return 0.1
        est_moves_left = max(10, 40 - self.moves_made)
        base = self.remaining / est_moves_left
        return min(base * 1.5, self.remaining * 0.25)

    def move_time_up(self):
        if self.move_start is None:
            return False
        alloc = self.move_alloc()
        return (time.monotonic() - self.move_start) >= alloc

    def time_up(self):
        return self.remaining <= 0
