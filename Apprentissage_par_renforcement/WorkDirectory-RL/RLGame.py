# RLGame.py
import numpy as np
from Enums.Moves import Moves
from Enums.Rewards import Rewards
from Enums.Symbols import Symbols
import random

class RLGame:
    def __init__(self, width, height, start_pos, end_pos, dragons):
        self.width = width
        self.height = height
        self.player_position = start_pos
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.dragons = dragons
        self.board = self.create_board()

    def create_board(self):
        board = np.full((self.height, self.width), Symbols.EMPTY.value)
        board[self.start_pos] = Symbols.START.value
        board[self.end_pos] = Symbols.END.value
        for dragon in self.dragons:
            board[dragon] = Symbols.DRAGON.value
        return board

    def apply_action(self, action, space, rewards):
        moves = {
            Moves.UP: (-1, 0),
            Moves.DOWN: (1, 0),
            Moves.LEFT: (0, -1),
            Moves.RIGHT: (0, 1)
        }

        dx, dy = moves.get(action, (0, 0))
        new_position = (self.player_position[0] + dx, self.player_position[1] + dy)
        hit_wall = False

        if not (0 <= new_position[0] < self.height and 0 <= new_position[1] < self.width):
            new_position = self.player_position
            reward = rewards.get('wall', Rewards.WALL.value)
            game_over = False
            hit_wall = True
            return new_position, reward, game_over, hit_wall

        cell_type = space[new_position]
        if cell_type == Symbols.DRAGON.value:
            reward = rewards.get('dragon', Rewards.ENEMIES.value)
            game_over = True
        elif cell_type == Symbols.END.value:
            reward = rewards.get('end', Rewards.END.value)
            game_over = True
        else:
            reward = rewards.get('normal', Rewards.NORMAL.value)
            game_over = False

        self.player_position = new_position

        return new_position, reward, game_over, hit_wall

    def reset_player_position(self):
        self.player_position = self.start_pos
        return self.player_position

    def display_board(self):
        board = self.board.copy()
        x, y = self.player_position
        board[x, y] = Symbols.PLAYER.value

        for row in board:
            print(' '.join(row))

    def reset_random_player_position(self):
        while True:
            position_index = random.randrange(0, 16)
            index_x = position_index % self.width
            index_y = position_index // self.width
            symbols = self.board[index_x, index_y]
            if symbols == Symbols.EMPTY.value:
                self.player_position = (index_x, index_y)
                return self.player_position

    @staticmethod
    def define_basic_game():
        width = 4
        height = 4
        start_pos = (0, 0)
        end_pos = (3, 3)
        dragons = [(1, 0), (1, 2), (2, 3), (3, 1)]
        return RLGame(width, height, start_pos, end_pos, dragons)
