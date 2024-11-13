import numpy as np

NORMAL_REWARD = -1     # Reward for the jail cell
ENNEMIES_REWARD = -10  # Reward for the dragon cell
END_REWARD = 0   # Reward for the normal cell


class Game:
    def __init__(self, width, height, start_pos, end_pos, dragons):
        """
        Initializes the game board and special elements.
        """
        self.width = width
        self.height = height
        self.player_position = start_pos
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.dragons = dragons
        self.board = self.create_board()

    def create_board(self):
        """
        Creates a board with special cells (start, end, dragons).
        """
        board = np.zeros((self.height, self.width))
        board[self.start_pos] = 1  # Start cell
        board[self.end_pos] = 2  # End cell
        for dragon in self.dragons:
            board[dragon] = -1  # Cells with dragons
        return board

    def apply_action(self, action):
        """
        Applies an action to the player and returns the new position, reward, and game status.
        """
        # Possible moves: 0 = up, 1 = down, 2 = left, 3 = right
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        dx, dy = moves.get(action, (0, 0))
        new_position = (self.player_position[0] + dx, self.player_position[1] + dy)

        # Check board limits
        if not (0 <= new_position[0] < self.height and 0 <= new_position[1] < self.width):
            new_position = self.player_position  # Stay in the same position if out of bounds

        # Check if the new position is a dragon, the end, or neutral
        if new_position in self.dragons:
            reward = ENNEMIES_REWARD  # Negative reward for a dragon
            game_over = True
        elif new_position == self.end_pos:
            reward = END_REWARD  # Positive reward for reaching the end
            game_over = True
        else:
            reward = NORMAL_REWARD  # Standard reward for a regular move
            game_over = False

        # Update player position
        self.player_position = new_position

        return new_position, reward, game_over

    def reset(self):
        """
        Resets the game to the starting position.
        """
        self.player_position = self.start_pos
        return self.player_position

    def display_board(self):
        """
        Displays the game board with the player's current position.
        """
        board = self.board.copy()
        x, y = self.player_position
        board[x, y] = 3  # Mark the player
        print(board)


