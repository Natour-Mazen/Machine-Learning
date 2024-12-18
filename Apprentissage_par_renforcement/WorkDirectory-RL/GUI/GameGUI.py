import pygame

from GUI.Animator import Animator
from GUI.Board import Board
from GUI.InfoPanel import InfoPanel
from GUI.QTablePanel import QTablePanel
from GUI.Utils import TILE_SIZE, WHITE, BLACK, LIGHTGRAY

class GameGUI:
    """Main class for managing the GUI of the RL game."""

    def __init__(self, game, use_q_table=False):
        """
        Initializes the Game GUI.
        :param game: The game environment.
        :param use_q_table: Boolean indicating if Q-table visualization is needed.
        """
        self.game = game
        self.board = Board(game, offset_x=200, offset_y=120)
        self.info_panel = InfoPanel(offset_y=20)

        # Initialize the Q-table panel only if needed
        self.use_q_table = use_q_table
        if self.use_q_table:
            self.q_table_panel = QTablePanel(offset_x=game.width * TILE_SIZE + 250, offset_y=20)

        self.animator = Animator(board_offset_x=200, board_offset_y=100, clock=pygame.time.Clock())

        # Adjust screen size based on whether Q-table is used
        self.screen_width = game.width * TILE_SIZE + (800 if self.use_q_table else 400)
        self.screen_height = game.height * TILE_SIZE + 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RL Game")

    def _update_display(self, position, next_position, hit_wall, action, reward, Q=None):
        """Updates the display for both Q-Learning and Deep Q-Learning modes."""
        def draw_callbacks():
            self.screen.fill(WHITE)
            self.board.draw(self.screen)
            self.info_panel.draw(self.screen, position, action, reward)
            if self.use_q_table and Q is not None:
                self.q_table_panel.draw(self.screen, Q)

        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.animator.animate_movement(self.screen, position, next_position, hit_wall, action, draw_callbacks)
        pygame.display.flip()

    def update_display_Q_Learning(self, position, next_position, hit_wall, action, reward, Q):
        """Updates the display for Q-Learning mode."""
        if not self.use_q_table:
            raise ValueError("Q-table display is not enabled for this instance of GameGUI.")
        self._update_display(position, next_position, hit_wall, action, reward,  Q=Q)

    def update_display_Deep_Q_Learning(self, position, next_position, hit_wall, action, reward):
        """Updates the display for Deep Q-Learning mode."""
        self._update_display(position, next_position, hit_wall, action, reward)

    def display_end(self):
        """Displays a pop-up to indicate the end of the game."""
        font = pygame.font.Font(None, 36)
        text = "Game Over!"
        text_surface = font.render(text, True, BLACK, LIGHTGRAY)
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)
