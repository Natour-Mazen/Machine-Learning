from GUI.Utils import TILE_SIZE, BACKGROUND_IMG, PLAYER_IMG, START_IMG, END_IMG, DRAGON_IMG


class Board:
    """Handles the drawing and management of the game board."""

    def __init__(self, game, offset_x, offset_y):
        self.game = game
        self.offset_x = offset_x
        self.offset_y = offset_y

    def draw(self, surface):
        """Draws the game board and its elements."""
        for i in range(self.game.height):
            for j in range(self.game.width):
                x = self.offset_x + j * TILE_SIZE
                y = self.offset_y + i * TILE_SIZE
                surface.blit(BACKGROUND_IMG, (x, y))

                if (i, j) == self.game.player_position:
                    surface.blit(PLAYER_IMG, (x, y))
                elif (i, j) == self.game.start_pos:
                    surface.blit(START_IMG, (x, y))
                elif (i, j) == self.game.end_pos:
                    surface.blit(END_IMG, (x, y))
                elif (i, j) in self.game.dragons:
                    surface.blit(DRAGON_IMG, (x, y))