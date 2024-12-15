import pygame

from GUI.Utils import BLACK


class QTablePanel:
    """Handles the drawing of the Q-table."""

    def __init__(self, offset_x, offset_y):
        self.offset_x = offset_x
        self.offset_y = offset_y

    def draw(self, surface, Q):
        """Displays the Q-table as an aligned table."""
        font = pygame.font.Font(None, 24)
        x_offset = self.offset_x
        y_offset = self.offset_y

        # Title
        header = font.render("Q-Table", True, BLACK)
        surface.blit(header, (x_offset, y_offset))
        y_offset += 40

        # Column headers
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        col_width = 110
        header_x = x_offset
        surface.blit(font.render("Position", True, BLACK), (header_x, y_offset))
        header_x += 150
        for action in actions:
            surface.blit(font.render(action, True, BLACK), (header_x, y_offset))
            header_x += col_width
        y_offset += 30

        # Rows
        for position, action_values in Q.items():
            row_x = x_offset
            surface.blit(font.render(str(position), True, BLACK), (row_x, y_offset))
            row_x += 150
            for value in action_values.values():
                color = (0, 255, 0) if value > 0 else (255, 0, 0) if value < 0 else (0, 0, 0)
                surface.blit(font.render(f"{value:.2f}", True, color), (row_x, y_offset))
                row_x += col_width
            y_offset += 30