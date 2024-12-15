import pygame

from GUI.Utils import BLACK

class InfoPanel:
    """Handles the drawing of player information."""

    def __init__(self, offset_y):
        self.offset_y = offset_y

    def draw(self, surface, position, action, reward):
        """Displays information about the player's state."""
        font = pygame.font.Font(None, 36)
        info_text = [
            f"Position: {position}",
            f"Action: {action.name if action else 'N/A'}",
            f"Reward: {reward}"
        ]
        for i, text in enumerate(info_text):
            info_surface = font.render(text, True, BLACK)
            surface.blit(info_surface, (20, self.offset_y + i * 40))