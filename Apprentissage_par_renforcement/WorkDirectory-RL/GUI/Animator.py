import pygame
from adodbapi.ado_consts import directions

from GUI.Utils import TILE_SIZE, PLAYER_IMG, FPS, draw_arrow


class Animator:
    """Handles animations for the game."""

    def __init__(self, board_offset_x, board_offset_y, clock):
        self.board_offset_x = board_offset_x
        self.board_offset_y = board_offset_y
        self.clock = clock

    def animate_movement(self, screen, start_pos, end_pos, hit_wall, action, draw_callbacks):
        """Animates player movement with optional hit-wall effect."""
        start_x = self.board_offset_x + start_pos[1] * TILE_SIZE
        start_y = self.board_offset_y + start_pos[0] * TILE_SIZE
        end_x = self.board_offset_x + end_pos[1] * TILE_SIZE
        end_y = self.board_offset_y + end_pos[0] * TILE_SIZE
        steps = 10
        delta_x = (end_x - start_x) / steps
        delta_y = (end_y - start_y) / steps

        direction = action.name if action else self.get_direction(delta_x, delta_y)

        if hit_wall:
            self.flash_effect(screen, start_x, start_y, direction, draw_callbacks)
        else:
            self.smooth_animation(screen, start_x, start_y, delta_x, delta_y, steps, direction, draw_callbacks)

    def flash_effect(self, screen, start_x, start_y, direction, draw_callbacks):
        """Flashes the player position to indicate a collision."""
        flash_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        flash_surface.fill((255, 255, 255, 128))

        for _ in range(3):
            draw_callbacks()
            screen.blit(flash_surface, (start_x, start_y))
            draw_arrow(screen, (start_x, start_y), direction)
            pygame.display.update()
            pygame.time.delay(100)

            draw_callbacks()
            pygame.display.update()
            pygame.time.delay(100)

    def smooth_animation(self, screen, start_x, start_y, delta_x, delta_y, steps, direction, draw_callbacks):
        """Performs a smooth movement animation."""
        for step in range(steps):
            current_x = start_x + step * delta_x
            current_y = start_y + step * delta_y
            draw_callbacks()
            screen.blit(PLAYER_IMG, (current_x, current_y))
            draw_arrow(screen, (current_x, current_y), direction)
            pygame.display.update()
            self.clock.tick(FPS)

    def get_direction(self, delta_x, delta_y):
        """Determines the direction of movement based on deltas."""
        if delta_x > 0:
            return 'RIGHT'
        elif delta_x < 0:
            return 'LEFT'
        elif delta_y > 0:
            return 'DOWN'
        return 'UP'