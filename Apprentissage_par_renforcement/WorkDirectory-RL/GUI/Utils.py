import pygame

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 100
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHTGRAY = (200, 200, 200)

# Load images
def load_and_scale_image(path, size):
    """Loads and scales an image to the given size."""
    image = pygame.image.load(path)
    return pygame.transform.scale(image, (size, size))

PLAYER_IMG = load_and_scale_image('GUI/Images/player.png', TILE_SIZE)
DRAGON_IMG = load_and_scale_image('GUI/Images/dragon.png', TILE_SIZE)
START_IMG = load_and_scale_image('GUI/Images/start.png', TILE_SIZE)
END_IMG = load_and_scale_image('GUI/Images/end.png', TILE_SIZE)
BACKGROUND_IMG = load_and_scale_image('GUI/Images/background.png', TILE_SIZE)

# Utility Functions
def draw_arrow(surface, position, direction):
    """Draws an arrow indicating the direction of movement, centered in the tile."""
    x, y = position
    center_x = x + TILE_SIZE // 2
    center_y = y + TILE_SIZE // 2
    half_size = TILE_SIZE // 8

    if direction == 'UP':
        points = [(center_x, center_y - half_size), (center_x - half_size, center_y + half_size), (center_x + half_size, center_y + half_size)]
    elif direction == 'DOWN':
        points = [(center_x, center_y + half_size), (center_x - half_size, center_y - half_size), (center_x + half_size, center_y - half_size)]
    elif direction == 'LEFT':
        points = [(center_x - half_size, center_y), (center_x + half_size, center_y - half_size), (center_x + half_size, center_y + half_size)]
    elif direction == 'RIGHT':
        points = [(center_x + half_size, center_y), (center_x - half_size, center_y - half_size), (center_x - half_size, center_y + half_size)]

    pygame.draw.polygon(surface, WHITE, points)