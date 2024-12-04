import pygame
from PIL import Image, ImageDraw

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 100
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load images
PLAYER_IMG = pygame.image.load('images/player.png')
DRAGON_IMG = pygame.image.load('images/dragon.png')
START_IMG = pygame.image.load('images/start.png')
END_IMG = pygame.image.load('images/end.png')
BACKGROUND_IMG = pygame.image.load('images/background.png')

# Scale images to fit the tile size
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (TILE_SIZE, TILE_SIZE))
DRAGON_IMG = pygame.transform.scale(DRAGON_IMG, (TILE_SIZE, TILE_SIZE))
START_IMG = pygame.transform.scale(START_IMG, (TILE_SIZE, TILE_SIZE))
END_IMG = pygame.transform.scale(END_IMG, (TILE_SIZE, TILE_SIZE))
BACKGROUND_IMG = pygame.transform.scale(BACKGROUND_IMG, (TILE_SIZE, TILE_SIZE))


class GameGUI:
    def __init__(self, env):
        self.env = env
        self.board_offset_x = 200  # Décalage horizontal pour le plateau
        self.board_offset_y = 100  # Décalage vertical pour le plateau
        self.info_offset_y = 20    # Décalage vertical pour les infos du joueur
        self.q_table_offset_x = env.width * TILE_SIZE + 250  # Décalage horizontal pour la Q-table
        self.q_table_offset_y = 20  # Décalage vertical pour la Q-table

        # Calcul de la taille de l'écran
        self.screen_width = env.width * TILE_SIZE + 800
        self.screen_height = env.height * TILE_SIZE + 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RL Game")
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_board(self):
        """Dessine le plateau de jeu et ses éléments."""
        for i in range(self.env.height):
            for j in range(self.env.width):
                x = self.board_offset_x + j * TILE_SIZE
                y = self.board_offset_y + i * TILE_SIZE
                self.screen.blit(BACKGROUND_IMG, (x, y))
                if (i, j) == self.env.player_position:
                    self.screen.blit(PLAYER_IMG, (x, y))
                elif (i, j) == self.env.start_pos:
                    self.screen.blit(START_IMG, (x, y))
                elif (i, j) == self.env.end_pos:
                    self.screen.blit(END_IMG, (x, y))
                elif (i, j) in self.env.dragons:
                    self.screen.blit(DRAGON_IMG, (x, y))

    def draw_info(self, position, action, reward):
        """Affiche les informations sur la position, l'action et la récompense."""
        font = pygame.font.Font(None, 36)
        info_text = [
            f"Position: {position}",
            f"Action: {action.name if action else 'N/A'}",
            f"Reward: {reward}"
        ]
        for i, text in enumerate(info_text):
            info_surface = font.render(text, True, BLACK)
            self.screen.blit(info_surface, (20, self.info_offset_y + i * 40))

    def draw_q_table(self, Q):
        """Affiche la Q-table sous forme de tableau aligné."""
        font = pygame.font.Font(None, 24)
        x_offset = self.q_table_offset_x
        y_offset = self.q_table_offset_y

        # Effacer la zone Q-table
      #  self.screen.fill(WHITE, (self.q_table_offset_x - 20, 0, 400, self.screen_height))

        # Titre
        header = font.render("Q-Table", True, BLACK)
        self.screen.blit(header, (x_offset, y_offset))
        y_offset += 40

        # En-têtes de colonnes
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        col_width = 100  # Largeur de chaque colonne
        header_x = x_offset
        self.screen.blit(font.render("Position", True, BLACK), (header_x, y_offset))
        header_x += 150  # Décalage pour les colonnes des actions
        for action in actions:
            self.screen.blit(font.render(action, True, BLACK), (header_x, y_offset))
            header_x += col_width
        y_offset += 30

        # Affichage des lignes de la Q-table
        for position, actions in Q.items():
            # Afficher la position
            row_x = x_offset
            self.screen.blit(font.render(str(position), True, BLACK), (row_x, y_offset))
            row_x += 150  # Décalage pour les colonnes des actions

            # Afficher les valeurs pour chaque action

            for action, action_values in actions.items():
                self.screen.blit(font.render(f"{action_values:.2f}", True, BLACK), (row_x, y_offset))
                row_x += col_width

            y_offset += 30

    def animate_movement(self, start_pos, end_pos, hit_wall, action, reward, Q):
        """Anime le déplacement du joueur."""
        start_x = self.board_offset_x + start_pos[1] * TILE_SIZE
        start_y = self.board_offset_y + start_pos[0] * TILE_SIZE
        end_x = self.board_offset_x + end_pos[1] * TILE_SIZE
        end_y = self.board_offset_y + end_pos[0] * TILE_SIZE
        steps = 10  # Diviser en 10 étapes pour une animation fluide
        delta_x = (end_x - start_x) / steps
        delta_y = (end_y - start_y) / steps

        for step in range(steps):
            current_x = start_x + step * delta_x
            current_y = start_y + step * delta_y

            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_info(self.env.player_position, action, reward)
            self.draw_q_table(Q)

            if not hit_wall:
                self.screen.blit(PLAYER_IMG, (current_x, current_y))
            else:
                self.screen.blit(PLAYER_IMG, (start_x, start_y))

            pygame.display.update()
            self.clock.tick(FPS)

    def update_display(self, position, next_position, hit_wall, action, reward, Q):
        """Mise à jour complète de l'écran."""
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        self.animate_movement(position, next_position, hit_wall, action, reward, Q)
        pygame.display.flip()
