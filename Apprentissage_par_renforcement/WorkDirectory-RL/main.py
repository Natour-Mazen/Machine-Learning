# Description: This file is the main file that will be run to start the game.
from Enums.Moves import Moves
from RLGame import RLGame

# Initialisation du jeu
game = RLGame.define_basic_game()
game.display_board()

# Plateau de jeu actuel
space = game.board  # L'organisation du plateau est représentée par game.board

# Test d'une action
new_position, reward, game_over = game.apply_action( Moves.DOWN, space, {})  # Action "down"
print(f"New Position: {new_position}, Reward: {reward}, Game Over: {game_over}")
