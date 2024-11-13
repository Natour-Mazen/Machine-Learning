# Description: This file is the main file that will be run to start the game.
from Game import RLGame

game = RLGame.define_basic_game()
game.display_board()

# Test an action
new_position, reward, game_over = game.apply_action(1)  # Action "down"
print(f"New Position: {new_position}, Reward: {reward}, Game Over: {game_over}")