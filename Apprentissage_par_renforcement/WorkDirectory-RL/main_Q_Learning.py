import pygame

from Enums.Rewards import Rewards
from GUI.GameGUI import GameGUI
from Q_learning import q_learning
from RLGame import RLGame


if __name__ == '__main__':
    ####################################
    ##    Parameters for Q-learning   ##
    ####################################
    alpha = 0.9
    gamma = 0.5
    # episodes = 1000 # Simple rewards
    episodes = 100 # More efficient rewards

    #######################################
    ##         General rewards           ##
    #######################################
    rewards_q_learning = { 'normal': Rewards.NORMAL.value,
                           'dragon': Rewards.ENEMIES.value,
                           'end': Rewards.END.value,
                           'wall': Rewards.WALL.value}

    ##############################################
    ## Initialize environment with the Game GUI ##
    ##############################################
    env = RLGame.define_basic_game()

    #######################################
    ##          Run Q-learning           ##
    #######################################
    game_gui = GameGUI(env, use_q_table=True)
    q_learning(env, episodes, alpha, gamma, rewards_q_learning, game_gui)

    #######################################
    ##                End                ##
    #######################################
    while True:
        game_gui.display_end()
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.display.update()
        pygame.time.delay(100)