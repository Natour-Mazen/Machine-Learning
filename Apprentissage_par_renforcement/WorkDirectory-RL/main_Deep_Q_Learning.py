import pygame

from Enums.Moves import Moves
from Enums.Rewards import Rewards
from GUI.GameGUI import GameGUI
from RLGame import RLGame
from Deep_Q_learning import q_deep_learning, choose_action, train_and_play_q_deep_learning, \
    load_and_play_q_deep_learning

if __name__ == '__main__':

    ####################################
    ## Parameters for Deep Q-learning ##
    ####################################
    gamma = 0.499
    # Simple model
    # episodes = 2000
    # better_model = False

    # Better model
    episodes = 2000
    better_model = True

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
    game = RLGame.define_basic_game()

    #######################################
    ##      Run Deep Q-learning          ##
    #######################################
    game_gui = GameGUI(game, use_q_table=False)
    #train_and_play_q_deep_learning(game, episodes, gamma, rewards_q_learning, game_gui, better_model)

    load_and_play_q_deep_learning(game, episodes, gamma, rewards_q_learning, game_gui, better_model)

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