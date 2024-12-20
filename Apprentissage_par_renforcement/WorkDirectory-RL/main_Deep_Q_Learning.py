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
    gamma = 0.999
    learn = False # True learn else run the saved model.

    # Simple model
    episodes_random = 1000
    episodes = 1000
    better_model = False

    # Better model
    # episodes_random = 1000
    # episodes = 1000
    # better_model = True

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

    random_spawn = True

    if learn:
        train_and_play_q_deep_learning(game, episodes_random, episodes, gamma, rewards_q_learning, better_model, random_spawn)
    else:
        load_and_play_q_deep_learning(game, rewards_q_learning, better_model)