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
    # gamma = 0.999
    # episodes = 10000 # More efficient rewards

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
    ##      Run Deep Q-learning          ##
    #######################################
    game_gui = GameGUI(env, use_q_table=False)
    #train_and_play_q_deep_learning(env, episodes, gamma, rewards_q_learning, True)

    # load_and_play_q_deep_learning(env, episodes, gamma, rewards_q_learning, True)


    # trained_model = q_deep_learning(env, episodes, gamma, rewards_q_learning, True)
    #
    # env_test = RLGame.define_basic_game()
    #
    # print("=== Reel game ===")
    # print("=== Start ===")
    # env_test.display_board()
    # print(f"position: {env_test.player_position}")
    #
    # game_gui = GameGUI(env_test)
    # Q = {}
    #
    # for x in range(env.height):
    #     for y in range(env.width):
    #         Q[(x, y)] = {move: 0 for move in Moves}
    #
    # while True:
    #     env_test.reset_player_position()
    #     done = False
    #     while not done:
    #
    #         action, vec_position = choose_action(env_test, 0, trained_model, env_test.width, env_test.height)
    #         next_position, reward, done, wall_hit = env_test.apply_action(action, env_test.board, rewards_q_learning)
    #
    #         env_test.display_board()
    #         print(f"position: {env_test.player_position}")
    #         print(f"Action: {action}, Reward: {reward}")
    #         print("===  ===")
    #         game_gui.update_display(env_test.player_position, next_position, wall_hit, action, reward, Q)

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