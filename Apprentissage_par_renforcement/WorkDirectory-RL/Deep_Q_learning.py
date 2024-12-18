import numpy as np
import random
import keras
import pygame
from keras import Sequential
from keras.src.layers import Dense
import tensorflow as tf

from Enums.Rewards import Rewards
from helper import plot
from Q_learning import print_q_board
from GUI.GameGUI import GameGUI

from Enums.Moves import Moves

def build_model(input_shape = 16, output_shape = 4):
    model = Sequential([
        Dense(8, activation='relu', input_shape=[input_shape]),
        Dense(output_shape)
    ])
    return model

def build_better_model(input_shape = 16, output_shape = 4):
    model = Sequential([
        Dense(8, activation='relu', input_shape=[input_shape]),
        Dense(8, activation='relu', input_shape=[8]),
        Dense(output_shape)
    ])
    return model


def get_vector_position(game, width, height):
    """ To get the input vector with the player position"""
    vector_position = np.zeros(width * height)
    position = game.player_position
    index = position[0] * width + position[1]
    vector_position[index] = 1
    return vector_position

def choose_action(game, epsilon, model, width, height):
    """ To choose an action randomly if epsilon is high else an action with the model"""
    vector_position = get_vector_position(game, width, height)
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(Moves)), vector_position
    else:

        q_values = model.predict(np.array([vector_position]),verbose = 0)
        return Moves(np.argmax(q_values)), vector_position

def q_deep_learning(game, episodes, gamma, rewards, better_model : bool = False):

    # Initialize model
    if better_model:
        model = build_better_model(game.width * game.height, len(Moves))
    else:
        model = build_model(game.width * game.height, len(Moves))

    # Declaration of the optimizer.
    if better_model:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    # Declaration of the loss function (MSE)
    loss_fn = keras.losses.mean_squared_error

    target = tf.keras.models.clone_model(model)
    target.set_weights(model.get_weights())

    rewards_array = []
    steps_array = []

    b_full_model = False
    total_wins = 0
    model_done = False

    # The last position of the last game.
    last_game_finish = -1
    # The number of time the last position has been played.
    last_game_finish_nb = 0

    # Run Deep Q-learning
    for episode in range(episodes):
        # Reset the player position and the end of the game.
        game.reset_random_player_position()
        done = False
        # Epsilon if function of the number of episodes (the more, the less we use randomness when predict).
        epsilon = 1 - (episode / episodes)

        # We stop the randomness of the model.
        if episode >= episodes * 0.9:
            epsilon = 0.
            b_full_model = True

        total_rewards = 0
        total_steps = 0

        array_position_nb = np.zeros([16])

        # We run one game.
        while not done:
            # action = UP, DOWN, LEFT or RIGHT.
            # Choose the best action with the "model" or a random position (depend on the epsilon).
            action, vec_position = choose_action(game, epsilon, model, game.width, game.height)
            # Play this action and get the reward.
            next_position, reward, done, wall_hit = game.apply_action(action, game.board, rewards)
            # Choose the next best action possible with the "target" model.
            next_vec_position = get_vector_position(game, game.width, game.height)
            next_Q = target.predict(np.array([next_vec_position]), verbose = 0)
            next_Q_max = np.max(next_Q)

            current_position = np.argmax(vec_position)
            array_position_nb[current_position] += 1

            # if done:
            #     if last_game_finish == current_position:
            #         last_game_finish_nb += 1
            #         # The reward is multiplied
            #         reward *= last_game_finish_nb
            #     else:
            #         last_game_finish = current_position
            #         last_game_finish_nb = 1

            if b_full_model:
                print(f"===================Episode: {episode}================================")
                game.display_board()
                print(f"position: {game.player_position}")
                print(f"Action: {action}, Reward: {reward}")
                print(f"Done: {done}")

                # game_gui.update_display(game.player_position, next_position, wall_hit, action, reward, Q)

            if done and reward == Rewards.ENEMIES.value and b_full_model:
                if total_wins > 0:
                    total_wins -= 1

            if done and b_full_model:
                # If we have found the end of the game.
                if reward == Rewards.END.value:
                    total_wins += 1
                    print(f"total_wins: {total_wins}")
                    if total_wins >= 10:
                        print(f"======================= return =============\n\n\n\n\n\n\n\n\n\n:")
                        model_done = True
                        break
                # If we lost the game on an enemy.
                elif reward == Rewards.ENEMIES.value and total_wins > 0:
                    total_wins -= 1

            # If we make too many steps, we break the loop.
            if (total_steps > 10 and b_full_model) or (total_steps > 100 and not b_full_model):
                done = True
                reward = Rewards.LOOP.value

            # if b_full_model:
            #     Q_target = {}
            #
            #     for x in range(game.height):
            #         for y in range(game.width):
            #             v_position = np.zeros(4 * 4)
            #             index = x * 4 + y
            #             v_position[index] = 1
            #             pre = model.predict(np.array([v_position]),verbose = 0)
            #             pre = pre[0]
            #
            #             Q_target[(x, y)] = {tup[0]: tup[1] for tup in zip(Moves, pre)}
            #
            #     print_q_board(Q_target)

            # The value of the reward that we want.
            reward_multiplier = 1
            if array_position_nb[current_position] > 1:
                reward_multiplier = array_position_nb[current_position]

            t = reward + gamma * next_Q_max * (1 - done) * (1 - wall_hit)

            # The gradient tape to allow the backpropagation.
            with tf.GradientTape() as tape:
                predict = model(np.array([vec_position]))

                val_predict = predict[:, action.value]

                loss = loss_fn([t], [val_predict])

            # We adapt the model neurones.
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if b_full_model:
                print(f"yTrue: {t}, yPred: {val_predict}")

            # We add our reward to the total.
            total_rewards += reward
            # We count the number of step until we finish the game.
            total_steps += 1

        # Arrays to display the plot of rewards and number of step.
        rewards_array.append(total_rewards)
        steps_array.append(total_steps)

        # Each 10 episodes, we update the "target" model with the current one ("model").
        if episode % 10 == 0:
            target.set_weights(model.get_weights())
            plot(rewards_array, steps_array)

        if episode % 100 == 0:
            Q_target = {}

            for x in range(game.height):
                for y in range(game.width):
                    v_position = np.zeros(4 * 4)
                    index = x * 4 + y
                    v_position[index] = 1
                    pre = model.predict(np.array([v_position]), verbose=0)
                    pre = pre[0]

                    Q_target[(x, y)] = {tup[0]: tup[1] for tup in zip(Moves, pre)}

            print_q_board(Q_target)

        if model_done:
            break

    Q_target = {}

    for x in range(game.height):
        for y in range(game.width):
            v_position = np.zeros(4 * 4)
            index = x * 4 + y
            v_position[index] = 1
            pre = model.predict(np.array([v_position]), verbose=0)
            pre = pre[0]

            Q_target[(x, y)] = {tup[0]: tup[1] for tup in zip(Moves, pre)}

    print_q_board(Q_target)

    if better_model:
        model.save('better_model.keras')
    else:
        model.save('model.keras')
    return model

def play_q_deep_learning(game, model, rewards, game_gui):
    print("=== Start reel game ===")
    game.display_board()

    running = True
    while running:
        game.reset_player_position()
        done = False
        while not done:
            action, vec_position = choose_action(game, 0, model, game.width, game.height)
            next_position, reward, done, wall_hit = game.apply_action(action, game.board, rewards)

            game.display_board()
            print(f"position: {game.player_position}")
            print(f"Action: {action}, Reward: {reward}")
            print("===  ===")

            game_gui.update_display_Deep_Q_Learning(game.player_position, next_position, wall_hit, action, reward)

    pygame.quit()


def train_and_play_q_deep_learning(game, episodes, gamma, rewards, game_gui, better_model = False):
    trained_model = q_deep_learning(game, episodes, gamma, rewards, better_model)

    play_q_deep_learning(game, trained_model, rewards, game_gui)


def load_and_play_q_deep_learning(game, episodes, gamma, rewards, game_gui, better_model = False):

    if better_model:
        trained_model =  keras.models.load_model('better_model.keras')
    else:
        trained_model = keras.models.load_model('model.keras')

    play_q_deep_learning(game, trained_model, rewards, game_gui)
