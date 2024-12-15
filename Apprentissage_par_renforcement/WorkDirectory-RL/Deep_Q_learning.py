import numpy as np
import random
import keras
from keras import Sequential
from keras.src.layers import Dense
import tensorflow as tf

from Enums.Rewards import Rewards
from helper import plot
from Q_learning import print_q_board

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


def get_vector_position(env, width, height):
    """ To get the input vector with the player position"""
    vector_position = np.zeros(width * height)
    position = env.player_position
    index = position[0] * width + position[1]
    vector_position[index] = 1
    return vector_position

def choose_action(env, epsilon, model, width, height):
    """ To choose an action randomly if epsilon is high else an action with the model"""
    vector_position = get_vector_position(env, width, height)
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(Moves)), vector_position
    else:

        q_values = model.predict(np.array([vector_position]),verbose = 0)
        return Moves(np.argmax(q_values)), vector_position

def q_deep_learning(env, episodes, gamma, rewards, better_model : bool = False):

    # Initialize model
    if better_model:
        model = build_better_model(env.width * env.height, len(Moves))
    else:
        model = build_model(env.width * env.height, len(Moves))

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

    # For the UI
    Q = {}
    for x in range(env.height):
        for y in range(env.width):
            Q[(x, y)] = {move: 0 for move in Moves}

    # Run Deep Q-learning
    for episode in range(episodes):
        # Reset the player position and the end of the game.
        env.reset_player_position()
        done = False
        # Epsilon if function of the number of episodes (the more, the less we use randomness when predict).
        epsilon = 1 - (episode / episodes)
        # We stop the randomness of the model.
        if (better_model and episode >= episodes / 1.5) or episode >= episodes / 2:
            epsilon = 0.
            b_full_model = True

        total_rewards = 0
        total_steps = 0

        # We run one game.
        while not done:
            # action = UP, DOWN, LEFT or RIGHT.
            # Choose the best action with the "model" or a random position (depend on the epsilon).
            action, vec_position = choose_action(env, epsilon, model, env.width, env.height)
            # Play this action and get the reward.
            next_position, reward, done, wall_hit = env.apply_action(action, env.board, rewards)
            # Choose the next best action possible with the "target" model.
            next_vec_position = get_vector_position(env, env.width, env.height)
            next_Q = target.predict(np.array([next_vec_position]),verbose = 0)
            next_Q_max = np.max(next_Q)

            if b_full_model:
                print(f"===================Episode: {episode}================================")
                env.display_board()
                print(f"position: {env.player_position}")
                print(f"Action: {action}, Reward: {reward}")
                print(f"Done: {done}")

                # game_gui.update_display(env.player_position, next_position, wall_hit, action, reward, Q)

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
                        return model
                # If we lost the game on an enemy.
                elif reward == Rewards.ENEMIES.value and total_wins > 0:
                    total_wins -= 1

            # If we make too many steps, we break the loop.
            if total_steps > 100:
                done = True
                reward = Rewards.LOOP.value

            if b_full_model:
                Q_target = {}

                for x in range(env.height):
                    for y in range(env.width):
                        v_position = np.zeros(4 * 4)
                        index = x * 4 + y
                        v_position[index] = 1
                        pre = model.predict(np.array([v_position]),verbose = 0)
                        pre = pre[0]

                        Q_target[(x, y)] = {tup[0]: tup[1] for tup in zip(Moves, pre)}

                print_q_board(Q_target)

            # The value of the reward that we want.
            t = reward + gamma * next_Q_max * (1 - done)

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

    if better_model:
        model.save('models/better_model.keras')
    else:
        model.save('models/model.keras')
    return model

# def play_q_deep_learning(model, rewards):
#     print("=== Start reel game ===")
#     game_board = RLGame.define_basic_game()
#     game_board.display_board()
#
#     game_gui = GameGUI(game_board)
#
#     running = True
#     while running:
#         game_board.reset_player_position()
#         done = False
#         while not done:
#             action, vec_position = choose_action(game_board, 0, model, game_board.width, game_board.height)
#             next_position, reward, done, wall_hit = game_board.apply_action(action, game_board.board, rewards)
#
#             game_board.display_board()
#             print(f"position: {game_board.player_position}")
#             print(f"Action: {action}, Reward: {reward}")
#             print("===  ===")
#             game_gui.update_display(game_board.player_position, next_position, wall_hit, action, reward)
#
#             for event in pygame.event.get():
#                 # If we qui the window.
#                 if event.type == pygame.QUIT:
#                     running = False
#
#                 # If a key is pressed.
#                 if event.type == pygame.KEYDOWN:
#                     # If the "q" key is pressed.
#                     if event.key == pygame.K_q:
#                         running = False
#
#     pygame.quit()


def train_and_play_q_deep_learning(env, episodes, gamma, rewards, better_model = False):
    trained_model = q_deep_learning(env, episodes, gamma, rewards, better_model)

    # play_q_deep_learning(trained_model, rewards)


def load_and_play_q_deep_learning(env, episodes, gamma, rewards, better_model = False):

    if better_model:
        trained_model =  keras.models.load_model('models/better_model.keras')
    else:
        trained_model = keras.models.load_model('models/model.keras')

    # play_q_deep_learning(trained_model, rewards)
