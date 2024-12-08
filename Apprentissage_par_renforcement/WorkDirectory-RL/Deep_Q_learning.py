import numpy as np
import random
import keras
from keras import Sequential
from keras.src.layers import Dense
import tensorflow as tf
from helper import plot
from Q_learning import print_q_board

from Enums.Moves import Moves

def build_model(input_shape = 16, output_shape = 4):
    model = Sequential([
        #Dense(16, activation='relu', input_shape=[input_shape]),
        Dense(8, activation='relu', input_shape=[input_shape]),
        Dense(output_shape)
    ])
    #model.compile(optimizer='adam', loss='mse')
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

def q_deep_learning(env, episodes, gamma, rewards, game_gui):

    # Initialize model
    model = build_model(env.width * env.height, len(Moves))

    # Declaration of the optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # Declaration of the loss function (MSE)
    loss_fn = keras.losses.mean_squared_error

    target = tf.keras.models.clone_model(model)
    target.set_weights(model.get_weights())

    rewards_array = []
    steps_array = []

    # For the UI
    Q = {}
    for x in range(env.height):
        for y in range(env.width):
            Q[(x, y)] = {move: 0 for move in Moves}

    # Run Deep Q-learning
    for episode in range(episodes):
        env.reset_player_position()
        done = False
        # Epsilon
        epsilon = 1 - (episode / episodes)
        if episode >= episodes / 1.2:
            epsilon = 0.

        total_rewards = 0
        total_steps = 0

        while not done:
            # action = UP, DOWN, LEFT or RIGHT.
            action, vec_position = choose_action(env, epsilon, model, env.width, env.height)
            next_position, reward, done, wall_hit = env.apply_action(action, env.board, rewards)
            next_action, next_vec_position = choose_action(env, 0, target, env.width, env.height)
            next_Q = target.predict(np.array([next_vec_position]),verbose = 0)
            next_Q_max = np.max(next_Q)

            if epsilon <= 0.2:
                print(f"===================Episode: {episode}================================")
                env.display_board()
                print(f"position: {env.player_position}")
                print(f"Action: {action}, Reward: {reward}")
                print(f"Done: {done}")

                # game_gui.update_display(env.player_position, next_position, wall_hit, action, reward, Q)

            t = reward + gamma * next_Q_max * (1 - done)

            with tf.GradientTape() as tape:
                predict = model(np.array([vec_position]))

                val_predict = predict[:, action.value]

                loss = loss_fn([t], [val_predict])

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # if epsilon <= 0.2:
            #     Q_target = {}
            #
            #     for x in range(env.height):
            #         for y in range(env.width):
            #             v_position = np.zeros(4 * 4)
            #             index = x * 4 + y
            #             v_position[index] = 1
            #             pre = model.predict(np.array([v_position]),verbose = 0)
            #             pre = pre[0]
            #
            #             Q_target[(x, y)] = {tup[0]: tup[1] for tup in zip(Moves, pre)}
            #
            #     print_q_board(Q_target)

            if epsilon <= 0.2:
                print(f"yTrue: {t}, yPred: {val_predict}")

            total_rewards += reward
            total_steps += 1

        rewards_array.append(total_rewards)
        steps_array.append(total_steps)

        if episode % 10 == 0:
            target.set_weights(model.get_weights())
            plot(rewards_array, steps_array)

    return model
