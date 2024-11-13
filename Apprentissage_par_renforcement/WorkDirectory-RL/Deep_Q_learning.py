import numpy as np
import random
from keras import Sequential
from keras.src.layers import Dense

from Enums.Moves import Moves

def play_optimal_policy(env, Q, rewards):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = max(Q[state], key=Q[state].get)
        next_state, reward, done = env.apply_action(action, env.board, rewards)
        total_reward += reward
        state = next_state
        env.display_board()
        print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(16, input_shape=(input_shape,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def state_to_one_hot(state, width, height):
    one_hot = np.zeros(width * height)
    index = state[0] * width + state[1]
    one_hot[index] = 1
    return one_hot

def choose_action(state, epsilon, model, width, height):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(Moves))
    else:
        state_one_hot = state_to_one_hot(state, width, height)
        q_values = model.predict(np.array([state_one_hot]))
        return Moves(np.argmax(q_values))

def update_model(model, state, action, reward, next_state, done, gamma, width, height):
    state_one_hot = state_to_one_hot(state, width, height)
    next_state_one_hot = state_to_one_hot(next_state, width, height)
    target = reward
    if not done:
        target += gamma * np.amax(model.predict(np.array([next_state_one_hot]))[0])
    target_f = model.predict(np.array([state_one_hot]))
    target_f[0][action.value] = target
    model.fit(np.array([state_one_hot]), target_f, epochs=1, verbose=0)