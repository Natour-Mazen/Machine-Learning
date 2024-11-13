import random
from Enums.Moves import Moves

def choose_action(position, epsilon, Q):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(list(Moves))
    else:
        state_actions = Q[position]
        action = max(state_actions, key=state_actions.get)
    return action

def update_q_table(Q, state, action, reward, next_state, alpha, gamma):
    best_next_action = max(Q[next_state], key=Q[next_state].get)
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error
    return Q

def q_learning(env, episodes, alpha, gamma, epsilon, rewards):
    Q = {}
    for x in range(env.height):
        for y in range(env.width):
            Q[(x, y)] = {move: 0 for move in Moves}

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = choose_action(state, epsilon, Q)
            next_state, reward, done = env.apply_action(action, env.board, rewards)
            Q = update_q_table(Q, state, action, reward, next_state, alpha, gamma)
            state = next_state
            env.display_board()
            print(f"===================Episode: {episode}================================")
            print(f"State: {state}")
            print(f"Action: {action}, Reward: {reward}")


    return Q

