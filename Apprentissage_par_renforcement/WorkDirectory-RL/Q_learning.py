import random
from Enums.Moves import Moves

def choose_action(position, epsilon, Q):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(list(Moves))
        rand = True
    else:
        position_actions = Q[position]
        action = max(position_actions, key=position_actions.get)
        rand = False
    return action, rand

def update_q_table(Q, position, action, reward, next_position, alpha, gamma):
    best_next_action = max(Q[next_position], key=Q[next_position].get)
    td_target = reward + gamma * Q[next_position][best_next_action]
    td_error = td_target - Q[position][action]
    Q[position][action] += alpha * td_error
    return Q

def print_q_table(Q):
    actions = list(next(iter(Q.values())).keys())
    action_names = '\t'.join([f"{action.name:<10}" for action in actions])
    print(f"Position\t\t{action_names}")
    for key, value in Q.items():
        action_values = '\t'.join([f"{val:<10.2f}" for val in value.values()])
        print(f"{key}\t\t\t{action_values}")



def q_learning(env, episodes, alpha, gamma, rewards):
    Q = {}

    for x in range(env.height):
        for y in range(env.width):
            Q[(x, y)] = {move: 0 for move in Moves}

    for episode in range(episodes):
        position = env.reset_player_position()
        done = False
        epsilon = 1 - (episode / episodes)
        if episode >= episodes - 5:
            epsilon = 0.

        while not done:
            action, rand = choose_action(position, epsilon, Q)
            next_position, reward, done = env.apply_action(action, env.board, rewards)
            Q = update_q_table(Q, position, action, reward, next_position, alpha, gamma)
            position = next_position
            print(f"===================Episode: {episode}================================")
            env.display_board()
            print(f"position: {position}")
            print(f"Action: {action}, Reward: {reward}")
            print(f"Rand move: {rand}")
            print_q_table(Q)
            #print(f"Q: {print_q_table(Q)}")

    return Q

