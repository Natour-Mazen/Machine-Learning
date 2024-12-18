import random
from Enums.Moves import Moves
import numpy as np

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

def get_string_position(position):
    if position == (0, 0):
        return "(Start) "
    elif position == (3, 3):
        return " (End)  "
    elif position in [(1, 0), (1, 2), (2, 3), (3, 1)]:
        return "(Dragon)"
    else:
        return " (Path) "

def print_q_board(Q):
    # Print the 'x' index of the board.
    print(f"\t\t\t\t", end="")
    for i in range(4):
        print(f"  {i}\t\t\t\t\t\t", end="")
    print("\n")

    for y in range(4):
        line_values = np.full(4, {})
        for x in range(4):
            line_values[x] = Q[(y, x)]

        # Print the UP values.
        print(f"\t\t\t\t", end="")
        for i in range(4):
            print(f"{line_values[i][Moves.UP]: 6.2f}\t\t\t\t\t", end="")
        print("")

        # Print the 'y' index of the board.
        print(f"{y}\t", end="")
        # Print the LEFT and RIGHT values.
        for i in range(4):
            print(
                f"\t{line_values[i][Moves.LEFT]: 6.2f}  {get_string_position((y, i))} {line_values[i][Moves.RIGHT]: 6.2f}",
                end="")
        print("")

        # Print the DOWN values.
        print(f"\t\t\t\t", end="")
        for i in range(4):
            print(f"{line_values[i][Moves.DOWN]: 6.2f} \t\t\t\t\t", end="")
        print("\n")


def print_q_table(Q):
    actions = list(next(iter(Q.values())).keys())
    action_names = '\t'.join([f"{action.name:<10}" for action in actions])
    print(f"Position\t\t{action_names}")
    for key, value in Q.items():
        action_values = '\t'.join([f"{val:<10.2f}" for val in value.values()])
        print(f"{key}\t\t\t{action_values}")

def q_learning(game, episodes, alpha, gamma, rewards, game_gui = None):
    Q = {}

    for x in range(game.height):
        for y in range(game.width):
            Q[(x, y)] = {move: 0 for move in Moves}

    for episode in range(episodes):

        position = game.reset_player_position()
        done = False
        epsilon = 1 - (episode / episodes)
        if episode >= episodes - 5:
            epsilon = 0.

        while not done:
            action, rand = choose_action(position, epsilon, Q)
            next_position, reward, done, hit_wall = game.apply_action(action, game.board, rewards)
            Q = update_q_table(Q, position, action, reward, next_position, alpha, gamma)
            if game_gui:
                game_gui.update_display_Q_Learning(position, next_position, hit_wall, action, reward, Q)
            position = next_position
            print(f"===================Episode: {episode}================================")
            game.display_board()
            print(f"position: {position}")
            print(f"Action: {action}, Reward: {reward}")
            print(f"Rand move: {rand}")
            print_q_table(Q)
            # print_q_board(Q)
            # print(f"Q: {print_q_table(Q)}")
            # sleep(0.5)


