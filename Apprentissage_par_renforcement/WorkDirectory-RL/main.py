from Enums.Moves import Moves
from Enums.Rewards import Rewards
from Q_learning import q_learning
from RLGame import RLGame
from Deep_Q_learning import play_optimal_policy, build_model, update_model, choose_action

if __name__ == '__main__':
    # Parameters for Q-learning
    alpha = 0.9
    gamma = 0.5
    # episodes = 200 # Simple rewards
    episodes = 40 # More efficient rewards
    rewards_q_learning = {'normal': Rewards.NORMAL.value,
                          'dragon': Rewards.ENEMIES.value,
                          'end': Rewards.END.value,
                          'wall': Rewards.WALL.value}

    # Initialize environment
    env = RLGame.define_basic_game()

    # Run Q-learning
    Q = q_learning(env, episodes, alpha, gamma, rewards_q_learning)

    # Parameters for Deep Q-learning
    # alpha = 0.9
    # gamma = 0.5
    # epsilon = 0.1
    # episodes = 1000
    # rewards_deep_q_learning = {'normal': -1, 'dragon': -20, 'end': 100}
    #
    # # Initialize model
    # model = build_model(env.width * env.height, len(Moves))
    #
    # # Run Deep Q-learning
    # for episode in range(episodes):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = choose_action(state, epsilon, model, env.width, env.height)
    #         next_state, reward, done = env.apply_action(action, env.board, rewards_deep_q_learning)
    #         update_model(model, state, action, reward, next_state, done, gamma, env.width, env.height)
    #         state = next_state
    #
    # # Play with optimal policy
    # play_optimal_policy(env, Q, rewards_q_learning)