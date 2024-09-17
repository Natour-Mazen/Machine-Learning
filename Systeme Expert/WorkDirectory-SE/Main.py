from Hanoi_Game import Jeu_Hanoi
from Plot_Display import animate_hanoi
from Resolve_Hanoi import solve_optimal, solve

if __name__ == '__main__':
    initial_game_state = Jeu_Hanoi()
    initial_game_state.nombre_palet[0] = 3
    initial_game_state.pic[0, 0] = 3
    initial_game_state.pic[0, 1] = 2
    initial_game_state.pic[0, 2] = 1

    final_game_state = Jeu_Hanoi()
    final_game_state.nombre_palet[2] = 3
    final_game_state.pic[2, 0] = 3
    final_game_state.pic[2, 1] = 2
    final_game_state.pic[2, 2] = 1

    move_history = solve(initial_game_state, final_game_state)
    #move_history = solve_optimal(initial_game_state, final_game_state)
    animate_hanoi(move_history)
