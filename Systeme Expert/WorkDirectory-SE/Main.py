from Hanoi_Game import Jeu_Hanoi
from Plot_Display import animate_hanoi
from Resolve_Hanoi import solve_optimal, solve, solve_find_best

if __name__ == '__main__':
    initial_game_state = Jeu_Hanoi()
    initial_game_state.nombre_palet[0] = 3
    initial_game_state.pic[0, 0] = 3
    initial_game_state.pic[0, 1] = 2
    initial_game_state.pic[0, 2] = 1
    initial_game_state.add_current_situation()

    final_game_state = Jeu_Hanoi()
    final_game_state.nombre_palet[2] = 3
    final_game_state.pic[2, 0] = 3
    final_game_state.pic[2, 1] = 2
    final_game_state.pic[2, 2] = 1
    final_game_state.add_current_situation()

    # Résolu en 26 déplacement.
    # move_history = solve(initial_game_state, final_game_state, [(0, 1), (1, 2), (2, 1), (1, 0)])

    # res_opti = solve_optimal(initial_game_state, final_game_state)
    # animate_hanoi(res_opti)

    best_moves, number_of_moves, best_param = solve_find_best(initial_game_state, final_game_state, [(0, 1), (1, 2), (2, 1), (1, 0), (2, 0), (0, 2)])

    print("Nombre de coups pour la meilleure solution : ", number_of_moves)
    print("Meilleurs paramètres : ", best_param)
    print("Coups fait : ")
    for m in best_moves:
        print(m)
    animate_hanoi(best_moves)

    # Résolu en 8 déplacement.
    # res_best = solve(initial_game_state, final_game_state, [(1, 2), (0, 2), (0, 1), (2, 1), (2, 0), (1, 0)])
    # animate_hanoi(res_best)


