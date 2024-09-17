from Hanoi_Game import Jeu_Hanoi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

def solve(initial_game, final_game):
    """Résout le jeu de Hanoi en trouvant une séquence de déplacements.

    Args:
        initial_game (Jeu_Hanoi): L'état initial du jeu.
        final_game (Jeu_Hanoi): L'état final du jeu.

    Returns:
        list: Une liste des états du jeu après chaque déplacement.
    """
    number_of_moves = 0
    moves_history = []
    visited_situations = set()
    last_move = None

    print("État initial:")
    initial_game.afficher()

    while initial_game.get_situation() != final_game.get_situation():
        current_situation = initial_game.get_situation()
        if current_situation in visited_situations:
            break
        visited_situations.add(current_situation)

        possible_moves = [(0, 1), (0, 2), (1, 2), (2, 1), (1, 0), (2, 0)]
        if last_move:
            possible_moves.remove((last_move[1], last_move[0]))

        move_made = False
        for source_peg, target_peg in possible_moves:
            if initial_game.effectue_deplacement(source_peg, target_peg):
                last_move = (source_peg, target_peg)
                move_made = True
                break

        if not move_made:
            return moves_history

        number_of_moves += 1
        moves_history.append(copy.deepcopy(initial_game))
        print(f"Après {number_of_moves} déplacement(s):")
        initial_game.afficher()

    print(f"Joué en {number_of_moves} déplacements.")
    return moves_history

def plot_hanoi(pegs, ax):
    """Trace l'état actuel des pics de Hanoi.

    Args:
        pegs (list): Une liste de pics contenant les disques.
        ax (matplotlib.axes.Axes): L'axe sur lequel tracer.
    """
    ax.clear()
    ax.set_xlim(-1, 3)
    ax.set_ylim(0, len(pegs[0]) + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    for peg_index, peg in enumerate(pegs):
        for disk_index, disk in enumerate(peg):
            if disk != 0:
                disk_width = disk / len(pegs)
                disk_height = 0.5
                rect = plt.Rectangle(
                    (peg_index - disk_width / 2, disk_index * disk_height),
                    disk_width,
                    disk_height,
                    edgecolor='k',
                    facecolor='c'
                )
                ax.add_patch(rect)

    for i in range(3):
        ax.plot([i, i], [0, len(pegs[0]) + 1], 'k-', lw=2)

def animate_hanoi(move_history, interval=500):
    """Anime la résolution du jeu de Hanoi.

    Args:
        move_history (list): Une liste des états du jeu après chaque déplacement.
        interval (int): L'intervalle de temps entre chaque image en millisecondes.
    """
    fig, ax = plt.subplots()

    def update(frame):
        plot_hanoi(move_history[frame].pic, ax)
        return ax.patches

    ani = animation.FuncAnimation(fig, update, frames=len(move_history), interval=interval, repeat=False)
    plt.show(block=True)

if __name__ == '__main__':
    initial_game_state = Jeu_Hanoi()
    initial_game_state.nombre_palet[0] = 3
    initial_game_state.pic[0, 0] = 3
    initial_game_state.pic[0, 1] = 2
    initial_game_state.pic[0, 2] = 1

    final_game_state = Jeu_Hanoi()
    final_game_state.nombre_palet[1] = 3
    final_game_state.pic[2, 0] = 3
    final_game_state.pic[2, 1] = 2
    final_game_state.pic[2, 2] = 1

    move_history = solve(initial_game_state, final_game_state)
    animate_hanoi(move_history)