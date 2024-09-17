from Hanoi_Game import Jeu_Hanoi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy


def solve(initial_game_state, final_game_state):
    nb_coups_joues = 0
    moves = []

    print(initial_game_state)
    while initial_game_state.get_situation() != final_game_state.get_situation():
        if not any(initial_game_state.effectue_deplacement(i, j) for i, j in [(0, 1), (1, 2), (2, 1), (1, 0)]):
            return moves
        nb_coups_joues += 1
        moves.append(copy.deepcopy(initial_game_state))
        #print(initial_game_state)

    print(f"Joues en {nb_coups_joues} deplacements.")
    return moves


def plot_hanoi(pegs, ax):
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


def animate_hanoi(moves, interval=500):
    fig, ax = plt.subplots()

    def update(frame):
        plot_hanoi(moves[frame].pic, ax)
        return ax.patches

    ani = animation.FuncAnimation(fig, update, frames=len(moves), interval=interval, repeat=False)
    plt.show(block=True)


if __name__ == '__main__':
    jeu_initial = Jeu_Hanoi()
    jeu_initial.nombre_palet[0] = 3
    jeu_initial.pic[0, 0] = 3
    jeu_initial.pic[0, 1] = 2
    jeu_initial.pic[0, 2] = 1

    jeu_final = Jeu_Hanoi()
    jeu_final.nombre_palet[1] = 3
    jeu_final.pic[2, 0] = 3
    jeu_final.pic[2, 1] = 2
    jeu_final.pic[2, 2] = 1

    moves = solve(jeu_initial, jeu_final)
    animate_hanoi(moves, 1000)
