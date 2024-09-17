import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_hanoi(pegs, ax, num_moves):
    """Trace l'état actuel des pics de Hanoi.

    Args:
        pegs (list): Une liste de pics contenant les disques.
        ax (matplotlib.axes.Axes): L'axe sur lequel tracer.
        num_moves (int): Le nombre de coups effectués.
    """
    ax.clear()
    ax.set_xlim(-1, 3)
    ax.set_ylim(0, len(pegs[0]) + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#FF8C33', '#FF3333', '#33FF8C',
              '#FF33F5']

    for peg_index, peg in enumerate(pegs):
        for disk_index, disk in enumerate(peg):
            if disk != 0:
                disk_width = disk / len(pegs)
                disk_height = 0.5
                rect = plt.Rectangle(
                    (peg_index - disk_width / 2, disk_index * disk_height),
                    disk_width,
                    disk_height,
                    edgecolor='black',
                    facecolor=colors[disk % len(colors)]
                )
                ax.add_patch(rect)

    for i in range(3):
        ax.plot([i, i], [0, len(pegs[0]) + 1], 'k-', lw=2)

    ax.set_title("== Tower of Hanoi ==", pad=20, fontweight='bold')
    ax.text(-1.5, len(pegs[0]) + 0.5, f"Moves: {num_moves}", ha='center', fontweight='bold')


def animate_hanoi(move_history, interval=500):
    """Anime la résolution du jeu de Hanoi.

    Args:
        move_history (list): Une liste des états du jeu après chaque déplacement.
        interval (int): L'intervalle de temps entre chaque image en millisecondes.
    """
    fig, ax = plt.subplots()

    def update(frame):
        plot_hanoi(move_history[frame].pic, ax, frame + 1)
        return ax.patches

    ani = animation.FuncAnimation(fig, update, frames=len(move_history), interval=interval, repeat=False)
    plt.show(block=True)
