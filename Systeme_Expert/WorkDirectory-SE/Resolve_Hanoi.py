import copy
import itertools

####################################################
##      Fonction solve Demandee par le Sujet      ##
####################################################
def solve(initial_game, final_game, moves : list) -> list:
    """Résout le jeu de Hanoi en trouvant une séquence de déplacements.

        Args:
            initial_game (Jeu_Hanoi): L'état initial du jeu.
            final_game (Jeu_Hanoi): L'état final du jeu.
            moves : Les coups possibles.
        Returns:
            list: Une liste des états du jeu après chaque déplacement.
        """
    history, finished = _solve(initial_game, final_game, moves)
    return history

def _solve(initial_game, final_game, moves : list) -> (list, bool):
    """Résout le jeu de Hanoi en trouvant une séquence de déplacements.

    Args:
        initial_game (Jeu_Hanoi): L'état initial du jeu.
        final_game (Jeu_Hanoi): L'état final du jeu.
        moves : Les coups possibles.

    Returns:
        list: Une liste des états du jeu après chaque déplacement.
        bool: Une boolean pour savoir si le jeu est terminé ou pas
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

        possible_moves = moves.copy()

        if last_move:
            possible_moves.remove((last_move[1], last_move[0]))

        move_made = False
        for source_peg, target_peg in possible_moves:
            if initial_game.effectue_deplacement(source_peg, target_peg):
                last_move = (source_peg, target_peg)
                move_made = True
                break

        if not move_made:
            return moves_history, False

        number_of_moves += 1
        moves_history.append(copy.deepcopy(initial_game))
        print(f"Après {number_of_moves} déplacement(s):")
        initial_game.afficher()

    print(f"Joué en {number_of_moves} déplacements.")
    return moves_history, True

####################################################################
##      Fonction solve pour trouver les meilleurs paramètres      ##
####################################################################
def solve_find_best(initial_game, final_game, moves: list) -> (list, int, list):
    """Test toutes les permutations de mouvements possibles et retourne la meilleure solution.

    Args:
        initial_game (Jeu_Hanoi): L'état initial du jeu.
        final_game (Jeu_Hanoi): L'état final du jeu.
        moves: La liste des coups possibles.

    Returns:
        tuple: La meilleure séquence des états du jeu et le nombre de coups.
    """
    permutations = list(itertools.permutations(moves))  # Toutes les permutations possibles des mouvements
    moves_list = [list(p) for p in permutations]

    best_situation = []
    best_number_of_moves = float('inf')  # Fixe un maximum initial pour le nombre de coups
    best_param = []

    for _moves in moves_list:
        ig = copy.deepcopy(initial_game)
        fg = copy.deepcopy(final_game)

        # Solve
        history, done = _solve(ig, fg, _moves)

        if len(history) < best_number_of_moves and done:
            best_number_of_moves = len(history)
            best_situation = history
            best_param = _moves.copy()

    return best_situation, best_number_of_moves, best_param

####################################################
##       Fonction solve Optimal en 7 Coups        ##
####################################################

def hanoi_recursive(n, source, target, auxiliary, game_state, moves):
    """
    Résout le problème de la Tour de Hanoi de manière récursive.

    Args:
        n (int): Nombre de disques à déplacer.
        source (int): Indice du pic source.
        target (int): Indice du pic cible.
        auxiliary (int): Indice du pic auxiliaire.
        game_state (Jeu_Hanoi): État actuel du jeu.
        moves (list): Liste des états du jeu après chaque déplacement.
    """
    if n > 0:
        # Déplacer n-1 disques du pic source au pic auxiliaire
        hanoi_recursive(n - 1, source, auxiliary, target, game_state, moves)

        # Déplacer le nième disque du pic source au pic cible
        if game_state.effectue_deplacement(source, target):
            # Ajouter l'état actuel du jeu à la liste des mouvements
            moves.append(copy.deepcopy(game_state))
            # Afficher l'état actuel du jeu
            print(f"Après {len(moves)} déplacement(s):")
            game_state.afficher()

        # Déplacer n-1 disques du pic auxiliaire au pic cible
        hanoi_recursive(n - 1, auxiliary, target, source, game_state, moves)


def solve_optimal(initial_game_state, final_game_state):
    """
    Résout le jeu de Hanoi de manière optimale en utilisant la fonction récursive.

    Args:
        initial_game_state (Jeu_Hanoi): État initial du jeu.
        final_game_state (Jeu_Hanoi): État final du jeu.

    Returns:
        list: Liste des états du jeu après chaque déplacement.
    """
    moves = []
    # Afficher l'état initial du jeu
    print("État initial:")
    initial_game_state.afficher()
    # Appeler la fonction récursive pour résoudre le problème
    hanoi_recursive(3, 0, 2, 1, initial_game_state, moves)
    # Afficher le nombre total de coups joués
    print(f"Nombre de coups joués: {len(moves)}")
    # Retourner la liste des mouvements
    return moves
