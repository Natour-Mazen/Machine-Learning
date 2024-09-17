import numpy as np
import copy

class Jeu_Hanoi:
    def __init__(self):
        """Initialise un jeu de Hanoi avec trois pics et trois disques."""
        self.pic = np.zeros([3, 3], dtype=int)
        self.nombre_palet = np.zeros(3, dtype=int)
        self.situations = []

    def __str__(self):
        """Retourne une représentation en chaîne de caractères du nombre de disques sur chaque pic."""
        return str(self.nombre_palet)

    def get_situation(self):
        """Retourne une représentation en chaîne de caractères de la situation actuelle des pics."""
        return ''.join(map(str, self.pic.flatten()))

    def pic_vide(self, indice_pic):
        """Vérifie si un pic est vide.

        Args:
            indice_pic (int): L'indice du pic à vérifier.

        Returns:
            bool: True si le pic est vide, False sinon.
        """
        return np.all(self.pic[indice_pic] == 0)

    def get_pic_top_index(self, indice_pic):
        """Retourne l'indice du disque au sommet d'un pic.

        Args:
            indice_pic (int): L'indice du pic.

        Returns:
            int: L'indice du disque au sommet.
        """
        return self.nombre_palet[indice_pic] - 1

    def get_pic_top(self, indice_pic):
        """Retourne le disque au sommet d'un pic.

        Args:
            indice_pic (int): L'indice du pic.

        Returns:
            int: Le disque au sommet, ou None si le pic est vide.
        """
        if self.nombre_palet[indice_pic] == 0:
            return None
        return self.pic[indice_pic][self.get_pic_top_index(indice_pic)]

    def regle_jeu(self, indice_pic1, indice_pic2):
        """Vérifie si un déplacement est valide selon les règles du jeu.

        Args:
            indice_pic1 (int): L'indice du pic source.
            indice_pic2 (int): L'indice du pic cible.

        Returns:
            bool: True si le déplacement est valide, False sinon.
        """
        if self.pic_vide(indice_pic1):
            return False
        if self.pic_vide(indice_pic2):
            return True
        return self.get_pic_top(indice_pic1) < self.get_pic_top(indice_pic2)

    def deplacer(self, indice_pic1, indice_pic2):
        """Déplace un disque d'un pic à un autre.

        Args:
            indice_pic1 (int): L'indice du pic source.
            indice_pic2 (int): L'indice du pic cible.
        """
        palet = self.get_pic_top(indice_pic1)
        self.pic[indice_pic1][self.get_pic_top_index(indice_pic1)] = 0
        self.nombre_palet[indice_pic1] -= 1
        self.pic[indice_pic2][self.get_pic_top_index(indice_pic2) + 1] = palet
        self.nombre_palet[indice_pic2] += 1

    def effectue_deplacement(self, indice_pic1, indice_pic2):
        """Effectue un déplacement si celui-ci est valide et n'a pas encore été effectué.

        Args:
            indice_pic1 (int): L'indice du pic source.
            indice_pic2 (int): L'indice du pic cible.

        Returns:
            bool: True si le déplacement a été effectué, False sinon.
        """
        if not self.pic_vide(indice_pic1) and self.regle_jeu(indice_pic1, indice_pic2):
            new_jeu = copy.deepcopy(self)
            new_jeu.deplacer(indice_pic1, indice_pic2)
            sit = new_jeu.get_situation()
            if sit not in self.situations:
                self.deplacer(indice_pic1, indice_pic2)
                self.situations.append(sit)
                return True
        return False

    def afficher(self):
        """Affiche l'état actuel des pics."""
        for i in range(3):
            print(f"Pic {i}: {self.pic[i]}")
        print("\n")