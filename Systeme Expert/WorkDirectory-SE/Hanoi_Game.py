import numpy as np
import copy


class Jeu_Hanoi:
    def __init__(self):
        self.pic = np.zeros([3, 3], dtype=int)
        self.nombre_palet = np.zeros(3, dtype=int)
        self.situations = []

    def __str__(self):
        return str(self.nombre_palet)

    def get_situation(self):
        return ''.join(map(str, self.pic.flatten()))

    def pic_vide(self, indice_pic):
        return self.nombre_palet[indice_pic] == 0

    def get_pic_top_index(self, indice_pic):
        # Returns -1 if the peg is empty, otherwise returns the index of the top disk
        return self.nombre_palet[indice_pic] - 1 if self.nombre_palet[indice_pic] > 0 else -1

    def get_pic_top(self, indice_pic):
        if self.nombre_palet[indice_pic] == 0:
            return None
        return self.pic[indice_pic][self.get_pic_top_index(indice_pic)]

    def regle_jeu(self, indice_pic1, indice_pic2):
        if self.pic_vide(indice_pic1):
            return False
        if self.pic_vide(indice_pic2):
            return True
        return self.get_pic_top(indice_pic1) < self.get_pic_top(indice_pic2)

    def deplacer(self, indice_pic1, indice_pic2):
        palet = self.get_pic_top(indice_pic1)
        if palet is None:
            return  # If there's no disk to move, exit the method

        # Remove the disk from the source peg
        self.pic[indice_pic1][self.get_pic_top_index(indice_pic1)] = 0
        self.nombre_palet[indice_pic1] -= 1

        # Determine the destination index on the destination peg
        destination_index = 0 if self.pic_vide(indice_pic2) else self.get_pic_top_index(indice_pic2) + 1

        # Ensure destination_index is within the valid range
        if 0 <= destination_index < len(self.pic[indice_pic2]):
            self.pic[indice_pic2][destination_index] = palet
            self.nombre_palet[indice_pic2] += 1

    def effectue_deplacement(self, indice_pic1, indice_pic2):
        if not self.pic_vide(indice_pic1) and self.regle_jeu(indice_pic1, indice_pic2):
            new_jeu = copy.deepcopy(self)
            new_jeu.deplacer(indice_pic1, indice_pic2)
            sit = new_jeu.get_situation()
            if sit not in self.situations:
                self.deplacer(indice_pic1, indice_pic2)
                self.situations.append(sit)
                return True
        return False

    def nombre_situation(self):
        situations = self.get_situation()
        nombre = 0
        for i, val in enumerate(situations):
            if val != '0':
                nombre += 2 ** i
        return nombre


def situation_non_vue(indice_pic1, indice_pic2, jeu, situation_etudiee):
    new_jeu = copy.deepcopy(jeu)
    new_jeu.deplacer(indice_pic1, indice_pic2)
    situation = new_jeu.nombre_situation()
    return situation not in situation_etudiee
