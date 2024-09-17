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
        return np.all(self.pic[indice_pic] == 0)

    def get_pic_top_index(self, indice_pic):
        return self.nombre_palet[indice_pic] - 1

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
        self.pic[indice_pic1][self.get_pic_top_index(indice_pic1)] = 0
        self.nombre_palet[indice_pic1] -= 1
        self.pic[indice_pic2][self.get_pic_top_index(indice_pic2) + 1] = palet
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
