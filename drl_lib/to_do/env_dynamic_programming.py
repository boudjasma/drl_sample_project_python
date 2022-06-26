import numpy as np
from tkinter import *

class Env_on_line:
    name = 'on line'
    def __init__(self,nb_cells): # constructeur on line world
        self.nb_cells = nb_cells  # nbre de cellules
        self.S = [i for i in range(nb_cells)] # ensemble des etats
        self.A = [0, 1]  # ensemble des actions, 0: à gauche, 1: à droite
        self.R = [-1.0, 0.0, 1.0] # les reward
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        for s in self.S:
            if not self.is_state_terminal(s):
                if s == 1:
                    self.p[s, 0, s - 1, 0] = 1.0
                else:
                    self.p[s, 0, s - 1, 1] = 1.0

                if s == nb_cells - 2:
                    self.p[s, 1, s + 1, 2] = 1.0
                else:
                    self.p[s, 1, s + 1, 1] = 1.0

    def states(self):
        return self.S

    def actions(self):
        return self.A

    def rewards(self):
        return self.R

    def transition_probability(self, s: int, a: int, s_p: int, r: float):
        return self.p[s, a, s_p, r]

    def is_state_terminal(self, s: int) -> bool:
        return s == self.nb_cells - 1 or s == 0

    def view_state(self, s: int):
        return "state exist" if s in self.S else "state does not exist"


    def display_on_line_world(self, V, name_function):
        V = [V[i][1] for i in range(len(V))]

        fenetre = Tk()
        fenetre.title(name_function)
        fenetre.geometry('800x100')

        label = Label(fenetre, text=name_function)
        label.pack()

        p = PanedWindow(fenetre, orient=HORIZONTAL)
        p.pack(side=TOP, padx=5, pady=5)
        for i in range(len(V)):
            p.add(Label(p, text=str(V[i]), background='#49A', anchor=CENTER, padx=10, pady=10))
        p.pack()
        fenetre.mainloop()


class Env_on_grid:
    name = 'on grid'

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.nb_cells = self.height * self.width
        self.S = [i for i in range(self.nb_cells)]
        self.A = [0, 1, 2, 3]  # 0 : left, 1: right, 2:up, 3:down
        self.R = [-1.0, 0.0, 1.0]
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        # p [etat_precedent, action, etat_suivant, reward]
        for s in self.S:
            if not self.is_state_terminal(s):
                # Left and right
                if (s + 1) % self.width == 0:
                    self.p[s, 0, s - 1, 1] = 1.0  # possibilité d'aller vers la gauche (left : 0)
                elif s % self.width == 0:
                    self.p[s, 1, s + 1, 1] = 1.0  # possibilité d'aller vers la droite (right : 1)
                # Up and down
                elif (s >= self.width and s <= (self.nb_cells - self.width - 1)):
                    self.p[s, 3, s + self.width, 1] = 1.0  # possibilité d'aller vers le bas
                    self.p[s, 2, s - self.width, 1] = 1.0  # possibilité d'aller vers le haut

                elif s < self.width - 1 or s > self.nb_cells-self.width:
                    self.p[s, 0, s - 1, 1] = 1.0
                    self.p[s, 1, s + 1, 1] = 1.0

                    # Defeat
                if s == 2*self.width - 1:
                    self.p[s, 2, s - self.width, 0] = 1.0  # défaite vers le haut

                if s == (self.width - 2):
                    self.p[s, 1, s + 1, 0] = 1.0  # défaite à droite

                # Success
                if s == self.nb_cells - self.width - 1:
                    self.p[s, 3, s + self.width, 2] = 1.0  # success vers le bas

                if s == self.nb_cells - 2:
                    self.p[s, 1, s + 1, 2] = 1.0  # success à droite

    def states(self):
        return self.S

    def actions(self):
        return self.A

    def rewards(self):
        return self.R

    def transition_probability(self, s: int, a: int, s_p: int, r: float):
        return self.p[s, a, s_p, r]

    def is_state_terminal(self, s: int) -> bool:
        return s == self.nb_cells - 1 or s == self.width - 1

    def view_state(self, s: int):
        return s if s in self.S else "state does not exist"

    def display_on_grid_world(self, height, width, V, name_function):
        fenetre = Tk()
        fenetre.title(name_function)
        fenetre.geometry('750x250')
        for line in range(height):
            for col in range(width):
                Button(fenetre, text=str(V[line*width+col]), background='#7e6a3e', height=2, width=20).grid(row=line, column=col)

        fenetre.mainloop()

