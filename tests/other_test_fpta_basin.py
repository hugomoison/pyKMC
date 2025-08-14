# Tester que la fpta méthode fonctionne comme il faut

import pandas as pd
import numpy as np
import math
from scipy.optimize import bisect
from scipy.optimize import fsolve
import copy
import matplotlib.pyplot as plt
from mpmath import mp


class BasinTestFPTA:
    
    def __init__(self):
        self.connexion_table = None
        self.temp = 300  # Kelvin
        self.transient_states = []
        self.absorbing_states = []
        self.M_matrix = None
    
    def initial_data(self):
        """
        Initialise des données de test pour la connexion_table
        """
        data = {
            'state': [0, 0, 1, 1, 3],
            'state_connexion': [1, 2, 0, 3, 1],
            'event_connexion': [22, 33, 44, 55, 66],
            'central_atom': [4528, 4529, 4543, 4544, 4849],
            'sym': [0, 0, 0, 0, 0],
            'transient': [True, False, True, True, True],
            'energy_barrier': [0.132, 2.682, 0.148, 0.247, 0.179]
        }
        
        self.connexion_table = pd.DataFrame(data)
        print("connexion_table")
        print(self.connexion_table)



    def compute_jump_rate(self, energy_barrier):
        #Calcule le taux de saut à partir de la barrière énergétique
        jump_frequency = 10 ** 13  # Hz
        boltzmann_constant = 1.380649e-23  # J/K
        # Convertir eV en J (supposant que energy_barrier est en eV)
        energy_barrier_j = energy_barrier * 1.602176634e-19                                               # ***
        jump_rate = jump_frequency * np.exp(-energy_barrier_j / (boltzmann_constant * self.temp))
        return jump_rate
    


    def identify_states(self):
        
        # Tous les états uniques
        all_states = set(self.connexion_table['state'].tolist() + 
                        self.connexion_table['state_connexion'].tolist())
        
        # États transitoires
        transient_from_table = set()
        for _, row in self.connexion_table.iterrows():
            if row['transient'] == True:
                transient_from_table.add(row['state'])
        
        # États absorbants
        self.transient_states = sorted(list(transient_from_table))
        self.absorbing_states = sorted(list(all_states - transient_from_table))
        
        print(f"États identifiés:")
        print(f"  - Transitoires: {self.transient_states}")
        print(f"  - Absorbants: {self.absorbing_states}")



    def build_rate_matrix(self):

        # Calculer les taux de saut
        self.connexion_table['jump_rate'] = self.connexion_table['energy_barrier'].apply(self.compute_jump_rate)
        
        print("\nTable avec taux de saut:")
        print(self.connexion_table[['state', 'state_connexion', 'energy_barrier', 'jump_rate']])

        # Construire la matrice M
        all_states = sorted(self.transient_states + self.absorbing_states)
        n_total = len(all_states)
        
        # Mapping état -> index
        state_to_idx = {state: i for i, state in enumerate(all_states)}
        
        # Initialiser la matrice
        M = np.zeros((n_total, n_total))

        # Remplir la matrice
        for _, row in self.connexion_table.iterrows():
            state_i = row['state']
            state_j = row['state_connexion']
            rate = row['jump_rate']
            
            if state_i in state_to_idx and state_j in state_to_idx:
                i = state_to_idx[state_i]
                j = state_to_idx[state_j]

                # Mij = -Rj->i si i ≠ j
                if i != j:
                    M[j, i] = -rate

        # Diagonale : Mii = ∑k Ri->k
        for i in range(n_total):
            outgoing_rates = 0
            state_i = all_states[i]
            
            for _, row in self.connexion_table.iterrows():
                if row['state'] == state_i:
                    outgoing_rates += row['jump_rate']
            
            M[i, i] = outgoing_rates
        
        self.M_matrix = M
        print(f"\nMatrice M construite ({n_total}x{n_total}):")
        print("États:", all_states)
        print("Matrice M:")
        print(M)
        
        return M
    


    def build_reduced_matrix(self):
        # Construit la matrice réduite (états absorbants combinés)
        n_transient = len(self.transient_states)
        self.M_reduced = np.zeros((n_transient + 1, n_transient + 1))
        
        # Copier la partie transitoire
        self.M_reduced[:n_transient, :n_transient] = self.M_matrix[:n_transient, :n_transient]
        
        # Sommer les taux vers les états absorbants
        for i in range(n_transient):
            state_i = self.transient_states[i]
            rate_to_absorbing = 0
            
            for _, row in self.connexion_table.iterrows():
                if row['state'] == state_i and row['state_connexion'] in self.absorbing_states:
                    rate_to_absorbing += row['jump_rate']

            # Taux vers l'état absorbant combiné
            self.M_reduced[n_transient, i] = -rate_to_absorbing
            
            # Recalculer la diagonale
            total_rate = rate_to_absorbing
            
            for j in range(n_transient):
                if i != j:
                    for _, row in self.connexion_table.iterrows():
                        if (row['state'] == state_i and 
                            row['state_connexion'] == self.transient_states[j]):
                            total_rate += row['jump_rate']
                            break
            
            self.M_reduced[i, i] = total_rate

        print(f"\nMatrice réduite ({n_transient + 1}x{n_transient + 1}):")
        print(self.M_reduced)
        
        return self.M_reduced


    def get_eigvals(self, t) :
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M_reduced)
        print(f"\n Valeurs propres:")
        print(self.eigenvalues)
        print(f"\n Vecteurs propres:")
        print(self.eigenvectors)


        # Probabilités individuelles des états absorbants
        n_total = len(self.transient_states) + len(self.absorbing_states)
        initial_vector_full = np.zeros(n_total)
        initial_vector_full[0] = 1.0

        # Transformation dans la base des vecteurs propres
        # Résoudre P * c = initial_vector_full
        self.c_coefficients = np.linalg.solve(self.eigenvectors, initial_vector_full)
        print("\nCoefficients dans la base des vecteurs propres:")
        print("c =", self.c_coefficients)
            
    def check_random_s(self, t) :

        sum = 0

        for i in range(len(self.c_coefficients)) :
            sum += self.c_coefficients[i] * mp.exp(-self.eigenvalues[i] * t ) * self.eigenvectors[i][-1]
            #print(mp.exp(-self.eigenvalues[i] * t ))
        return sum
    

    def f(self, t):
        random_s = 0.9507
        return self.check_random_s(t) - random_s          




    


def test_fpta_complete():
    """Test complet FPTA"""
    print("=== TEST FPTA ===")
    
    # Créer l'instance
    basin = BasinTestFPTA()
    
    # Charger les données
    basin.initial_data()
    
    # Identifier les états
    basin.identify_states()

    # Construire les matrices
    basin.build_rate_matrix()

    basin.build_reduced_matrix()

    basin.get_eigvals(t=None)

    #root_solce = fsolve(basin.f, 0.0001)
    t_tab = np.linspace(0,0.0001,300)

    y_tab = np.zeros(len(t_tab))

    for i,j in zip(t_tab,range(len(t_tab))):
        y_tab[j] = basin.f(i)
    
    data = np.array([t_tab,y_tab])
    plt.plot(t_tab,y_tab,'.')
    plt.savefig('plot.png')

    

    #t_min = 0.0
    #t_max = 100.0
    #t_exit = bisect(basin.f, t_min, t_max)
    #print(t_exit)



if __name__ == "__main__":
    # Exécuter le test
    success = test_fpta_complete()     