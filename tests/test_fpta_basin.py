# Tester que la fpta méthode fonctionne comme il faut

import pandas as pd
import numpy as np
from scipy.linalg import expm
import copy
from scipy.optimize import bisect
import matplotlib.pyplot as plt 


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
        self.all_states = sorted(self.transient_states + self.absorbing_states)
        n_total = len(self.all_states)
        
        # Mapping état -> index
        state_to_idx = {state: i for i, state in enumerate(self.transient_states)} | {state: i + len(self.transient_states) for i, state in enumerate(self.absorbing_states)}
        print(state_to_idx)

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
            state_i = state_to_idx[i]
            
            
            for _, row in self.connexion_table.iterrows():
                if row['state_connexion'] == state_i:
                    
                    rate = row['jump_rate']
                    outgoing_rates += rate
                    
            
            M[i, i] = outgoing_rates
        
        self.M_matrix = M
        print(f"\nMatrice M construite ({n_total}x{n_total}):")
        print("États:", self.all_states)
        print("Matrice M:")
        print(M)
        
        return M



    def build_reduced_matrix(self):
        # Construit la matrice réduite (états absorbants combinés)
        n_transient = len(self.transient_states)
        M_reduced = np.zeros((n_transient + 1, n_transient + 1))
        
        # Copier la partie transitoire
        M_reduced[:n_transient, :n_transient] = self.M_matrix[:n_transient, :n_transient]
        

        for i in range(n_transient, len(self.all_states)) :
            for j in range(n_transient):
                
                M_reduced[i, j] = self.M_matrix[i:, j].sum()

        
        M_reduced[-1, -1] = abs(M_reduced[-1, :].sum())

        print(f"\nMatrice réduite ({n_transient + 1}x{n_transient + 1}):")
        print(M_reduced)
        
        return M_reduced



    def run_fpta_step(self, current_state):
        # Exécute une étape FPTA
        self.initial_state_idx = self.transient_states.index(current_state)
        self.n_transient = len(self.transient_states)
        
        # Construire la matrice réduite
        self.M_reduced = self.build_reduced_matrix()

        # Vecteur initial
        self.initial_vector = np.zeros(self.n_transient + 1)
        self.initial_vector[self.initial_state_idx] = 1.0
        
        # Nombre aléatoire
        self.random_r = np.random.random()
        print(f"\nNombre aléatoire r = {self.random_r:.4f}")
        
        # Temps initial
        self.initial_time = 1.0 / np.max(np.diag(self.M_matrix))
        
        # Méthode de bisection
        t_min, t_max = 0.0, self.initial_time
        tolerance = 1e-5

        # Étendre la recherche si nécessaire
        iteration = 0
        while True:
            prob_vector = expm(-t_max * self.M_reduced) @ self.initial_vector
            prob_absorbing = prob_vector[-1]
            print(f"  Extension {iteration}: t_max={t_max:.6f}, P_abs={prob_absorbing:.4f}")
            
            if prob_absorbing > self.random_r:
                break
            t_max *= 2
            iteration += 1
            
            if iteration > 20:  # Éviter les boucles infinies
                print("  Arrêt de l'extension après 20 itérations")
                break

        self.bisection()    




    def f(self, time):
        prob_vector = expm(-time * self.M_reduced) @ self.initial_vector
        prob_absorbing = prob_vector[-1]
        return prob_absorbing - self.random_r


    def bisection(self):
        
        t_tab = np.linspace(0, 1e-12, 20)
        y_tab = np.array([self.f(t) for t in t_tab])  # compute f for each scalar t
        plt.plot(t_tab, y_tab, '.')
        plt.show
        



        # Bisection
        exit_time = bisect(self.f, 0.0, 1000000000, xtol=1e-5, maxiter=100)
        print(f"  Temps de sortie: {exit_time:.6f}")

        # Probabilités individuelles des états absorbants
        n_total = len(self.transient_states) + len(self.absorbing_states)
        initial_vector_full = np.zeros(n_total)
        initial_vector_full[self.initial_state_idx] = 1.0
        
        prob_vector_full = expm(-exit_time * self.M_matrix) @ initial_vector_full
        absorbing_probs = prob_vector_full[self.n_transient:]
        
        # Normalisation
        total_absorbing_prob = np.sum(absorbing_probs)
        if total_absorbing_prob > 0:
            absorbing_probs = absorbing_probs / total_absorbing_prob
        
        print(f"  Probabilités absorbantes: {absorbing_probs}")

        # Sélection de l'état
        random_s = np.random.random()
        print(f"  Nombre aléatoire s = {random_s:.4f}")
        
        cumulative_prob = 0
        selected_state = None
        for i, prob in enumerate(absorbing_probs):
            cumulative_prob += prob
            if random_s <= cumulative_prob:
                selected_state = self.absorbing_states[i]
                break
        
        if selected_state is None:
            selected_state = self.absorbing_states[-1]
        
        print(f"  État sélectionné: {selected_state}")
        
        return selected_state, exit_time



    def run_fpta(self):
        """Exécute plusieurs étapes FPTA"""
        total_exit_time = 0
        simulation_results = []
        
        next_state, exit_time = self.run_fpta_step(0)  # Toujours partir de l'état 0
        total_exit_time += exit_time
        simulation_results.append(next_state)

        print(f"\n=== Résultats finaux ===")
        print(f"États visités: {simulation_results}")
        print(f"Temps total: {total_exit_time:.6f}")
        
        # Statistiques
        from collections import Counter
        stats = Counter(simulation_results)
        print("\nStatistiques:")
        for state, count in stats.items():
            print(f"  État {state}")
        
        return simulation_results, total_exit_time



def test_fpta_complete():
    """Test complet FPTA"""
    print("=== TEST FPTA ===")
    
    # Créer l'instance
    basin = BasinTestFPTA()
    
    # Charger les données
    basin.initial_data()
    
    # Identifier les états
    basin.identify_states()
    
    # Vérifier la configuration
    if len(basin.transient_states) == 0:
        print("❌ Aucun état transitoire trouvé!")
        return False
        
    if len(basin.absorbing_states) == 0:
        print("❌ Aucun état absorbant trouvé!")
        return False
    
    # Construire les matrices
    basin.build_rate_matrix()
    
    # Exécuter FPTA
    try:
        results, time_total = basin.run_fpta()
        print(" ✅ Test FPTA réussi!")
        return True
    except Exception as e:
        print(f"❌ Erreur FPTA: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Fixer la graine pour la reproductibilité
    np.random.seed(42)
    
    # Exécuter le test
    success = test_fpta_complete()
      