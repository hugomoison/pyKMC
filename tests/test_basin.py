import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import os
import copy
from pykmc import Basin, System, Config



class TestBasin:
    
    @pytest.fixture
    def mock_reference_table(self):
        """Fixture pour créer une table de référence mock"""
        mock_table = Mock()
        # Créer un DataFrame mock avec les colonnes nécessaires
        mock_table.table = pd.DataFrame({
            'event_id': ['env1', 'env2', 'env3'],
            'id_final': ['env2', 'env1', 'env4'],
            'energy_barrier': [0.1, 0.05, 0.3],
            'num_reference_event': [0, 1, 2],
            'final_positions': [np.array([[0, 0, 0]]), np.array([[1, 1, 1]]), np.array([[2, 2, 2]])],
            'sym_matrix': [
                [np.eye(3)],  # Une seule matrice identité pour la première symétrie
                [np.eye(3), np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])],  # Deux matrices pour la deuxième
                [np.eye(3)]
            ]
        })
        return mock_table
    


    @pytest.fixture
    def mock_system(self):
        """Fixture pour créer un système mock"""
        system = Mock()
        system.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        system.update_positions = Mock()
        return system
    


    @pytest.fixture
    def mock_config(self):
        """Fixture pour créer une configuration mock"""
        config = Mock()
        config.atomicenvironment.rnei = 3.0
        config.atomicenvironment.rcut = 5.0
        config.atomicenvironment.style = 'cna/graph'
        config.atomicenvironment.neighbors_add = 0  
        config.basin.energy_thr = 0.2
        return config
    


    @pytest.fixture
    def mock_engine(self):
        """Fixture pour créer un engine mock"""
        engine = Mock()
        engine.minimize = Mock(return_value=(np.array([[0, 0, 0], [1, 0, 0]]), -10.5))
        return engine
    


    @pytest.fixture
    def basin_instance(self, mock_reference_table):
        """Fixture pour créer une instance de Basin"""
        return Basin(mock_reference_table)
    


    def test_basin_basic_functionality(self, basin_instance):
        """Test de base sans patches pour vérifier que Basin fonctionne"""
        # Test d'initialisation des attributs de base
        assert basin_instance.connexion_table is None
        assert basin_instance.states == []
        assert basin_instance.explored_states == []
        assert basin_instance.reference_table is not None
        


    def test_detectin_simple(self, basin_instance):
        """Test simple de detectin sans dépendances externes"""
        # Créer une série mock pour l'événement actif
        selected_active_event = pd.Series({
            'energy_barrier': 0.3,  # Énergie élevée
            'num_reference_event': 0
        })
        
        # Test avec seuil bas -> pas de bassin détecté
        result = basin_instance.detectin(selected_active_event, 0.2)
        assert result == False
    


    def test_detectin_basin_detected(self, basin_instance):
        """Test la détection de bassin quand les conditions sont remplies"""
        # Créer une série mock pour l'événement actif
        selected_active_event = pd.Series({
            'energy_barrier': 0.1,
            'num_reference_event': 0
        })
        
        # Mock de la table de référence pour le test
        basin_instance.reference_table.table = pd.DataFrame({
            'event_id': ['env1', 'env2'],
            'id_final': ['env2', 'env1'],
            'energy_barrier': [0.1, 0.05]
        })
        
        result = basin_instance.detectin(selected_active_event, 0.2, index_event=0)
        assert result == True
    


    def test_detectin_basin_not_detected_high_forward_energy(self, basin_instance):
        """Test quand l'énergie forward est trop haute"""
        selected_active_event = pd.Series({
            'energy_barrier': 0.3,
            'num_reference_event': 0
        })
        
        result = basin_instance.detectin(selected_active_event, 0.2)
        assert result == False
    


    def test_detectin_basin_not_detected_high_backward_energy(self, basin_instance):
        """Test quand l'énergie backward est trop haute"""
        selected_active_event = pd.Series({
            'energy_barrier': 0.1,
            'num_reference_event': 0
        })
        
        # Mock avec énergie backward élevée
        basin_instance.reference_table.table = pd.DataFrame({
            'event_id': ['env1', 'env2'],
            'id_final': ['env2', 'env1'],
            'energy_barrier': [0.1, 0.3]  # Énergie backward élevée
        })
        
        result = basin_instance.detectin(selected_active_event, 0.2, index_event=0)
        assert result == False



    @patch('pykmc.basin.NeighborsList')
    @patch('pykmc.basin.AtomicEnvironment')
    def test_initialize(self, mock_atomic_env, mock_neighbors, mock_system, mock_config, mock_reference_table):
        """Test de la méthode initialize avec mocks corrects"""
        
        # Configuration des mocks
        mock_neighbors_instance = Mock()
        mock_neighbors_instance.neighbors_list = {
            'rnei': 'dummy_rnei',
            'rcut': 'dummy_rcut'
        }
        mock_neighbors.return_value = mock_neighbors_instance

        mock_atomic_env_instance = Mock()
        mock_atomic_env_instance.atomic_environment_list = ['env1', 'env2', 'env3']
        mock_atomic_env.return_value = mock_atomic_env_instance

        # Créer une instance de Basin
        basin_instance = Basin(mock_reference_table)
        
        # Appeler initialize
        basin_instance.initialize(mock_system, mock_config)

        # Vérifications
        assert basin_instance.states == [0,1,2,3]
        assert len(basin_instance.state_system) == 1
        assert basin_instance.state_system[0] == mock_system
        assert basin_instance.states_to_explore == [1,2]
        assert basin_instance.visited_states == [0]
        
        # Vérifier que les mocks ont été appelés
        mock_neighbors.assert_called_once()
        mock_atomic_env.assert_called_once()
        
        # Vérifier que la connexion_table a été créée
        assert basin_instance.connexion_table is not None
        expected_columns = ['state', 'state_connexion', 'event_connexion', 'central_atom', 'sym', 'transient']
        assert list(basin_instance.connexion_table.columns) == expected_columns



    def test_get_atomic_environment(self, basin_instance):
        """Test la récupération de l'environnement atomique"""
        # Setup
        basin_instance.state_environment = [['env1', 'env2'], ['env3', 'env4']]
        
        # Test
        result = basin_instance.get_atomic_environment(0)
        assert result == ['env1', 'env2']
        
        result = basin_instance.get_atomic_environment(1)
        assert result == ['env3', 'env4']
    


    def test_get_applicable_generic_event(self, basin_instance, mock_reference_table):
        """Test la récupération des événements génériques applicables"""
        # Setup
        basin_instance.state_environment = [['env1', 'env3']]
        basin_instance.reference_table = mock_reference_table
        
        # Test
        result = basin_instance.get_applicable_generic_event(0)
        
        # Vérifier que seuls les événements avec event_id dans ['env1', 'env3'] sont retournés
        expected_events = mock_reference_table.table[
            mock_reference_table.table['event_id'].isin(['env1', 'env3'])
        ]
        pd.testing.assert_frame_equal(result, expected_events)
    


    def test_are_structures_equivalent_identical(self, basin_instance, mock_system):
        """Test la comparaison de structures identiques"""
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        pos2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        result = basin_instance.are_structures_equivalent(pos1, pos2, mock_system)
        assert result == True
    

    
    def test_are_structures_equivalent_different(self, basin_instance, mock_system):
        """Test la comparaison de structures différentes"""
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        pos2 = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]])  # Position différente
        
        result = basin_instance.are_structures_equivalent(pos1, pos2, mock_system)
        assert result == False
    


    def test_are_structures_equivalent_different_lengths(self, basin_instance, mock_system):
        """Test la comparaison de structures avec nombres d'atomes différents"""
        pos1 = np.array([[0, 0, 0], [1, 0, 0]])
        pos2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        result = basin_instance.are_structures_equivalent(pos1, pos2, mock_system)
        assert result == False
    


    def test_find_existing_state_found(self, mock_reference_table, mock_config):
        """Test find_existing_state quand un état équivalent existe"""
        
        # Créer une instance de Basin avec reference_table
        basin_instance = Basin(mock_reference_table)
        
        # Créer des systèmes mock avec des positions
        existing_system1 = Mock()
        existing_system1.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        existing_system1.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        existing_system2 = Mock()
        existing_system2.positions = np.array([[2, 2, 2], [3, 2, 2], [2, 3, 2]])
        existing_system2.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        # Système à tester (similaire au premier)
        new_system = Mock()
        new_system.positions = np.array([[0.01, 0.01, 0.01], [1.01, 0.01, 0.01], [0.01, 1.01, 0.01]])  # Légèrement différent
        new_system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        # Ajouter les systèmes existants à state_system
        basin_instance.state_system = [existing_system1, existing_system2]
        
        # Test : devrait trouver une correspondance avec le premier système (index 0)
        result = basin_instance.find_existing_state(new_system, tolerance=0.1)
        assert result == 0
        


    def test_find_existing_state_not_found(self, mock_reference_table):
        """Test find_existing_state quand aucun état équivalent n'existe"""
        
        # Créer une instance de Basin
        basin_instance = Basin(mock_reference_table)
        
        # Créer des systèmes mock avec des positions très différentes
        existing_system = Mock()
        existing_system.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        existing_system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        new_system = Mock()
        new_system.positions = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]])  # Très différent
        new_system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        # Ajouter le système existant
        basin_instance.state_system = [existing_system]
        
        # Test : ne devrait pas trouver de correspondance
        result = basin_instance.find_existing_state(new_system, tolerance=0.1)
        assert result is None



    def test_are_structures_equivalent_true(self, mock_reference_table):
        """Test are_structures_equivalent pour des structures équivalentes"""
        
        basin_instance = Basin(mock_reference_table)
        
        # Positions similaires (dans la tolérance)
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        pos2 = np.array([[0.05, 0.05, 0.05], [1.05, 0.05, 0.05], [0.05, 1.05, 0.05]])
        
        system = Mock()
        system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        result = basin_instance.are_structures_equivalent(pos1, pos2, system, tol=0.1)
        assert result == True
        


    def test_are_structures_equivalent_false(self, mock_reference_table):
        """Test are_structures_equivalent pour des structures différentes"""
        
        basin_instance = Basin(mock_reference_table)
        
        # Positions très différentes
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        pos2 = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]])
        
        system = Mock()
        system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        
        result = basin_instance.are_structures_equivalent(pos1, pos2, system, tol=0.1)
        assert result == False    
    


    @patch('pykmc.basin.NeighborsList')
    @patch('pykmc.basin.AtomicEnvironment')
    def test_update_state_environment(self, mock_atomic_env, mock_neighbors, mock_reference_table, mock_system, mock_config):
        """Test de la méthode update_state_environment"""
        
        # Configuration des mocks - utiliser les valeurs de mock_config
        mock_neighbors_instance = Mock()
        mock_neighbors_instance.neighbors_list = {'rnei': 3.0, 'rcut': 5.0}  # Correspond à mock_config
        mock_neighbors.return_value = mock_neighbors_instance

        mock_atomic_env_instance = Mock()
        mock_atomic_env_instance.atomic_environment_list = ['env1', 'env2', 'env3']
        mock_atomic_env.return_value = mock_atomic_env_instance

        # Créer une instance de Basin
        basin_instance = Basin(mock_reference_table)
        
        # Initialiser les listes vides
        basin_instance.state_environment = []
        basin_instance.state_neighbors_list = []
        
        # Appeler update_state_environment
        result = basin_instance.update_state_environment(mock_system, mock_config, state=0)
        
        # Vérifications
        assert len(basin_instance.state_environment) == 1
        assert len(basin_instance.state_neighbors_list) == 1
        assert basin_instance.state_environment[0] == ['env1', 'env2', 'env3']
        assert basin_instance.state_neighbors_list[0] == mock_neighbors_instance
        assert result == mock_neighbors_instance
        
        # Vérifier que les constructeurs ont été appelés avec les bons arguments
        # Utiliser les valeurs de mock_config (rnei=3.0, rcut=5.0)
        mock_neighbors.assert_called_once_with(mock_system, 3.0, 5.0)
        mock_atomic_env.assert_called_once_with(
            'cna/graph',  # style
            3.0,          # neighbors_list['rnei'] 
            5.0,          # neighbors_list['rcut']
            0             # neighbors_add
        )



    @patch('pykmc.basin.NeighborsList')
    @patch('pykmc.basin.AtomicEnvironment')
    def test_update_state_environment_with_state_index(self, mock_atomic_env, mock_neighbors, mock_reference_table, mock_system, mock_config):
        """Test update_state_environment avec state_index spécifié"""
        
        # Configuration des mocks
        mock_neighbors_instance = Mock()
        mock_neighbors_instance.neighbors_list = {'rnei': 3.0, 'rcut': 3.5}
        mock_neighbors.return_value = mock_neighbors_instance

        mock_atomic_env_instance = Mock()
        mock_atomic_env_instance.atomic_environment_list = ['env4', 'env5']
        mock_atomic_env.return_value = mock_atomic_env_instance

        # Créer une instance de Basin avec des données existantes
        basin_instance = Basin(mock_reference_table)
        basin_instance.state_environment = [['old_env1'], ['old_env2']]
        basin_instance.state_neighbors_list = [Mock(), Mock()]
        
        # Appeler update_state_environment avec state_index
        result = basin_instance.update_state_environment(mock_system, mock_config, state=1, state_index=1)
        
        # Vérifications - l'index 1 devrait être mis à jour
        assert len(basin_instance.state_environment) == 2
        assert basin_instance.state_environment[0] == ['old_env1']  
        assert basin_instance.state_environment[1] == ['env4', 'env5']  
        assert basin_instance.state_neighbors_list[1] == mock_neighbors_instance
        assert result == mock_neighbors_instance



    # Test d'intégration simplifié
    @patch('pykmc.basin.NeighborsList')
    @patch('pykmc.basin.AtomicEnvironment')
    def test_execute_integration(self, MockAtomicEnvironment, MockNeighborsList):
        mock_system = Mock()
        mock_system.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        mock_config = Mock()

        mock_neighbors = Mock()
        mock_neighbors_instance = Mock()
        mock_neighbors_instance.neighbors_list = {'rnei': 3.0, 'rcut': 3.5}
        mock_neighbors.return_value = mock_neighbors_instance

        mock_config.atomicenvironment = Mock()
        mock_config.atomicenvironment.style = 'cna/graph'
        mock_config.atomicenvironment.rnei = 3.0
        mock_config.atomicenvironment.rcut = 3.5
        mock_config.atomicenvironment.neighbors_add = 0

       
        mock_atomic_env_instance = Mock()
        mock_atomic_env_instance.atomic_environment_list = [b'env1']
        MockAtomicEnvironment.return_value = mock_atomic_env_instance


        
        df = pd.DataFrame({'event_id': ['env1', 'env2', 'env3']})
        reference_table = Mock()
        reference_table.table = df
        basin_instance = Basin(reference_table)
        basin_instance.initialize(mock_system, mock_config)

        assert len(basin_instance.state_system) >= 1
        assert len(basin_instance.state_environment) >= 1



# Tests paramétrés pour différents seuils d'énergie
class TestBasinParametrized:
    
    @pytest.mark.parametrize("energy_barrier,threshold,expected", [
        (0.1, 0.2, True),   # Énergie faible, seuil élevé -> bassin détecté
        (0.3, 0.2, False),  # Énergie élevée, seuil faible -> pas de bassin
        (0.15, 0.15, False), # Énergie égale au seuil -> pas de bassin (strictement supérieur)
        (0.05, 0.1, True),  # Énergie très faible -> bassin détecté
    ])
    def test_detectin_various_thresholds(self, energy_barrier, threshold, expected):
        """Test de détection avec différents seuils d'énergie"""
        mock_ref_table = Mock()
        basin = Basin(mock_ref_table)
        
        # Créer un événement mock
        selected_event = pd.Series({
            'energy_barrier': energy_barrier,
            'num_reference_event': 0
        })
        
        # Mock de la table avec événement réciproque
        basin.reference_table.table = pd.DataFrame({
            'event_id': ['env1', 'env2'],
            'id_final': ['env2', 'env1'],
            'energy_barrier': [energy_barrier, 0.05]  # Énergie backward faible
        })
        
        result = basin.detectin(selected_event, threshold, index_event=0)
        assert result == expected

