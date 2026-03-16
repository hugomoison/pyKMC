from pykmc import Engine  # adapter selon le chemin réel
import numpy as np

class EngineContractTests:
    """
    Suite de tests définissant le contrat que toute implémentation
    d'Engine doit respecter. Ne pas instancier directement.

    Les sous-classes doivent implémenter `make_engine()`, qui retourne
    une instance configurée mais non démarrée de l'engine.
    """

    def make_engine(self) -> Engine:
        raise NotImplementedError

    # ── start / close ─────────────────────────────────────────────────────────

    def test_start_does_not_raise(self):
        """Test open and close engine."""
        engine = self.make_engine()
        engine.start()
        engine.close()


    def initialize(self, engine) : 
        """Initialization parameters and system convenience method."""
        engine.initialize_parameters()
        engine.initialize_system(
            types=self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        engine.initialize_potential()
    def test_initialize(self)  : 
        """Test initialization parameters and system."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        engine.close()


    def test_set_get_positions(self):
        """Test set_positions() and get_positions() consistency."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        positions = self.system.positions 
        positions[0,0] += 0.2
        engine.set_positions(positions)
        result = engine.get_positions()
        np.testing.assert_allclose(result, positions, atol=1e-10)
        engine.close()

    def test_get_potential_energy(self) : 
        """Test get potential energy."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        pe = engine.get_potential_energy()
        assert isinstance(pe, float)
        engine.close()


    def test_get_total_energy(self) : 
        """Test get total energy."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        tot_e = engine.get_total_energy()
        assert isinstance(tot_e, float)
        engine.close()

    def minimize_with_results(self) : 
        """Test minimization and lower energy after."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        positions = self.system.positions
        #perturbations
        rng = np.random.default_rng(seed=42)
        positions = self.system.positions + rng.uniform(-0.1, 0.1, size=self.system.positions.shape)
        tot_e1 = engine.get_potential_energy()
        #minimization 
        min_positions, tot_e2 =  engine.minimize_with_results(positions=positions)
        assert min_positions.shape == (self.system.n_atoms, 3)
        assert tot_e2 < tot_e1
        engine.close()