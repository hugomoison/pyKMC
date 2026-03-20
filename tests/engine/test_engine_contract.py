from pykmc import Engine, EngineExtension  # adapter selon le chemin réel
import numpy as np
import pytest

class EngineContractTests:
    """
    Suite de tests définissant le contrat que toute implémentation
    d'Engine doit respecter. Ne pas instancier directement.

    Les sous-classes doivent implémenter `make_engine()`, qui retourne
    une instance configurée mais non démarrée de l'engine.
    """

    def make_engine(self) -> Engine:
        raise NotImplementedError
    
    @property
    def is_rank0(self) -> bool:
        """True si le rank courant doit valider les assertions."""
        return True

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
        if self.is_rank0:
            np.testing.assert_allclose(result, positions, atol=1e-10)
        engine.close()

    def test_get_potential_energy(self) : 
        """Test get potential energy."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        pe = engine.get_potential_energy()
        if self.is_rank0:
            assert isinstance(pe, float)
        engine.close()


    def test_get_total_energy(self) : 
        """Test get total energy."""
        engine = self.make_engine() 
        engine.start() 
        self.initialize(engine)
        tot_e = engine.get_total_energy()
        if self.is_rank0:
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
        if self.is_rank0:
            assert min_positions.shape == (self.system.n_atoms, 3)
            assert tot_e2 < tot_e1
        engine.close()

    #Extension part 
    def make_test_extension(self, engine) -> EngineExtension:
        """Return a concrete extension compatible with this engine.
        Must be overridden in each concrete test class.
        """
        raise NotImplementedError
    
    def make_conflicting_extension(self, engine) -> EngineExtension:
        """Return an extension that conflicts with make_test_extension().
        Must be overridden in each concrete test class.
        """
        raise NotImplementedError

    def test_extension_registers_and_delegates(self):
        """A registered extension exposes its public methods through the engine."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        ext = self.make_test_extension(engine)
        public = [m for m in dir(ext) if not m.startswith("_")]
        assert all(hasattr(engine, m) for m in public)
        engine.close()

    def test_extension_conflict_raises(self):
        """Registering two extensions with a method of the same name raises ValueError."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        self.make_test_extension(engine)
        with pytest.raises(ValueError):
            self.make_conflicting_extension(engine)
        engine.close()
