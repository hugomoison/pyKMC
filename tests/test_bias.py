"""Tests for DirectionBias, PointBias, and TopoBias."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
import importlib.util, sys, pathlib

def _load(rel):
    path = pathlib.Path(__file__).parent.parent / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

_bias = _load("pykmc/bias.py")
_algo = _load("pykmc/algorithms.py")
DirectionBias = _bias.DirectionBias
PointBias     = _bias.PointBias
TopoBias      = _bias.TopoBias
rejection_free = _algo.rejection_free


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_system(positions: np.ndarray):
    """Return a minimal System-like mock with the given positions array."""
    system = MagicMock()
    system.positions = np.asarray(positions, dtype=float)
    return system


def make_reference_table(move_atom_idx: int, idx_ref: int = 0):
    """Return a minimal ReferenceEventTable-like mock."""
    df = pd.DataFrame({
        "idx_ref": [idx_ref],
        "move_atom_idx": [move_atom_idx],
    })
    ref = MagicMock()
    ref.table = df
    return ref


def make_event(atom_index: int, final_positions: np.ndarray,
               num_reference_event: int = 0, k: float = 1.0) -> pd.Series:
    """Return an active-event row (pd.Series)."""
    return pd.Series({
        "atom_index": atom_index,
        "final_positions": np.asarray(final_positions, dtype=float),
        "num_reference_event": num_reference_event,
        "energy_barrier": 0.5,
        "k": k,
        "refined": "yes",
    })


def make_active_table(events: list[pd.Series]):
    """Return a minimal ActiveEventTable-like mock from a list of event Series."""
    df = pd.DataFrame(events).reset_index(drop=True)
    table = MagicMock()
    table.table = df
    return table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def system_origin():
    """Single atom sitting at the origin."""
    return make_system([[0.0, 0.0, 0.0]])


@pytest.fixture
def ref_table_atom0():
    """Reference table: atom 0 is the moving atom (move_atom_idx=0)."""
    return make_reference_table(move_atom_idx=0, idx_ref=0)


# ---------------------------------------------------------------------------
# DirectionBias
# ---------------------------------------------------------------------------

class TestDirectionBias:

    def test_accept_displacement_aligned(self, system_origin, ref_table_atom0):
        """Event moving atom in +x should be accepted for direction=[1,0,0]."""
        bias = DirectionBias(direction=[1, 0, 0])
        event = make_event(atom_index=0, final_positions=[[1.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is True

    def test_reject_displacement_opposite(self, system_origin, ref_table_atom0):
        """Event moving atom in -x should be rejected for direction=[1,0,0]."""
        bias = DirectionBias(direction=[1, 0, 0])
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is False

    def test_accept_zero_projection_at_threshold(self, system_origin, ref_table_atom0):
        """Displacement perpendicular to direction has projection 0, which equals
        the default threshold of 0 and should be accepted."""
        bias = DirectionBias(direction=[1, 0, 0])
        event = make_event(atom_index=0, final_positions=[[0.0, 1.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is True

    def test_custom_threshold_rejects_small_projection(self, system_origin, ref_table_atom0):
        """Projection 0.1 should be rejected when threshold=0.5."""
        bias = DirectionBias(direction=[1, 0, 0], threshold=0.5)
        event = make_event(atom_index=0, final_positions=[[0.1, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is False

    def test_custom_threshold_accepts_large_projection(self, system_origin, ref_table_atom0):
        """Projection 1.0 should be accepted when threshold=0.5."""
        bias = DirectionBias(direction=[1, 0, 0], threshold=0.5)
        event = make_event(atom_index=0, final_positions=[[1.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is True

    def test_atom_not_in_atom_set_pass_unlisted_true(self, system_origin, ref_table_atom0):
        """With pass_unlisted=True, atoms not in atom_indices always pass."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[99], pass_unlisted=True)
        event = make_event(atom_index=0, final_positions=[[-5.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is True

    def test_atom_not_in_atom_set_pass_unlisted_false(self, system_origin, ref_table_atom0):
        """With pass_unlisted=False (default), atoms not in atom_indices return False."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[99])
        event = make_event(atom_index=0, final_positions=[[-5.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is False

    def test_atom_in_atom_set_is_filtered(self, system_origin, ref_table_atom0):
        """Event from atom in atom_indices is subject to the direction filter."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[0])
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is False

    def test_direction_is_normalised(self, system_origin, ref_table_atom0):
        """Bias should normalise the direction vector internally."""
        bias = DirectionBias(direction=[2, 0, 0])
        assert np.allclose(np.linalg.norm(bias._direction), 1.0)

    def test_enabled_false_bypasses_filter(self, system_origin, ref_table_atom0):
        """When enabled=False, select() returns unbiased result without calling accept."""
        bias = DirectionBias(direction=[1, 0, 0])
        bias.enabled = False
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=1.0),
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active,
                                          system_origin, ref_table_atom0)
        assert idx == 0

    def test_select_returns_accepted_event(self, ref_table_atom0):
        """select() should return an event that passes accept()."""
        system = make_system([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        ref.table = pd.DataFrame({
            "idx_ref": [0, 0],
            "move_atom_idx": [0, 0],
        })
        bias = DirectionBias(direction=[1, 0, 0])
        # event 0: moves in -x (rejected), event 1: moves in +x (accepted)
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=10.0),
            make_event(atom_index=1, final_positions=[[ 1.0, 0.0, 0.0]], k=1.0),
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active, system, ref)
        # only event 1 is in +x direction
        assert idx == 1

    def test_select_fallback_when_all_rejected(self, system_origin, ref_table_atom0):
        """select() should fall back to unbiased selection when all events are rejected."""
        bias = DirectionBias(direction=[1, 0, 0])
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=1.0),
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        # all events are rejected; fallback should return index 0
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active,
                                          system_origin, ref_table_atom0)
        assert idx == 0
        assert ktot > 0.0


# ---------------------------------------------------------------------------
# PointBias
# ---------------------------------------------------------------------------

class TestPointBias:

    def test_accept_moving_toward_target(self, ref_table_atom0):
        """Event that moves atom toward the target should be accepted."""
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0])
        # displacement moves atom from origin toward [5,0,0]
        event = make_event(atom_index=0, final_positions=[[1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is True

    def test_reject_moving_away_from_target(self, ref_table_atom0):
        """Event that moves atom away from the target should be rejected."""
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0])
        # displacement moves atom away from [5,0,0]
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is False

    def test_accept_when_already_at_target(self, ref_table_atom0):
        """When atom is already at target (dist ≈ 0), event should be accepted."""
        system = make_system([[5.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0])
        event = make_event(atom_index=0, final_positions=[[6.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is True

    def test_atom_not_in_atom_set_pass_unlisted_true(self, ref_table_atom0):
        """With pass_unlisted=True, atoms not in atom_indices always pass."""
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0], atom_indices=[99], pass_unlisted=True)
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is True

    def test_atom_not_in_atom_set_pass_unlisted_false(self, ref_table_atom0):
        """With pass_unlisted=False (default), atoms not in atom_indices return False."""
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0], atom_indices=[99])
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is False

    def test_negative_threshold_accepts_away(self, ref_table_atom0):
        """Negative threshold: accept events that move away from the target."""
        system = make_system([[0.0, 0.0, 0.0]])
        # projection onto toward-vector for displacement [-1,0,0] is -1; threshold=-2 → -1 >= -2 → True
        bias = PointBias(target_point=[5.0, 0.0, 0.0], threshold=-2.0)
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is True

    def test_enabled_false_bypasses_filter(self, ref_table_atom0):
        """When enabled=False, select() returns unbiased result."""
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5.0, 0.0, 0.0])
        bias.enabled = False
        events = [make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=1.0)]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, _, _ = bias.select(rejection_free, l_k, active, system, ref_table_atom0)
        assert idx == 0

    def test_select_returns_accepted_event(self, ref_table_atom0):
        """select() should return the event moving toward the target."""
        system = make_system([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ref = MagicMock()
        ref.table = pd.DataFrame({"idx_ref": [0, 0], "move_atom_idx": [0, 0]})
        bias = PointBias(target_point=[5.0, 0.0, 0.0])
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=10.0),  # away
            make_event(atom_index=1, final_positions=[[ 1.0, 0.0, 0.0]], k=1.0),   # toward
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active, system, ref)
        assert idx == 1


# ---------------------------------------------------------------------------
# TopoBias
# ---------------------------------------------------------------------------

def make_atomic_environment(topo_map: dict[str, list[int]]):
    """Return an AtomicEnvironment-like mock.

    Parameters
    ----------
    topo_map : dict[str, list[int]]
        Maps topology ID to list of atom indices carrying that topology.
    """
    ae = MagicMock()
    ae.get_atoms_with_id.side_effect = lambda topo_id: topo_map.get(topo_id, [])
    return ae


class TestTopoBias:

    # ------------------------------------------------------------------
    # _prepare + accept basics
    # ------------------------------------------------------------------

    def test_accept_source_moving_toward_target(self):
        """Source-topology atom moving toward target-topology atom is accepted."""
        # atom 0 is source at origin; atom 1 is target at [5,0,0]
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        # displacement +x: moves from [0,0,0] toward [5,0,0] → final [1,0,0]
        event = make_event(atom_index=0, final_positions=[[1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref)

    def test_reject_source_moving_away_from_target(self):
        """Source-topology atom moving away from target is rejected."""
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        # displacement -x: moves away from [5,0,0] → final [-1,0,0]
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert not bias.accept(event, system, ref)

    def test_accept_non_source_atom_pass_unlisted_true(self):
        """With pass_unlisted=True, non-source atoms always pass."""
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia", pass_unlisted=True)
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=2, final_positions=[[1.0, 1.0, 0.0]])
        assert bias.accept(event, system, ref) is True

    def test_accept_non_source_atom_pass_unlisted_false(self):
        """With pass_unlisted=False (default), non-source atoms return False."""
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=2, final_positions=[[1.0, 1.0, 0.0]])
        assert bias.accept(event, system, ref) is False

    def test_accept_when_source_topology_absent(self):
        """When no atom carries topo_source, all events are accepted (fallback)."""
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        ae = make_atomic_environment({"sia": [1]})          # no "vac" atoms
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref) is True

    def test_accept_when_target_topology_absent(self):
        """When no atom carries topo_target, all events are accepted (fallback)."""
        system = make_system([[0.0, 0.0, 0.0]])
        ae = make_atomic_environment({"vac": [0]})          # no "sia" atoms
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref) is True

    # ------------------------------------------------------------------
    # select() integration
    # ------------------------------------------------------------------

    def test_select_returns_event_reducing_distance(self):
        """select() should return the event that reduces source-target distance."""
        # atom 0 = source at origin, atom 1 = target at [10,0,0]
        # atom 2 = source at origin (same pos), used for second event
        system = make_system([
            [0.0, 0.0, 0.0],   # source atom (event 0 and 1 both use atom 0)
            [10.0, 0.0, 0.0],  # target atom
        ])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = MagicMock()
        ref.table = pd.DataFrame({"idx_ref": [0, 0], "move_atom_idx": [0, 0]})
        bias = TopoBias(topo_source="vac", topo_target="sia")
        # event 0: moves source away from target (rejected)
        # event 1: moves source toward target (accepted)
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=10.0),
            make_event(atom_index=0, final_positions=[[ 1.0, 0.0, 0.0]], k=1.0),
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active, system, ref,
                                          atomic_environment=ae)
        assert idx == 1

    def test_select_fallback_when_all_rejected(self):
        """select() falls back to unbiased when all events are rejected."""
        system = make_system([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        events = [
            make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=1.0),
        ]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, delta_t, ktot = bias.select(rejection_free, l_k, active, system, ref,
                                          atomic_environment=ae)
        # fallback: only one event exists so it must be returned
        assert idx == 0
        assert ktot > 0.0

    def test_select_enabled_false_bypasses(self):
        """When enabled=False, select() returns unbiased result."""
        system = make_system([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias.enabled = False
        events = [make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=1.0)]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        idx, _, _ = bias.select(rejection_free, l_k, active, system, ref,
                                 atomic_environment=ae)
        assert idx == 0


# ---------------------------------------------------------------------------
# Boost mode
# ---------------------------------------------------------------------------

def _two_event_setup():
    """System with 2 events: idx=0 moves +x (desired), idx=1 moves -x (undesired)."""
    system = make_system([[0.0, 0.0, 0.0]])
    ref = MagicMock()
    ref.table = pd.DataFrame({"idx_ref": [0, 0], "move_atom_idx": [0, 0]})
    events = [
        make_event(atom_index=0, final_positions=[[ 1.0, 0.0, 0.0]], k=1.0),
        make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=9.0),
    ]
    active = make_active_table(events)
    l_k = np.array([e["k"] for e in events])
    return system, ref, active, l_k


class TestBoostMode:

    def test_selection_fraction_matches_bias_weight(self):
        """With bias_weight=0.5, roughly 50% of selections should be the desired event."""
        system, ref, active, l_k = _two_event_setup()
        bias = DirectionBias(direction=[1, 0, 0], mode="boost", bias_weight=0.5)
        n_desired = sum(
            bias.select(rejection_free, l_k, active, system, ref)[0] == 0
            for _ in range(2000)
        )
        assert abs(n_desired / 2000 - 0.5) < 0.05

    def test_time_correction_gives_correct_mean(self):
        """δt * k_total_true should be Exponential(1), mean ≈ 1."""
        system, ref, active, l_k = _two_event_setup()
        bias = DirectionBias(direction=[1, 0, 0], mode="boost", bias_weight=0.5)
        k_total_true = float(l_k.sum())  # 10.0
        products = [
            bias.select(rejection_free, l_k, active, system, ref)[1] * k_total_true
            for _ in range(1000)
        ]
        assert abs(np.mean(products) - 1.0) < 0.1

    def test_fallback_no_desired_events(self):
        """When k_boost=0 (no desired events), fall back to unbiased selection."""
        system = make_system([[0.0, 0.0, 0.0]])
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        events = [make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]], k=5.0)]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        bias = DirectionBias(direction=[1, 0, 0], mode="boost", bias_weight=0.5)
        idx, _, ktot = bias.select(rejection_free, l_k, active, system, ref)
        assert idx == 0
        assert np.isclose(ktot, 5.0)

    def test_fallback_all_desired_events(self):
        """When k_free=0 (all events desired), fall back to unbiased selection."""
        system = make_system([[0.0, 0.0, 0.0]])
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        events = [make_event(atom_index=0, final_positions=[[1.0, 0.0, 0.0]], k=5.0)]
        active = make_active_table(events)
        l_k = np.array([e["k"] for e in events])
        bias = DirectionBias(direction=[1, 0, 0], mode="boost", bias_weight=0.5)
        idx, _, ktot = bias.select(rejection_free, l_k, active, system, ref)
        assert idx == 0
        assert np.isclose(ktot, 5.0)

    def test_undesired_events_can_still_fire(self):
        """In boost mode, undesired events (idx=1) are never blocked."""
        system, ref, active, l_k = _two_event_setup()
        # Very low bias_weight: desired event rarely selected, undesired must appear
        bias = DirectionBias(direction=[1, 0, 0], mode="boost", bias_weight=0.01)
        selections = [
            bias.select(rejection_free, l_k, active, system, ref)[0]
            for _ in range(200)
        ]
        assert 1 in selections

    def test_pass_unlisted_true_with_atom_indices_raises_direction(self):
        """DirectionBias: boost + pass_unlisted=True + atom_indices → ValueError."""
        with pytest.raises(ValueError, match="pass_unlisted=True"):
            DirectionBias(direction=[1, 0, 0], atom_indices=[0],
                          mode="boost", pass_unlisted=True)

    def test_pass_unlisted_true_with_atom_indices_raises_point(self):
        """PointBias: boost + pass_unlisted=True + atom_indices → ValueError."""
        with pytest.raises(ValueError, match="pass_unlisted=True"):
            PointBias(target_point=[1, 0, 0], atom_indices=[0],
                      mode="boost", pass_unlisted=True)

    def test_pass_unlisted_true_raises_topo(self):
        """TopoBias: boost + pass_unlisted=True → ValueError (always, no atom_indices check)."""
        with pytest.raises(ValueError, match="pass_unlisted=True"):
            TopoBias(topo_source="vac", topo_target="sia",
                     mode="boost", pass_unlisted=True)

    def test_pass_unlisted_false_with_atom_indices_is_valid(self):
        """boost + pass_unlisted=False + atom_indices set is valid; non-listed atoms go to k_free."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[0],
                             mode="boost", pass_unlisted=False)
        assert bias.mode == "boost"
        assert not bias.pass_unlisted


# ---------------------------------------------------------------------------
# pass_unlisted
# ---------------------------------------------------------------------------

class TestPassUnlisted:

    def test_direction_pass_unlisted_false_non_listed(self, system_origin, ref_table_atom0):
        """Default pass_unlisted=False: non-listed atoms return False from accept()."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[99])
        event = make_event(atom_index=0, final_positions=[[-5.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is False

    def test_direction_pass_unlisted_true_non_listed(self, system_origin, ref_table_atom0):
        """pass_unlisted=True: non-listed atoms return True from accept()."""
        bias = DirectionBias(direction=[1, 0, 0], atom_indices=[99], pass_unlisted=True)
        event = make_event(atom_index=0, final_positions=[[-5.0, 0.0, 0.0]])
        assert bias.accept(event, system_origin, ref_table_atom0) is True

    def test_point_pass_unlisted_false_non_listed(self, ref_table_atom0):
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5, 0, 0], atom_indices=[99])
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is False

    def test_point_pass_unlisted_true_non_listed(self, ref_table_atom0):
        system = make_system([[0.0, 0.0, 0.0]])
        bias = PointBias(target_point=[5, 0, 0], atom_indices=[99], pass_unlisted=True)
        event = make_event(atom_index=0, final_positions=[[-1.0, 0.0, 0.0]])
        assert bias.accept(event, system, ref_table_atom0) is True

    def test_topo_pass_unlisted_false_non_source(self):
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia")
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=2, final_positions=[[1.0, 1.0, 0.0]])
        assert bias.accept(event, system, ref) is False

    def test_topo_pass_unlisted_true_non_source(self):
        system = make_system([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        ae = make_atomic_environment({"vac": [0], "sia": [1]})
        ref = make_reference_table(move_atom_idx=0, idx_ref=0)
        bias = TopoBias(topo_source="vac", topo_target="sia",
                        mode="filter", pass_unlisted=True)
        bias._prepare(system, ref, ae)
        event = make_event(atom_index=2, final_positions=[[1.0, 1.0, 0.0]])
        assert bias.accept(event, system, ref) is True
