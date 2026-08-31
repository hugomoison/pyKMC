"""Module implementing ShapeTable, the persistent table of shape identity.

Distinct from `event_table.ReferenceEventTable` (the table of discovered
transitions between shapes): this is the table of the shapes themselves --
what minima and saddles are known, keyed by their own resolved identity,
independent of how many (if any) transitions currently reference them.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Literal

from .parameters import Parameters
from .system import System, Configuration
from .neighbors_list import NeighborsList
from .result import ShapeID, SaddleID
from .point_set_registration import simple_ira, check_match
from .utils.geometry import unwrap_around


@dataclass
class ShapeKnowledge:
    """Cross-step knowledge for one `ShapeID`, persisted on `ShapeTable`.

    Survives across steps and restarts -- it is the record of "what do we
    know, ever, about this shape's event population," independent of
    whether any of that knowledge came from a dispatched search of this
    exact shape or an opportunistic discovery made while searching
    something else. Keyed externally (on `ShapeTable.shape_knowledge`) by
    the `ShapeID` itself, not by its coarse id alone -- nauty's coarse id
    can collide across genuinely different local neighborhoods, so a
    coarse-only key would let one shape's convergence history mask a
    different shape's total lack of evidence just because they happen to
    share a coarse id.

    Attributes
    ----------
    status : Literal["unsearched", "opportunistic", "completed"]
        "unsearched": no rows and no dispatched session yet. "opportunistic":
        has rediscovery/rate data (e.g. from a disconnected event landing on
        this shape) but has never been through its own dispatched session.
        "completed": a dispatched session (adaptive convergence/cap, or a
        full `nsearch` round in non-adaptive mode) for this exact shape has
        finished -- the only status `resolve_new_shapes()` trusts as
        "don't search again."
    rediscovery_counts : dict[int, int]
        idx_ref -> number of times catalogued, same semantics as
        `pykmc.adaptive_search.ShapeSearchStats.rediscovery_counts` but
        accumulated over this shape's whole lifetime, not just one step's
        session.
    known_rates : dict[int, float]
        idx_ref -> cached rate constant k_i, same lifetime as above.
    rep_config : Configuration | None
        This shape's own representative local-cluster geometry, set once
        when the shape is first resolved (minted) and never changed
        afterward -- `ShapeTable` is the sole catalog of minima identity, so
        this is the only geometry any later classification or resolution
        compares a candidate against. `None` only for the brief span before
        a shape has ever actually been resolved (e.g. a `setdefault`-created
        placeholder from `mark_shape_completed`).
    rep_atom_idx : int | None
        `rep_config`'s own central atom index, `None` iff `rep_config` is
        `None`.

    """

    status: Literal["unsearched", "opportunistic", "completed"] = "unsearched"
    rediscovery_counts: dict[int, int] = field(default_factory=dict)
    known_rates: dict[int, float] = field(default_factory=dict)
    rep_config: Configuration | None = None
    rep_atom_idx: int | None = None

    def record(self, idx_ref: int, k: float) -> None:
        """Record one sighting (new or repeat) of `idx_ref`, rate `k`.

        Advances `status` from "unsearched" to "opportunistic" if this is the
        first knowledge ever recorded for this shape; never downgrades a
        "completed" shape back to "opportunistic".
        """
        self.rediscovery_counts[idx_ref] = self.rediscovery_counts.get(idx_ref, 0) + 1
        self.known_rates[idx_ref] = k
        if self.status == "unsearched":
            self.status = "opportunistic"


@dataclass
class SaddleKnowledge:
    """A saddle's permanent catalog entry: its own representative geometry, nothing else.

    Kept separate from `ShapeKnowledge` (minima) purely so a candidate's
    matching pool never mixes the two kinds -- a saddle is never
    independently dispatched or searched the way a minimum is (it only ever
    arises as a byproduct of a minimum's dispatched search), so it carries
    none of `ShapeKnowledge`'s search-completeness/rediscovery bookkeeping.

    Attributes
    ----------
    rep_config : Configuration
        This saddle's own representative local-cluster geometry, set once
        when it is first resolved (minted) and never changed afterward.
    rep_atom_idx : int
        `rep_config`'s own central atom index.

    """

    rep_config: Configuration
    rep_atom_idx: int


class ShapeTable:
    """The persistent table of shape identity: what minima and saddles are known.

    Distinct from `event_table.ReferenceEventTable` (the table of discovered
    transitions between shapes): a shape's existence here is independent of
    how many (if any) transitions in `ReferenceEventTable.table` currently
    reference it -- a shape resolved live with zero catalogued events is
    just as real an entry as one with many.

    Minima (`shape_knowledge`) and saddles (`saddle_knowledge`) are kept in
    separate dicts, keyed by `ShapeID`/`SaddleID` respectively, so resolving
    one never has to filter out entries of the other kind. Each entry holds
    exactly one permanent representative geometry, set once when the shape
    is first resolved (minted) and never replaced afterward.

    Parameters
    ----------
    params : Parameters
        The atomic simulations configuration.

    """

    def __init__(self, params: Parameters) -> None:
        self.params = params
        if params.control.topology_search_status is not None:
            with open(params.control.topology_search_status, "rb") as file:
                data = pickle.load(file)
            self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = data["shape_knowledge"]
            self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = data["saddle_knowledge"]
        else:
            self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = {}
            self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = {}

    def known_ids(self) -> set[str]:
        """The `id_initial` values with at least one catalogued shape."""
        return {shape.id for shape in self.shape_knowledge}

    def _classify(
        self,
        knowledge: dict,
        coarse_id: str,
        configuration: Configuration,
        move_atom_idx: int | None = None,
    ) -> int | None:
        """Match `configuration` against every representative in `knowledge` sharing `coarse_id`; `None` if none match.

        Shared mechanism behind minima/saddle classification, live or
        cataloguing-time: `knowledge` (`self.shape_knowledge` or
        `self.saddle_knowledge`) already holds exactly one representative
        geometry per entry, so this is a single IRA/PSR comparison per
        candidate sid, tried in ascending sid order for determinism.

        Parameters
        ----------
        knowledge : dict[ShapeID, ShapeKnowledge] | dict[SaddleID, SaddleKnowledge]
            The catalog to classify against.
        coarse_id : str
            The candidate's coarse topology hash (``id_initial`` or ``id_saddle``).
        configuration : Configuration
            The candidate's local-cluster types/positions/cell.
        move_atom_idx : int | None
            Index of the candidate's own central/moving atom within ``configuration``, if known -- seeds the IRA search against each representative's own ``rep_atom_idx`` instead of leaving it to find a geometric center.

        Returns
        -------
        int | None
            The matching sid, or `None` if nothing under `coarse_id` matches
            (including the case where nothing is catalogued under it at all).

        """
        full = self.params.atomicenvironment.coloring_mode == "full"
        candidates = sorted(shape for shape in knowledge if shape.id == coarse_id)
        for shape in candidates:
            know = knowledge[shape]
            result = simple_ira(
                configuration,
                know.rep_config,
                self.params.ira.kmax_factor,
                full=full,
                candidate1=move_atom_idx,
                candidate2=know.rep_atom_idx,
            )
            if check_match(result, self.params.psr.matching_score_thr).is_ok():
                return shape.sid
        return None

    def _resolve(
        self,
        knowledge: dict,
        key_cls,
        entry_cls,
        coarse_id: str,
        configuration: Configuration,
        move_atom_idx: int | None,
    ) -> int:
        """Classify-or-mint: match `configuration` against `knowledge` under `coarse_id`, minting a fresh entry if nothing matches.

        Shared mechanism behind `resolve_sid` (minima or saddle -- `key_cls`/
        `entry_cls` select which) and `resolve_live_shape`'s mint path.
        """
        sid = self._classify(knowledge, coarse_id, configuration, move_atom_idx)
        if sid is not None:
            return sid

        existing = [shape.sid for shape in knowledge if shape.id == coarse_id]
        new_sid = max(existing) + 1 if existing else 0
        knowledge[key_cls(coarse_id, new_sid)] = entry_cls(
            rep_config=configuration, rep_atom_idx=move_atom_idx
        )
        return new_sid

    def resolve_sid(
        self,
        kind: Literal["initial", "saddle"],
        coarse_id: str,
        configuration: Configuration,
        move_atom_idx: int | None = None,
    ) -> int:
        """Resolve a candidate geometry's sid under `coarse_id`, within the `kind` catalog -- together they form its `ShapeID`/`SaddleID`.

        `kind="initial"` classifies against `self.shape_knowledge` (minima).
        `kind="saddle"` classifies against `self.saddle_knowledge`,
        free-standing -- the same transition-state geometry is recognized as
        one catalogued saddle regardless of which reaction pathway reaches
        it, unlike the old pathway-scoped resolution this replaced. Either
        way, if nothing matches, this candidate becomes a freshly minted
        entry's own permanent representative.
        """
        if kind == "initial":
            knowledge, key_cls, entry_cls = self.shape_knowledge, ShapeID, ShapeKnowledge
        else:
            knowledge, key_cls, entry_cls = self.saddle_knowledge, SaddleID, SaddleKnowledge
        return self._resolve(knowledge, key_cls, entry_cls, coarse_id, configuration, move_atom_idx)

    def _live_configuration(
        self, system: System, neighbors_list: NeighborsList, atom_idx: int
    ) -> tuple[Configuration, int]:
        """This atom's own rcut-neighbor cluster, unwrapped around itself, and its own index within it."""
        nl = neighbors_list.get_neighbors("rcut", atom_idx).copy()
        cfg = unwrap_around(system.configuration[nl], system.positions[atom_idx])
        return cfg, nl.index(atom_idx)

    def classify_live_sid(
        self,
        system: System,
        neighbors_list: NeighborsList,
        central_atom_index: int,
        id_initial: str,
    ) -> int | None:
        """Classify a live atom's ``sid_initial`` against already-catalogued shapes sharing its ``id_initial``.

        Unlike `resolve_sid` (used when cataloguing a freshly found event,
        which always assigns a new sid if none match), this never invents
        one: it only reports whether the atom's current geometry already
        matches a known `ShapeID`, so callers can decide whether a fresh
        event search is warranted for it.

        Parameters
        ----------
        system : System
            The live atomic system.
        neighbors_list : NeighborsList
            The system's neighbor lists.
        central_atom_index : int
            Index of the atom to classify.
        id_initial : str
            The atom's coarse topology hash.

        Returns
        -------
        int | None
            The matching ``sid_initial``, or ``None`` if the atom's live
            geometry matches no `ShapeID` already catalogued under
            ``id_initial`` (including the case where ``id_initial`` has no
            rows at all).

        """
        if not any(shape.id == id_initial for shape in self.shape_knowledge):
            return None

        live_configuration, move_atom_idx = self._live_configuration(
            system, neighbors_list, central_atom_index
        )
        return self._classify(self.shape_knowledge, id_initial, live_configuration, move_atom_idx)

    def resolve_live_shape(
        self,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment,
        atom_idx: int,
        *,
        mint: bool,
    ) -> ShapeID | None:
        """Resolve one atom's live `ShapeID`.

        With `mint=False`, never invents one: `None` if the atom's current
        geometry matches nothing already catalogued under its `id_initial`,
        so a caller can decide whether a fresh search is warranted at all.

        With `mint=True`, always returns a real `ShapeID`, minting one backed
        by the atom's own geometry if nothing matches -- meant to be called
        once a search has already been dispatched for this atom: by then
        there's nothing left to decide, only a shape left to record
        knowledge against, event or no event. The minted representative is
        this atom's own rcut-neighbor cluster, unwrapped around itself, not
        anything the dispatched search returns, so this never has to wait on
        (or depend on the outcome of) that search.
        """
        id_initial = atomic_environment.atomic_environment_list[atom_idx]
        if not mint:
            sid = self.classify_live_sid(system, neighbors_list, atom_idx, id_initial)
            return None if sid is None else ShapeID(id_initial, sid)

        live_configuration, move_atom_idx = self._live_configuration(system, neighbors_list, atom_idx)
        sid = self.resolve_sid("initial", id_initial, live_configuration, move_atom_idx)
        return ShapeID(id_initial, sid)

    def record_shape_knowledge(
        self, id_initial: str, sid_initial: int, idx_ref: int, k: float
    ) -> None:
        """Record one sighting (new catalogue entry or rediscovery) of `idx_ref` under the shape `ShapeID(id_initial, sid_initial)`.

        Called unconditionally for every successful catalogue outcome
        (`ReferenceEventTable.add_events`), regardless of which atom's
        search produced it or whether this shape has an open per-step
        adaptive session -- this is the single place persistent, cross-step
        knowledge is kept up to date.
        """
        know = self.shape_knowledge.setdefault(
            ShapeID(id_initial, sid_initial), ShapeKnowledge()
        )
        know.record(idx_ref, k)

    def mark_shape_completed(self, shape: ShapeID) -> None:
        """Mark `shape`'s persistent search-completeness status `"completed"`.

        Together with `record_shape_knowledge`/`get_shape_knowledge`/
        `forget`, the only sanctioned way to touch `shape_knowledge` --
        callers must never read or write that dict directly.
        """
        self.shape_knowledge.setdefault(shape, ShapeKnowledge()).status = "completed"

    def get_shape_knowledge(self, shape: ShapeID) -> ShapeKnowledge | None:
        """Look up `shape`'s persistent knowledge, or `None` if never sighted."""
        return self.shape_knowledge.get(shape)

    def forget(self, shape: ShapeID, idx_ref: int) -> None:
        """Prune `idx_ref` out of `shape`'s rediscovery bookkeeping, if `shape` is catalogued.

        Called by `ReferenceEventTable.remove()` for each row it deletes, so
        a removed transition's `idx_ref` doesn't keep skewing the
        Good-Turing undiscovered-mass estimate or `n_distinct_events` stat.
        `shape`'s own catalog entry is never deleted here -- a shape with
        zero remaining table rows is a normal state, not staleness.
        """
        know = self.shape_knowledge.get(shape)
        if know is not None:
            know.rediscovery_counts.pop(idx_ref, None)
            know.known_rates.pop(idx_ref, None)

    def save(self) -> None:
        """Save the shape catalogs to `params.control.topology_search_status_output`, if set."""
        if self.params.control.topology_search_status_output is None:
            return
        with open(self.params.control.topology_search_status_output, "wb") as file:
            pickle.dump(
                {"shape_knowledge": self.shape_knowledge, "saddle_knowledge": self.saddle_knowledge},
                file,
            )
