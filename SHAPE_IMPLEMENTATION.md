# Shape/ShapeID implementation


This document introduces a two-level identity for local atomic geometry. The
pre-existing coarse nauty graph-hash string is used and
treated as a pre-filter to resulve exact shape in a catalogue of Shape.
A new integer `sid` (`shape_id`) disambiguates within a given `id` (`topo_id`)
via IRA/PSR geometric matching against a persistent catalogue of
representatives. The pair `(id, sid)` travels together as `ShapeID` (for
minima) or `SaddleID` (for saddle points).

## The issue: shape matching used to reject, not to discover

Shape matching was used to reject events downstream, instead of to
determine what's new in the system in the first place. Events get removed
for the wrong reason: a shape match rejects an event
because its catalogued initial configuration doesn't match the live
shape, but that check only exists because the system doesn't already
know, upfront, what shape the atom actually has. The right thing would
have been to use that same shape-matching — against catalogued
configurations sharing the topo_id — directly, to determine what's new in
the system.

Concretely: `get_new_environments()` decides whether an atom's local
environment still needs a search by checking its coarse nauty topo id
against the set of already-known/visited topo ids
(`self.known_environments`/`self.visited_environments`). If any atom
sharing that topo id has already triggered a search, every other atom
with the same topo id — even one whose actual local geometry is a
genuinely different shape — isn't considered new and never gets searched.
The topo id gives no way to tell the two shapes apart.

Separately, at refinement time, any catalogued event whose initial_id
matches an atom's topo id is treated as an applicable candidate
(`has_id_subset_table`/`get_atoms_with_id`), with nothing checking whether
the event's actual shape matches the atom's actual shape first. Only
afterward does `Refinement`'s own PSR/IRA verification reject the
candidate for not matching. That rejection is itself evidence: the topo id
represented a different shape, and something upstream is new and should
have been searched.

Together, a shape can end up with **no valid event at all**: never
searched, because its topo id already looks known from an unrelated
shape, and any topo-id-only candidate it does get handed is correctly
rejected at refinement for not matching. An atom stuck this way simply
never fires — the system doesn't advance for it.

Basins run into the same mistake. Basin exploration builds connectivity
only from events already in the reference table, so
`is_states_has_unknown_environments` decides whether everything a state
needs to know is already known, by checking the coarse topo id against
the same visited-id set the main loop uses. A coarse id can look known
purely because a different shape sharing it was searched elsewhere in the
system. A state whose atom's real shape has not been searched still
passes this check, gets explored as an ordinary transient state, and is
handed whatever catalogued events happen to share its coarse id — events
belonging to a different physical shape. Depending on how that mismatch
is later classified, one of two things happens: either it silently
becomes part of the explored basin and exploration continues past it as
if it were real, compounding the error deeper, or a later geometry check
catches the mismatch and the entire basin computation is abandoned,
falling back to a plain KMC step. Either way, the state's environment was
never actually known in the first place.

## The fix: shape resolved upfront, before refinement ever runs

Shape matching is now used the right way: to determine what's new in the
system directly, so rejection downstream doesn't even need to occur. An
atom's real shape — its nauty topo_id together with its shape_id — is
resolved up front (`resolve_new_shapes()`), before refinement is ever
involved, so the system knows whether this atom's exact shape has already
been catalogued (`ShapeTable`, keyed by `ShapeID`). Refinement candidates
are then pre-selected by that same `ShapeID` match (`live_events()`), so a
candidate handed to `Refinement` is already known to be geometrically the
right shape.

For basins, `is_states_has_unknown_environments` now classifies each
atom's real local shape directly (`ShapeTable` / `classify_live_sid`/
`resolve_live_shape`). Basin exploration only ever proceeds on states
whose every atom's shape is genuinely resolved, so neither the silent
compounding nor the wasted, abandoned computation described above can
happen anymore.

## 1. `ShapeID` and `SaddleID` — the identity types (`pykmc/result.py`)

Two small `NamedTuple` types give a name to the compound identity the rest
of the system is built on:

```python
class ShapeID(NamedTuple):
    id: str    # topo_id: coarse nauty graph-hash of the local topology
    sid: int   # shape_id: disambiguates within that topo_id

class SaddleID(NamedTuple):
    id: str    # topo_id of the saddle-point geometry
    sid: int   # shape_id, resolved against a separate saddle catalogue
```

Neither field means anything on its own. `sid=2` only has meaning relative
to one specific `id` — it says nothing by itself, and the same `sid` value
under a different `id` is an unrelated shape. Two atoms sharing the same
`id` but different `sid` are, physically, two different local environments
that happen to hash to the same coarse topology.

`ShapeID` and `SaddleID` are structurally identical but kept as two
distinct types on purpose: a minimum's identity is resolved against one
catalogue, a saddle-point's identity against a separate one (Section 2), and
making them different types means a saddle's `SaddleID` can never be passed
where a minimum's `ShapeID` is expected — a mix-up would be a type error,
not a silent bug.

**What connects to this:** every other structure in this document is keyed
by one of these two types.

- `ShapeTable` (Section 2) stores its catalogue as
  `dict[ShapeID, ShapeKnowledge]` for minima and
  `dict[SaddleID, SaddleKnowledge]` for saddles.
- `ReferenceEventTable`'s reference table (Section 4) stores `id_initial` /
  `sid_initial` as two columns on every catalogued event — together they
  are that event's initial-state `ShapeID` — and `id_saddle` / `sid_saddle`
  the same way for its `SaddleID`.
- `KMC.resolve_new_shapes()` (Section 7) produces a
  `dict[ShapeID, list[int]]`: every live atom's identity for the current
  step, threaded through search dispatch, refinement candidate selection,
  and basin exploration.

## 2. `ShapeTable` — the persistent shape catalogue (`pykmc/shape_table.py`)

`ShapeTable` is the one place in the codebase that actually knows what
shapes exist. It is explicitly distinct from `ReferenceEventTable`
(Section 4): `ReferenceEventTable` catalogues *transitions* (events between
two states); `ShapeTable` catalogues *identity* (which local geometries are
known, independent of what events they're involved in). A
`ReferenceEventTable` owns one `ShapeTable` instance (`self.shapes`) and
calls into it every time it needs to know "is this geometry something
we've seen before."

### The two catalogues

```python
@dataclass
class ShapeKnowledge:
    status: Literal["unsearched", "completed"] = "unsearched"
    rep_config: Configuration = None   # this shape's permanent representative geometry
    rep_atom_idx: int = None           # the representative's own central-atom index

@dataclass
class SaddleKnowledge:
    rep_config: Configuration = None
    rep_atom_idx: int = None
```

`ShapeKnowledge` carries a `status` because minima are the thing whose
search-completeness the main KMC loop needs to track (Section 7:
`resolve_new_shapes()` reads exactly this field to decide if an atom still
needs an ARTn search). `SaddleKnowledge` has no `status` at all — saddle
points are never searched independently, they only ever show up as a
byproduct of a minimum's search, so there is nothing for a saddle to be
"completed" with respect to. Keeping the two as separate dataclasses,
instead of one generic record reused for both, means a minimum's
completion bookkeeping can never leak into saddle handling.

```python
class ShapeTable:
    def __init__(self, params):
        self.params = params
        if params.control.topology_search_status is not None:
            data = load_pickle(params.control.topology_search_status)
            self.shape_knowledge = data["shape_knowledge"]
            self.saddle_knowledge = data["saddle_knowledge"]
        else:
            self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = {}
            self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = {}
```

**Connects to:** `params.control.topology_search_status` names a restart
file — the whole catalogue survives a simulation restart, so a shape
discovered in a previous run is still known in this one.

### Classifying a geometry against the catalogue

This is the actual shape-matching primitive, shared by every other method
in this class:

```python
def _classify(self, knowledge, coarse_id, configuration, move_atom_idx=None):
    full = self.params.atomicenvironment.coloring_mode == "full"
    candidates = sorted(shape for shape in knowledge if shape.id == coarse_id)
    for shape in candidates:                       # ascending sid, for determinism
        rep = knowledge[shape]
        result = simple_ira(configuration, rep.rep_config,
                             self.params.ira.kmax_factor, full=full,
                             candidate1=move_atom_idx, candidate2=rep.rep_atom_idx)
        if check_match(result, self.params.psr.matching_score_thr).is_ok():
            return shape.sid
    return None   # nothing catalogued under coarse_id matches this geometry
```

`_classify` never looks outside `coarse_id`: it only ever compares
`configuration` against representatives that already share the same
topo_id. This is a genuine limitation, not an oversight — see the "Notes"
section at the end of this document for what it means in practice (two
physically identical shapes that happen to hash to *different* topo_ids
are never compared to each other, and end up catalogued as two separate
shapes).

`_resolve` builds on `_classify` to add "mint a new entry if nothing
matches":

```python
def _resolve(self, knowledge, key_cls, entry_cls, coarse_id, configuration, move_atom_idx):
    sid = self._classify(knowledge, coarse_id, configuration, move_atom_idx)
    if sid is not None:
        return sid
    existing = [shape.sid for shape in knowledge if shape.id == coarse_id]
    new_sid = max(existing) + 1 if existing else 0
    knowledge[key_cls(coarse_id, new_sid)] = entry_cls(rep_config=configuration,
                                                        rep_atom_idx=move_atom_idx)
    return new_sid
```

Minting means the candidate geometry itself becomes the new shape's
permanent representative — there is no separate "best" representative
chosen later; whichever geometry happens to trigger the mint is what every
future candidate under that `sid` gets compared against, from then on.

```python
def resolve_sid(self, kind, coarse_id, configuration, move_atom_idx=None):
    if kind == "initial":
        knowledge, key_cls, entry_cls = self.shape_knowledge, ShapeID, ShapeKnowledge
    else:  # kind == "saddle"
        knowledge, key_cls, entry_cls = self.saddle_knowledge, SaddleID, SaddleKnowledge
    return self._resolve(knowledge, key_cls, entry_cls, coarse_id, configuration, move_atom_idx)
```

**Connects to:** `ReferenceEventTable._build_event_series()` (Section 4) is
the only caller of `resolve_sid()` — it is where a freshly found ARTn
event's initial state, final state, and saddle point each get assigned a
`sid`, at the moment the event is catalogued.

### Classifying a live atom

```python
def classify_live_sid(self, system, neighbors_list, central_atom_index, id_initial):
    configuration, local_idx = self._live_configuration(system, neighbors_list, central_atom_index)
    return self._classify(self.shape_knowledge, id_initial, configuration, local_idx)
    # None if the atom's current geometry matches nothing catalogued

def resolve_live_shape(self, system, neighbors_list, atomic_environment, atom_idx, *, mint):
    id_initial = atomic_environment.atomic_environment_list[atom_idx]
    configuration, local_idx = self._live_configuration(system, neighbors_list, atom_idx)
    if not mint:
        sid = self._classify(self.shape_knowledge, id_initial, configuration, local_idx)
        return ShapeID(id_initial, sid) if sid is not None else None
    sid = self._resolve(self.shape_knowledge, ShapeID, ShapeKnowledge, id_initial, configuration, local_idx)
    return ShapeID(id_initial, sid)
```

`mint=False` answers "do we already know this atom's shape" without
creating a new catalogue entry — used whenever the caller only needs to
*check*. `mint=True` always returns a real `ShapeID`, minting one from the atom's own live
geometry if nothing matches — used once a search has actually been
dispatched for that atom, when there is no longer anything to decide, only
a shape identity left to record knowledge against.

**Connects to:**
- `KMC.resolve_new_shapes()` (Section 7) calls `resolve_live_shape(mint=True)`
  for every non-crystal atom, every step — this is the "shape resolved
  upfront" step described in "The fix" at the top of this document.
- `basins.is_states_has_unknown_environments()` (Section 5) calls
  `resolve_live_shape(mint=False)` — a basin never invents new shapes,
  since it never runs a live search; an atom that fails to classify means
  the state must be handed back to the main KMC loop.

### Recording and completing knowledge

```python
def record_shape_knowledge(self, id_initial, sid_initial, idx_ref, k):
    shape = ShapeID(id_initial, sid_initial)
    self.shape_knowledge.setdefault(shape, ShapeKnowledge()).record(idx_ref, k)
    # ShapeKnowledge.record advances status unsearched -> at-least-seen-once,
    # and never downgrades once it reaches "completed"

def mark_shape_completed(self, shape):
    self.shape_knowledge.setdefault(shape, ShapeKnowledge()).status = "completed"

def forget(self, shape, idx_ref):
    # called when a reference-table row is removed: keeps this shape's own
    # bookkeeping consistent, without deleting the shape's catalogue entry
    ...

def save(self):
    dump_pickle({"shape_knowledge": self.shape_knowledge,
                 "saddle_knowledge": self.saddle_knowledge}, ...)
```

`mark_shape_completed` is the only place `status` is ever set to
`"completed"` — the one value `resolve_new_shapes()` trusts as "this exact
shape doesn't need searching again." `record_shape_knowledge` is called
for every successful cataloguing outcome (Section 4:
`ReferenceEventTable._record_knowledge_from_result()`), whether the
discovered event turned out to be brand new or a rediscovery of something
already known — either way, the sighting itself is evidence about this
shape, independent of which specific event it produced.

**Connects to:** `ReferenceEventTable.save()` (Section 4) calls
`ShapeTable.save()` as part of its own save — the shape catalogue and the
event table are persisted together, even though they are two logically
separate stores.

## 3. Geometric matching primitives feeding `ShapeTable`

`ShapeTable._classify()` (Section 2) relies on two supporting capabilities
it doesn't implement itself: a way to actually compare two geometries
(`simple_ira`), and a way to compute a coarse `topo_id` for a geometry that
isn't the live system at all — e.g. a minimum or saddle point handed back
by an ARTn search.

### `simple_ira` — the geometric comparison itself

```python
def simple_ira(configuration_1, configuration_2, kmax_factor, full=False,
                candidate1=None, candidate2=None):
    nat, typ1, typ2 = ...  # derived from the two Configurations
    return ira_mod.IRA().match(nat, typ1, configuration_1.positions,
                                nat, typ2, configuration_2.positions,
                                kmax_factor, candidate1=candidate1, candidate2=candidate2)
```

`candidate1`/`candidate2` seed the search with a known atom correspondence
(the two geometries' own central/moving atom).

**Connects to:** this is the one comparison primitive shared by
`ShapeTable._classify()` (Section 2) and `PointSetRegistration.ira()` —
used elsewhere to align a catalogued event's geometry onto a live atom,
e.g. `basins.system_from_state` (Section 5). The same underlying code
decides both "is this the same shape" and "how do I rotate/translate this
template onto that atom."

### `compute_atomic_environment_id` — coarse id for an arbitrary geometry

```python
def compute_atomic_environment_id(configuration, atom_idx, params):
    neighbors_list = NeighborsList(configuration, atom_indices=[atom_idx], ...)
    local_cluster = configuration[neighbors_list.get_neighbors("rcut", atom_idx)]
    types = local_cluster.types if params.atomicenvironment.coloring_mode == "full" else None
    coarse_id = graph(local_cluster, types=types)   # same nauty hashing as the live system
    return coarse_id, neighbors_list
```

This always uses the same nauty graph-hashing the live system's own
`atomic_environment_list` uses, regardless of what classification style
the simulation is otherwise configured with — a catalogued shape's
`topo_id` has to stay comparable to a live atom's `topo_id` no matter how
the live system happens to be classified.

`NeighborsList`'s `atom_indices` parameter is what makes this cheap: it
builds a real neighbor list for just the one requested atom.

**Connects to:** `ReferenceEventTable._build_event_series()` (Section 4)
calls this once for each of an event's three states (initial, saddle,
final) to get the `topo_id` half of what then gets paired with a `sid`
from `ShapeTable.resolve_sid()`.

## 4. `ReferenceEventTable` — the catalogue of transitions (`pykmc/event_table.py`)

`ReferenceEventTable` is the table of *events*: an initial state, a saddle,
a final state. It owns a `ShapeTable` (Section 2) and calls
into it every time it needs to decide "is this event new," "which events
currently apply to this atom," or "what shape does this row actually
belong to."

```python
class ReferenceEventTable:
    def __init__(self, params):
        self.table = self._initialize_table()   # schema below
        self.shapes = ShapeTable(params)
```

### The row schema

Every row carries three coarse `topo_id`s (one per state) and, alongside
each, an integer `sid` — together, each `(id, sid)` pair is that row's
identity for its initial, saddle, and final states:

```
id_initial    str     # topo_id of the initial state
sid_initial   int     #   -> ShapeID(id_initial, sid_initial)
id_saddle     str     # topo_id of the saddle point
sid_saddle    int     #   -> SaddleID(id_saddle, sid_saddle)
id_final      str     # topo_id of the final state
sid_final     int     #   -> ShapeID(id_final, sid_final)
idx_ref       int     # this row's own index
idx_backward  int     # idx_ref of the paired reverse-direction row
...           # dE_forward, dE_backward, k, sym_matrix, sym_perm, move_atom_idx,
              # initial_configuration, saddle_configuration, final_configuration, ...
```

### `_rows_with_shape` — filtering by compound identity

```python
def _rows_with_shape(self, df, id_column, sid_column, shape):
    return df[(df[id_column] == shape.id) & (df[sid_column] == shape.sid)]
```

A tiny helper, but it is the one place the schema's `(id, sid)` column
pairs actually get treated as a single compound key, rather than as two
independent columns.

### `_build_event_series` — cataloguing a freshly found event

This is where a raw ARTn result (initial minimum, saddle, final minimum,
as plain configurations) turns into a table row with a real `ShapeID`
attached:

```python
def _build_event_series(self, artn_result, move_atom_idx):
    id_min1, _   = compute_atomic_environment_id(artn_result.min1, move_atom_idx, self.params)
    id_saddle, _ = compute_atomic_environment_id(artn_result.saddle, move_atom_idx, self.params)
    id_min2, _   = compute_atomic_environment_id(artn_result.min2, move_atom_idx, self.params)

    sid_initial_forward  = self.shapes.resolve_sid("initial", id_min1, artn_result.min1, move_atom_idx)
    sid_initial_backward = self.shapes.resolve_sid("initial", id_min2, artn_result.min2, move_atom_idx)
    sid_saddle           = self.shapes.resolve_sid("saddle",  id_saddle, artn_result.saddle, move_atom_idx)
    # the saddle is resolved once and shared: it's the same physical geometry
    # whether it's reached going forward or backward, so it has to land under
    # the same SaddleID both times, not be split in two by which direction
    # happened to resolve it first

    forward_row  = Series(event_id=combine_ids(id_min1, id_saddle, id_min2),
                           id_initial=id_min1, sid_initial=sid_initial_forward,
                           id_saddle=id_saddle, sid_saddle=sid_saddle,
                           id_final=id_min2,   sid_final=sid_initial_backward, ...)
    backward_row = Series(event_id=combine_ids(id_min2, id_saddle, id_min1),
                           id_initial=id_min2, sid_initial=sid_initial_backward,
                           id_saddle=id_saddle, sid_saddle=sid_saddle,
                           id_final=id_min1,   sid_final=sid_initial_forward, ...)
    return forward_row, backward_row
```

**Connects to:** `compute_atomic_environment_id` and
`ShapeTable.resolve_sid` (Sections 2 and 3) are the only sources of any
`topo_id`/`sid` value anywhere in the table — every row's shape identity
traces back to exactly these two calls, made once, at cataloguing time.

### `is_new_event` — is this the same event as one already catalogued

```python
def is_new_event(self, dfevent):
    shape = ShapeID(dfevent["id_initial"], dfevent["sid_initial"])
    subset = self._rows_with_shape(self.table, "id_initial", "sid_initial", shape)
    # subset now only contains rows whose initial state is genuinely the
    # same shape as dfevent's -- not just the same coarse topo_id

    saddle = SaddleID(dfevent["id_saddle"], dfevent["sid_saddle"])
    match = self._rows_with_shape(subset, "id_saddle", "sid_saddle", saddle)
    if len(match) > 0:
        return False, int(match.iloc[0]["idx_ref"])   # already known
    return True, None                                  # genuinely new
```

Both filters key on the full `(id, sid)` pair. This is
the concrete mechanism behind "The fix" at the top of this document:
`dfevent`'s `sid_initial`/`sid_saddle` were already resolved by
`_build_event_series` before `is_new_event` is ever called, so "is this
new" is answered directly from already-known shape identity, rather than
being inferred afterward from a coarse-id match that later fails to
verify.

### `live_events` — which atoms does a catalogued event actually apply to

```python
def live_events(self, atom_shapes: dict[int, ShapeID]):
    atoms_by_shape: dict[ShapeID, list[int]] = {}
    for atom, shape in atom_shapes.items():
        atoms_by_shape.setdefault(shape, []).append(atom)

    live_ids = {shape.id for shape in atoms_by_shape}          # coarse pre-filter only
    subset = self.table[self.table["id_initial"].isin(live_ids)]

    for _, row in subset.iterrows():
        shape = ShapeID(row["id_initial"], row["sid_initial"])
        for atom in atoms_by_shape.get(shape, []):              # exact match only
            yield atom, row
```

`live_ids` never gets exposed to any caller — it exists purely so the
dataframe filter doesn't have to scan every row on every call. Every pair
this function actually yields has already been confirmed to share the
exact `ShapeID`, both `id` and `sid`. That is what makes this function safe to
use as the applicability test everywhere it's called, instead of a
pre-filter whose false positives only get discovered later.

**Connects to:** `atom_shapes` always comes from `KMC.resolve_new_shapes()`
(Section 7), which resolves it once per step. `live_events()` is then
called from `KMC.build_refinement_candidates()` (Section 7) and
`BasinGenericEventExplorer.explore()` (Section 5) to find, respectively,
which catalogued events are worth refining this step, and which events
apply to a basin state being explored — the same underlying identity
feeding both.

### Everything else, briefly

- `is_valid_new_event()` orchestrates the above: it checks for
  self-reversal (`id_initial == id_final`, disambiguated by a geometric
  match if the coarse ids agree, rather than a raw distance check), then
  calls `is_new_event` for the forward direction. If new, it mints and
  adds both the forward and backward row together; if not, it returns
  `Err(EVENT_NOT_NEW, variables={"matched_idx_ref": ...})` so the caller
  can still credit the rediscovery.
- `_record_knowledge_from_result()` takes the result of the above and
  calls `self.shapes.record_shape_knowledge(...)` — a brand-new event and a
  rediscovered one are both sightings of a shape, so both get recorded.
- `remove(idx_ref)` deletes a row and calls
  `self.shapes.forget(ShapeID(...), idx_ref)` — the shape's catalogue
  entry survives; only its bookkeeping of which table rows reference it is
  updated.
- `save()` pickles `self.table`, then calls `self.shapes.save()` — the
  event table and the shape catalogue are always persisted together.

`ActiveEventTable`, in the same file, has **no** ShapeID/sid logic of its
own — live-atom-to-event matching happens entirely upstream via
`ReferenceEventTable.live_events`. It is nonetheless a downstream consumer
of shape-matched selection: the rows it receives (via
`KMC.add_active_events`) are `Refinement`'s successful outcomes for
candidates that `build_refinement_candidates()` already narrowed by
`ShapeID` (Section 6/7) — but `ActiveEventTable` does not store or check
any shape/sid field itself.

## 5. Basin package — accelerated exploration (`pykmc/basins/`)

Shape/sid logic is confined to three files: `basin.py`, `detection.py`,
`exploration.py`. `connectivity.py`, `selection.py`, `exit_time_solver.py`,
and `utils.py` operate one level below — on the basin's transition graph
(state indices, rates, reference-event/symmetry indices) — so their own
inputs are unaffected by how a transition was matched upstream.

### `is_states_has_unknown_environments` — is this state safe to explore

```python
def is_states_has_unknown_environments(self, state) -> tuple[bool, dict[int, ShapeID]]:
    atom_shapes: dict[int, ShapeID] = {}
    for atom_idx, id_initial in enumerate(state.environment.atomic_environment_list):
        if id_initial == "crystal":
            continue
        shape = self.reference_table.shapes.resolve_live_shape(
            state.system, state.neighbors_list, state.environment, atom_idx, mint=False)
        if shape is None:
            return True, atom_shapes     # unknown -> must be handed back to the main loop
        atom_shapes[atom_idx] = shape
    return False, atom_shapes            # every atom resolved -> safe to explore
```

This function exists to decide, before
spending any effort exploring, whether that's actually possible for this
state. `mint=False` is deliberate: a basin must never invent a new shape,
only recognize one that has already catalogued.

**Connects to:** `construct_connexion_table()` calls this for every state
about to be explored, and threads the returned `atom_shapes` straight
through to `explorer.explore(atom_shapes=atom_shapes)` — the explorer
never has to reclassify shapes the caller already resolved.

### `BasinGenericEventExplorer.explore` — which events actually apply here

```python
def explore(self, state, state_index, start_index, atom_shapes: dict[int, ShapeID]):
    row_info_cache = {}
    count = 0
    for atom, row in self.reference_table.live_events(atom_shapes):
        idx_ref = int(row["idx_ref"])
        if idx_ref not in row_info_cache:
            row_info_cache[idx_ref] = self.detector.detect(row, self.reference_table.table,
                                                             self.params.basin.energy_thr)
        is_transient = row_info_cache[idx_ref]
        for sym_idx in range(len(row["sym_matrix"])):
            self.connectivity_table.add_connectivity(
                state=state_index, state_connexion=start_index + count,
                event_connexion=idx_ref, central_atom=atom, sym=sym_idx,
                transient=is_transient, dE_forward=row["dE_forward"], ...)
            count += 1
```

`live_events(atom_shapes)` (Section 4) does all of the actual shape work
here: by the time a `(atom, row)` pair reaches this loop, its shape has
already been confirmed to match exactly, so nothing in `explore()` itself
needs to check shape again. `row_info_cache` exists purely so
`detector.detect()` — an energy-threshold check — runs
once per matched reference-table row and is reused across every atom that
happens to share it, instead of once per `(atom, row)` pair.

**Connects to:** `atom_shapes` comes from `is_states_has_unknown_environments`
(via `construct_connexion_table`) — the same "resolve shape once, reuse
everywhere" pattern as the main loop's
`resolve_new_shapes()` → `build_refinement_candidates()` (Section 7).

### `DetectorThreshold.detect` — following a specific backward pathway

```python
def detect(self, forward_row, table, energy_thr):
    backward_row = table[table["idx_ref"] == forward_row["idx_backward"]].iloc[0]
    shape = ShapeID(backward_row["id_initial"], backward_row["sid_initial"])
    candidates = self._rows_with_shape(table, "id_initial", "sid_initial", shape)
    dE_backward = candidates["dE_forward"].min()
    return (forward_row["dE_forward"] < energy_thr) and (dE_backward < energy_thr)
```

The backward direction is looked up through `idx_backward` — a link
resolved once, at cataloguing time, in `_build_event_series` (Section 4) —
and from that exact row, its `ShapeID` (not just its coarse id) is used to
gather every candidate pathway genuinely sharing that same shape.

### `system_from_state` — applying an already-selected event

```python
def system_from_state(self, from_state, event_idx, central_atom, sym_idx):
    ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx].iloc[0]
    psr_output = PointSetRegistration(self.params, new_system, ref_event,
                                       neighbors_list, central_atom).match()
    # ... apply ref_event's symmetry variant, then psr_output's rotation/
    # translation/permutation, to reconstruct the new state's geometry
```

This function has no shape/sid logic of its own — it looks up a row by
`idx_ref` and reuses `PointSetRegistration.match()` (the same registration
machinery `simple_ira` builds on, Section 3) purely to fit a known event's
geometry onto the current atom. But `event_idx`/`sym_idx` come from
`connectivity_table.get_transition_to_state()`, and that transition only
exists in the connectivity table because `explore()`'s shape-matched
`live_events()` found it in the first place — so which event gets
reconstructed here is entirely a downstream consequence of shape
resolution, even though this function never touches a `ShapeID` itself.

## 6. `EventSearch` / `Reconstruction` / `Refinement` — searching and applying events

None of these three modules contain any `ShapeID`/`sid` logic themselves.
They matter to this document because each one sits directly downstream of
a shape-resolved decision made elsewhere, even though none of them ever
import `ShapeID`.

### `EventSearch` — dispatches ARTn, blind to shape

```python
class EventSearch:
    def execute(self, central_atom_research_list: list[int]):
        for atom_idx in central_atom_research_list:
            self.submit(atom_idx)   # dispatch an ARTn search for this atom
```

`execute()` takes a plain list of atom indices and has no concept of
`ShapeID` at all. What determines *which* atoms end up in that list is
entirely the caller's responsibility.

**Connects to:** `KMC.central_atoms_research()` (Section 7) builds
`central_atom_research_list` by grouping eligible atoms by `ShapeID` and
drawing a fixed number of atoms per shape; `KMC.execute_event_searches()`
passes that list straight through, unchanged. So which atoms get
search dispatched each step is a downstream consequence of shape
resolution — `EventSearch` itself never has to know that.

### `Reconstruction` — applies a selected event, blind to shape

```python
def _reconstruction_active_event(self, idx_selected_event, active_table):
    row = active_table.table.loc[idx_selected_event]
    result = Reconstruction(self.params, self.manager).reconstruct(
        supposed_initial, row["final_configuration"], self.system.configuration,
        self.params.psr.matching_score_thr, neighbors)
    return result
```

`idx_selected_event` is chosen by `KMC._select_event()` from the active
table by rate-based selection alone — the active table (Section 4) carries
no shape/sid field at all, so nothing about *which* event fires this step
is shape-driven. `Reconstruction` is handed plain geometry (positions),
never a shape identity.

### `Refinement` — the point where shape-resolved candidates get verified

```python
class Refinement:
    def execute(self, candidates: list[RefinementCandidate]):
        for candidate in candidates:
            if candidate.verify:
                self.manager.partn_refine(candidate.dfevent, candidate.central_atom_index, ...)
            else:
                # trusted as-is: barrier well above threshold, skip the real ARTn call
                ...
```

`Refinement.execute()` takes a ready-made worklist — it never computes its
own candidates. `RefinementCandidate` carries the full matched
reference-table row (`dfevent`) alongside the atom and symmetry index, so
by the time a candidate reaches this loop, its shape has already been
confirmed (Section 4's `live_events()`); `Refinement` only ever decides
*whether* to spend an search call verifying it, never *whether* it applies.

**Connects to:** `KMC.build_refinement_candidates()` (Section 7) is the
sole producer of this worklist, built from
`ReferenceEventTable.live_events()` (Section 4).

### `event_recycling.py` — orthogonal, but feeds the same worklist

```python
class DistanceRecycling(Recycling):
    def select_recyclable(self, active_table, neighbors_list):
        # keep an active-table row only if its atom hasn't moved far enough
        # to invalidate the cached event -- pure geometry, no shape involved
        ...
```

`DistanceRecycling` decides which previously-active rows survive into the
next step purely from atom displacement — no shape/sid logic at all.

**Connects to:** the rows it keeps become `ActiveEventTable.existing_pairs()`
(Section 4), which `KMC.build_refinement_candidates()` (Section 7) uses to
skip `(atom, event)` pairs that are already active — so a recycling
decision, made with no knowledge of shape, can still suppress what the
shape-matched worklist would otherwise include.

## 7. `KMC` main loop — orchestrating a step (`pykmc/kmc.py`)

This is where every piece from the previous sections gets wired together
into one KMC step.

### `resolve_new_shapes` — resolving every atom's shape before anything else runs

```python
def resolve_new_shapes(self) -> tuple[dict[ShapeID, list[int]], dict[int, ShapeID]]:
    eligible: dict[ShapeID, list[int]] = {}
    atom_shapes: dict[int, ShapeID] = {}
    for atom_idx, id_initial in enumerate(self.atomic_environment.atomic_environment_list):
        if id_initial == "crystal":
            continue
        shape = self.reference_table.shapes.resolve_live_shape(
            self.system, self.neighbors_list, self.atomic_environment, atom_idx, mint=True)
        atom_shapes[atom_idx] = shape
        knowledge = self.reference_table.shapes.get_shape_knowledge(shape)
        if knowledge is None or knowledge.status != "completed":
            eligible.setdefault(shape, []).append(atom_idx)
    return eligible, atom_shapes
```

This is the entry point for everything else in this section, and the
concrete implementation at the top of this document: every
non-crystal atom's real `ShapeID` is resolved (minting a fresh one if
nothing matches) *before* any search, refinement, or basin logic runs this
step. Eligibility for search is answered per exact `ShapeID` — `eligible`
groups atoms by shape.

**Connects to:** `atom_shapes`, the per-atom map, is threaded through the
rest of the step to `build_refinement_candidates()` below and to
`BasinGenericEventExplorer.explore()` (Section 5) — resolved once, reused
everywhere. `eligible`, the per-shape worklist, feeds
`central_atoms_research()`.

### `central_atoms_research` — spending search budget per shape, not per topo_id

```python
def central_atoms_research(self, new_shapes: dict[ShapeID, list[int]], nsearch: int) -> list[int]:
    central_atom_research_list = []
    for atoms in new_shapes.values():          # once per ShapeID, not once per topo_id
        central_atom_research_list += [random.choice(atoms) for _ in range(nsearch)]
    return central_atom_research_list
```

Every shape gets its own `nsearch` draws — this is what actually
enforces the "grouped by `ShapeID`" guarantee `resolve_new_shapes()` sets
up.

**Connects to:** the returned list feeds `EventSearch.execute()`
(Section 6) unchanged. After a fixed-`nsearch` dispatch, every dispatched
atom's shape is re-resolved and marked `completed` regardless of the
search's outcome — there is no continuous per-shape credit in this mode,
so a shape either gets its one round of search this step, or waits for
another atom of the same shape to be drawn on a future step.

### `build_refinement_candidates` — the shape-first worklist for refinement

```python
def build_refinement_candidates(self, atom_shapes: dict[int, ShapeID],
                                 existing_pairs: set[tuple[int, int]] = None) -> list[RefinementCandidate]:
    existing_pairs = existing_pairs or set()
    raw_entries, matched_rows, supposed_ktot = [], [], 0.0
    for atom, row in self.reference_table.live_events(atom_shapes):
        if (atom, int(row["idx_ref"])) in existing_pairs:
            continue                                     # already active, recycled
        matched_rows.append(row)
        for sym_idx in range(len(row["sym_matrix"])):
            raw_entries.append((atom, row, sym_idx))
            supposed_ktot += row["k"]

    e_thr = self._refinement_energy_threshold(matched_rows, supposed_ktot)
    return [RefinementCandidate(central_atom_index=atom, dfevent=row, symmetry_index=sym_idx,
                                 verify=row["dE_forward"] <= e_thr)
            for atom, row, sym_idx in raw_entries]
```

Every candidate this produces has already passed through
`live_events(atom_shapes)` (Section 4) — the exact-`ShapeID` filter.
`existing_pairs`, from `ActiveEventTable` (Section 4/6),
subtracts out pairs a recycling decision already kept active, so the same
pair isn't refined twice.

```python
def _refinement_energy_threshold(self, matched_rows, supposed_ktot) -> float:
    # k_thr is a fraction of the total catalogued rate among rows that
    # actually matched a live atom this step -- narrower than "every row
    # sharing the coarse topo_id", so a shape with no live atom this step
    # can't skew where the verify/trust line gets drawn
    ...
```

**Connects to:** `atom_shapes` comes straight from `resolve_new_shapes()`;
the resulting worklist is handed to `Refinement.execute()` (Section 6).

### `execute_refinements` and `add_active_events` — closing the loop

```python
def execute_refinements(self, atom_shapes, existing_pairs=None) -> Refinement:
    candidates = self.build_refinement_candidates(atom_shapes, existing_pairs)
    refinement = Refinement(self.params, self.system, self.manager)
    refinement.execute(candidates)
    return refinement

def add_active_events(self, events: list[EventRefinementOutput]) -> ActiveEventTable:
    self.active_table.add_events(events)
    return self.active_table
```

`prune_for_recycling()` sets `self.active_table`'s contents at the start
of each step, before `add_active_events` runs: with no recycler attached,
it sets the table to zero rows; with a recycler attached, it keeps exactly
the rows the recycler's own geometric check approves of, further
restricted to rows whose reference event still exists and whose neighbor
list still matches the current step (`drop_stale_rows()`). Whatever rows
survive that are the ones `ActiveEventTable.existing_pairs()` reports, and
`build_refinement_candidates()` uses that to skip refining a pair the
recycler already kept active.

### One step, end to end

```python
def run(self):
    while not done:
        new_shapes, atom_shapes = self.resolve_new_shapes()
        central_atoms = self.central_atoms_research(new_shapes, self.params.eventsearch.nsearch)
        event_search = self.execute_event_searches(central_atoms)
        self.add_reference_events(event_search.get_successes_results())

        refinement = self.execute_refinements(atom_shapes, self.active_table.existing_pairs())
        self.add_active_events(refinement.get_successes_results())

        idx_selected, *_ = self._select_event(self.active_table)   # rate-based, shape-agnostic
        self.reconstruction(self.active_table)
        # apply the selected event, advance simulated time, log

        self.active_table.prune_for_recycling(idx_selected, self.system, configuration_pre, self.reference_table)
        self.active_table.drop_stale_rows(self.neighbors_list)
```

Everything upstream of `_select_event` is shape-aware; everything from
`_select_event` onward (Section 6) operates on plain rates and geometry,
with no further shape logic — by the time an event is selected and
applied, its shape has already done its job.

## 8. Logging — reporting shape-resolved data (`pykmc/info_simulation.py`)

```python
def info_atomic_environments(kmc, new_shapes: dict[ShapeID, list[int]]) -> AtomicEnvironmentInfo:
    return AtomicEnvironmentInfo(n_new_shapes=len(new_shapes), ...)
```

`new_shapes` is exactly `resolve_new_shapes()`'s first return value
(Section 7) — the per-step count of shapes still needing search is a count
of distinct `ShapeID`s, not distinct coarse topo_ids, so this number
reflects the same "search budget is spent per shape" accounting described
in Section 7.

## Notes

- Rethink the way shape matching is used in the first place.
- Probably get rid of the shape-matching criterion in `Refinement.execute()`
  and its basin equivalent.
- A shape id (`sid`) is assigned from the first matching representative
  found during classification, not the best match among all candidates
  sharing the coarse id. If a coarse id has two catalogued shapes and a
  live configuration matches representative `sid=1` first, it is assigned
  `sid=1` even if `sid=2` would have matched more closely.
- Should `is_new_event()` also check the final state's `ShapeID`, not just
  the initial state's and the saddle's?
- The code's column for an event's initial-state topo_id is `id_initial`;
  this document calls it `event_id` throughout, for consistency.
- My code contain much more change than the shape/`sid` mechanism described;
  i dont cover in this document.
