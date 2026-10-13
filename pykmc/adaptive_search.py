"""Count-weighted adaptive event-search stopping/continuation rules.

Pure, per-shape bookkeeping and decision logic used by
``KMC.adaptive_event_search`` to decide, as each individual ARTn search
result streams in, whether a given shape needs more searching or has
converged. Kept independent of pandas/MPI/engine machinery so the estimator
math is unit-testable on its own.

Three selectable rules (``params.eventsearch.adaptive_stopping_rule``), all
built on the shape's own *discovery frequency* (how often each cataloged
event has been (re)matched so far) rather than its physical rate constant --
draws are sampled proportional to how often an event has actually been
rediscovered, so frequency is the quantity with a valid per-draw probability
interpretation under this module's resampling process, unlike rate:

- ``"q_sprt"`` (default): a formal Wald sequential probability ratio test on
  the dry spell since the shape's last genuinely new discovery. Tests
  H0 (a per-draw discovery probability of `adaptive_epsilon` still remains)
  against H1 (nothing left, probability 0); converges once that dry spell
  crosses the analytically-derived boundary. No separate estimator at all --
  pure discovery-timing.
- ``"q_qmin"``: tracks the Good-Turing count-based undiscovered-mass
  fraction (singleton events / all discovered), same as `"q_sprt"`'s
  threshold in spirit, but the confirmation window is *dynamic*: it is
  derived every draw from the shape's own smallest currently-observed
  discovery frequency (capped at `adaptive_epsilon`) rather than a flat
  constant, so shapes whose evidence is thinner are held to a longer wait
  automatically. The most accurate of the three in testing, at the highest
  cost.
- ``"q_se"``: the same count-based fraction, but confirmed via a statistical
  confidence bound (fraction + Z*SE(fraction) <= epsilon) instead of a
  waiting window -- converges the instant that bound is satisfied, no
  separate confirmation period. The cheapest of the three, at a real cost in
  per-shape reliability (see `as-diagnostic/RESEARCH_CONTEXT.md`).

All three share `_MISS_PROBABILITY`, the one confidence constant every
derived threshold in this module is built from -- `adaptive_epsilon` is the
only other free parameter; nothing here is a second, independently-tuned
knob.

Orthogonal to which rule is chosen, ``params.eventsearch.adaptive_scope``
picks which discovered events a rule's own estimator is built from:

- ``"all"`` (default): every discovered event counts, as described above.
- ``"important"``: each rule's own estimator (the singleton-count fraction
  for `"q_qmin"`/`"q_se"`, the "was the latest discovery genuinely new" test
  for `"q_sprt"`) is restricted to the shape's own currently-important
  events -- the smallest leading group, ranked by rate descending, whose
  cumulative rate covers `params.control.refine_thr` of the shape's own
  currently-discovered total (see `_important_idx_refs` -- the same
  cumulative-rate-coverage threshold `KMC.execute_refinements` uses to
  decide which candidates need real ARTn verification). A shape can then
  converge without ever confirming its own negligible-rate long tail, at a
  real cost: a discovered-but-unconfirmed tail event is invisible to the
  estimator by construction, so `"important"` trades away exactly the
  guarantee `"all"` provides. Membership is recomputed fresh from current
  knowledge every call, not fixed once -- a later, larger discovery can
  retroactively push a smaller, previously-important event back out.

Because of this, every function/method below that used to take just
``EventSearchParameters`` now takes the top-level ``Parameters`` instead, so
both its `.eventsearch` and `.control.refine_thr` are reachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, NamedTuple

from .result import EventSearchOutput, Result, ShapeID
from .shape_table import ShapeKnowledge

# The one confidence constant every derived threshold below is built from:
# the accepted chance of still being wrong (about a hidden event, or about a
# dry spell meaning convergence) after that threshold has been met.
_MISS_PROBABILITY = 0.05
_REQUIRED_DRAWS_PER_EPSILON = math.log(1.0 / _MISS_PROBABILITY)

# norm.ppf(1 - _MISS_PROBABILITY) for _MISS_PROBABILITY=0.05, i.e. the
# one-sided z-score for a (1 - _MISS_PROBABILITY) confidence bound. Hardcoded
# rather than computed via scipy.stats.norm to keep this module's only
# dependency on the standard library, matching its existing minimal-
# dependency design; recompute this if `_MISS_PROBABILITY` ever changes.
_Z_SCORE = 1.6448536269514722

# Fraction of adaptive_max_searches at which a still-open shape is flagged
# "escalated" (a log-visible signal that it's taking a while), derived from
# the single ceiling rather than exposed as its own soft-cap knob.
_ESCALATION_FRACTION_OF_MAX_SEARCHES = 0.5


@dataclass
class ShapeSearchStats:
    """Per-`ShapeID`, per-KMC-step adaptive search bookkeeping.

    Every atom dispatch draws from is already resolved to its exact
    `ShapeID` before searching (`KMC.resolve_new_shapes()`'s eager per-atom
    classification), so one `AdaptiveSearchSession` shape maps to exactly one
    dispatch pool -- there is no coarse `id` collision to guard against here;
    a search issued for this shape is always a genuine draw from this
    shape's own event distribution.

    Lives only for the duration of one step's adaptive dispatch loop; not
    persisted across steps.

    Attributes
    ----------
    shape : ShapeID
        The shape this struct tracks.
    n_dispatched : int
        This shape's own count of ARTn searches issued so far this session,
        whether or not they produced a candidate event. Drives the cap and
        escalation checks below.
    n_valid_draws : int
        This shape's own count: results actually attributed to *this*
        `ShapeID`'s bucket whose outcome reached the new/duplicate decision
        (`is_valid_new_event`'s `Ok` or `Err(EVENT_NOT_NEW)`) -- i.e. a
        genuine draw from this shape's own event distribution.
        Energy/asymmetry-filtered searches are not evidence and are excluded.
    rediscovery_counts : dict[int, int]
        idx_ref -> number of times this cataloged event has been (re)matched
        as belonging to this shape, including the first ("new") occurrence
        (starts at 1). Every rule's own estimator is built from this alone
        under `adaptive_scope="all"`.
    known_rates : dict[int, float]
        idx_ref -> rate constant k_i, at first (re)discovery. Not read under
        `adaptive_scope="all"` (every rule's estimator is discovery-frequency
        based, not rate-based); under `"important"`, ranks discovered events
        to pick out the coverage-based important subset (see
        `_important_idx_refs`).
    last_reset_at_draws : int
        This shape's `n_valid_draws` as of the most recent point its own
        rule-specific evidence was reset (a fresh, in-scope discovery for
        `"q_sprt"`, the count-based fraction crossing back above epsilon for
        `"q_qmin"`; unused by `"q_se"`, which has no waiting window at all).
        Convergence requires enough valid draws since this point.
    last_s_obs : int
        This shape's own `len(rediscovery_counts)` as of the last
        `step_stats` call, used by `"q_sprt"` to detect a fresh discovery
        (`len(rediscovery_counts)` has grown) without needing the specific
        `idx_ref` of the triggering draw threaded through the call.
    confirming : bool
        Whether this shape is currently past its own rule-specific reset
        point, i.e. actively waiting out a confirmation window rather than
        still accumulating fresh evidence -- read by `pick_next` to decide
        dispatch priority. Always `False` for `"q_se"` (no waiting window).
    escalated : bool
        True once this shape's own `n_dispatched` has crossed half of
        `adaptive_max_searches` while it's still open -- it's being searched
        beyond its normal pace, up to the ceiling, before finally giving up.
    state : Literal["open", "converged", "capped", "skipped"]
        This shape's own terminal outcome. "open" while still being
        searched; terminal otherwise. "capped" means this shape was still
        open when its own dispatch budget hit `adaptive_max_searches`;
        "skipped" means it never had any eligible atom to search in the
        first place (e.g. all its atoms are inactive) -- distinct from
        "capped" so a shape that was never searched at all doesn't trigger
        the same "may still have undiscovered events" warning as one that
        genuinely exhausted its own budget.
    last_fraction : float | None
        This shape's most recently computed count-based undiscovered-mass
        fraction, in whichever scope `adaptive_scope` selects (for logging)
        -- `None` until at least 2 distinct events have been discovered
        (`"q_sprt"`/`"q_qmin"`) or the bound has ever been evaluated
        (`"q_se"`).

    """

    shape: ShapeID
    n_dispatched: int = 0
    n_valid_draws: int = 0
    rediscovery_counts: dict[int, int] = field(default_factory=dict)
    known_rates: dict[int, float] = field(default_factory=dict)
    last_reset_at_draws: int = 0
    last_s_obs: int = 0
    confirming: bool = False
    escalated: bool = False
    state: Literal["open", "converged", "capped", "skipped"] = "open"
    last_fraction: float | None = None


def seed_stats(know: ShapeKnowledge, shape: ShapeID) -> ShapeSearchStats:
    """Build an ephemeral `ShapeSearchStats` seeded from `know`, `shape`'s persistent record.

    Carries forward `rediscovery_counts`/`known_rates` (and the matching
    `n_valid_draws`, since every recorded sighting -- opportunistic or
    dispatched -- is exactly one valid draw by construction) so a fresh
    session doesn't discard prior evidence; the per-session pacing fields
    (`n_dispatched`, `escalated`, `state`, `last_reset_at_draws`, `last_s_obs`,
    `confirming`) start blank regardless, since they describe this session's
    own dispatch budget, not this shape's lifetime knowledge.

    A free function rather than a `ShapeKnowledge` method: `ShapeKnowledge`
    lives in `pykmc.shape_table`, which this module depends on (for the
    persistent record `AdaptiveSearchSession` is seeded from) -- not the
    other way around, so `ShapeKnowledge` itself carries no knowledge of
    `ShapeSearchStats`.
    """
    return ShapeSearchStats(
        shape=shape,
        n_valid_draws=sum(know.rediscovery_counts.values()),
        rediscovery_counts=dict(know.rediscovery_counts),
        known_rates=dict(know.known_rates),
    )


def record_draw(stats: ShapeSearchStats, idx_ref: int, k: float) -> None:
    """Record one valid draw -- `idx_ref` (re)discovered at rate `k` -- for one shape's tally.

    The caller (`KMC.adaptive_event_search`, via
    `ReferenceEventTable.resolve_forward_and_backward_rows`) has already
    resolved `idx_ref`/`k` and confirmed they genuinely belong to this
    shape's own distribution -- this function only updates the tally, it
    doesn't decide relevance.

    Parameters
    ----------
    stats : ShapeSearchStats
        The shape's running tally, updated in place.
    idx_ref : int
        The cataloged event (re)discovered by this draw.
    k : float
        Its rate constant.

    """
    stats.n_valid_draws += 1
    stats.rediscovery_counts[idx_ref] = stats.rediscovery_counts.get(idx_ref, 0) + 1
    stats.known_rates[idx_ref] = k


def _important_idx_refs(stats: ShapeSearchStats, coverage: float) -> set[int]:
    """The shape's own currently-important discovered events, by cumulative rate coverage.

    The smallest leading group, ranked by rate descending, whose cumulative
    rate reaches `coverage` of the shape's own currently-discovered total --
    an ABC/PCA-style "top contributors covering X%" set, not a fixed
    absolute rate cutoff. Recomputed fresh every call from current knowledge
    alone (never the shape's eventual, still-unknown full event set), so
    membership is not monotonic: a smaller event already counted as
    important can be pushed back out once a big enough new event is
    discovered and shifts the ranking.
    """
    ranked = sorted(stats.rediscovery_counts, key=lambda idx: stats.known_rates[idx], reverse=True)
    total = sum(stats.known_rates[idx] for idx in ranked)
    important: set[int] = set()
    cumulative = 0.0
    for idx in ranked:
        important.add(idx)
        cumulative += stats.known_rates[idx]
        if cumulative >= coverage * total:
            break
    return important


def _scoped_counts(stats: ShapeSearchStats, scope: str, coverage: float) -> dict[int, int]:
    """The rediscovery counts a rule's own estimator should be built from under `scope`."""
    if scope == "all":
        return stats.rediscovery_counts
    important = _important_idx_refs(stats, coverage)
    return {idx: count for idx, count in stats.rediscovery_counts.items() if idx in important}


def _sprt_threshold_draws(epsilon: float) -> int:
    """Wald SPRT boundary for H0 (per-draw new-discovery probability = epsilon) vs H1 (= 0)."""
    accept_h1_at = (1 - _MISS_PROBABILITY) / _MISS_PROBABILITY
    return math.ceil(math.log(accept_h1_at) / -math.log(1 - epsilon))


def _count_fraction(counts: dict[int, int]) -> float:
    """Good-Turing count-based undiscovered-mass fraction: currently-singleton events / total occurrences in `counts`."""
    f1 = sum(1 for count in counts.values() if count == 1)
    return f1 / sum(counts.values())


def _qmin_patience(counts: dict[int, int], epsilon: float) -> int:
    """Dynamic `"q_qmin"` confirmation window: derived from the smallest observed discovery frequency in `counts`.

    Recomputed every draw since it drifts even without new discoveries
    (every other event's own relative frequency within `counts` dilutes as
    its total grows). Capped at `epsilon`: a shape that has demonstrably
    produced something rarer than epsilon should never get a *shorter* wait
    than the flat `epsilon`-only bound would give it, only a longer one when
    its own evidence warrants it.
    """
    t = sum(counts.values())
    q_min = min(count / t for count in counts.values())
    q_min = min(q_min, epsilon)
    return math.ceil(_REQUIRED_DRAWS_PER_EPSILON / q_min)


def _step_q_sprt(stats: ShapeSearchStats, epsilon: float, scope: str, coverage: float) -> bool:
    s_obs = len(stats.rediscovery_counts)
    if s_obs > stats.last_s_obs:
        newest = next(reversed(stats.rediscovery_counts))
        if scope == "all" or newest in _important_idx_refs(stats, coverage):
            stats.last_reset_at_draws = stats.n_valid_draws
    stats.last_s_obs = s_obs
    threshold = _sprt_threshold_draws(epsilon)
    gap = stats.n_valid_draws - stats.last_reset_at_draws
    stats.last_fraction = gap / threshold
    stats.confirming = gap > 0
    if gap >= threshold:
        stats.state = "converged"
        return False
    return True


def _step_q_qmin(stats: ShapeSearchStats, epsilon: float, scope: str, coverage: float) -> bool:
    if len(stats.rediscovery_counts) < 2:
        stats.confirming = False
        return True
    counts = _scoped_counts(stats, scope, coverage)
    fraction = _count_fraction(counts)
    stats.last_fraction = fraction
    if fraction > epsilon:
        stats.last_reset_at_draws = stats.n_valid_draws
        stats.confirming = False
        return True
    stats.confirming = True
    patience = _qmin_patience(counts, epsilon)
    if stats.n_valid_draws - stats.last_reset_at_draws >= patience:
        stats.state = "converged"
        return False
    return True


def _step_q_se(stats: ShapeSearchStats, epsilon: float, scope: str, coverage: float) -> bool:
    if len(stats.rediscovery_counts) < 2:
        stats.confirming = False
        return True
    counts = _scoped_counts(stats, scope, coverage)
    fraction = _count_fraction(counts)
    f1 = sum(1 for count in counts.values() if count == 1)
    se = math.sqrt(f1) / sum(counts.values())
    stats.last_fraction = fraction
    stats.confirming = False  # instantaneous test only -- no separate waiting window
    if fraction + _Z_SCORE * se <= epsilon:
        stats.state = "converged"
        return False
    return True


_STEP_BY_RULE: dict[str, Callable[[ShapeSearchStats, float, str, float], bool]] = {
    "q_sprt": _step_q_sprt,
    "q_qmin": _step_q_qmin,
    "q_se": _step_q_se,
}


def remaining_demand(stats: ShapeSearchStats, params) -> int:
    """How many more valid draws a confirming shape needs before it can converge.

    Only meaningful for a shape whose `confirming` is already `True` -- it
    is the exact remaining size of the confirmation window (dynamic for
    `"q_qmin"`, fixed for `"q_sprt"`), not an estimate, used by `pick_next`
    (net of each shape's own in-flight count) so a confirming shape close to
    converging is never handed more searches than it actually still needs.
    `"q_se"` has no waiting window at all (it converges the instant its
    bound is satisfied), so this always returns the defensive floor for it.

    Parameters
    ----------
    stats : ShapeSearchStats
        The shape's running tally.
    params : Parameters
        Needs `.eventsearch.adaptive_epsilon`, `.eventsearch.adaptive_stopping_rule`,
        `.eventsearch.adaptive_scope`, `.control.refine_thr`.

    Returns
    -------
    int
        At least 1 (defensive floor; a shape whose gap already meets the
        requirement should have been converged by `step_stats` before
        `pick_next` is ever asked about it).

    """
    ep = params.eventsearch
    rule = ep.adaptive_stopping_rule
    if rule == "q_se":
        return 1
    if rule == "q_sprt":
        required = _sprt_threshold_draws(ep.adaptive_epsilon)
    elif rule == "q_qmin":
        counts = _scoped_counts(stats, ep.adaptive_scope, params.control.refine_thr)
        required = _qmin_patience(counts, ep.adaptive_epsilon)
    else:
        raise ValueError(f"unknown adaptive_stopping_rule {rule!r}")
    gap = stats.n_valid_draws - stats.last_reset_at_draws
    return max(1, required - gap)


def step_stats(stats: ShapeSearchStats, params) -> bool:
    """Advance one shape's state by one result; return whether to keep searching it.

    State machine:
    1. If `n_dispatched >= adaptive_max_searches` -> `state="capped"`, stop.
    2. If `n_dispatched` has crossed half of `adaptive_max_searches` and not
       yet flagged -> set `escalated=True` (the caller logs this); does NOT
       stop the loop.
    3. If no valid draws yet -> keep going (bounded by step 1).
    4. Else dispatch to whichever rule `adaptive_stopping_rule` names
       (`"q_sprt"`, `"q_qmin"`, or `"q_se"` -- see the module docstring for
       what each one does), scoped to `adaptive_scope` (`"all"` or
       `"important"`, the latter using `control.refine_thr`).

    Parameters
    ----------
    stats : ShapeSearchStats
        The shape's running tally, updated in place.
    params : Parameters
        Needs `.eventsearch.adaptive_max_searches`, `.eventsearch.adaptive_epsilon`,
        `.eventsearch.adaptive_stopping_rule`, `.eventsearch.adaptive_scope`,
        `.control.refine_thr`.

    Returns
    -------
    bool
        True if this shape should be searched again.

    """
    if stats.state != "open":
        return False

    ep = params.eventsearch

    if stats.n_dispatched >= ep.adaptive_max_searches:
        stats.state = "capped"
        return False

    escalation_threshold = ep.adaptive_max_searches * _ESCALATION_FRACTION_OF_MAX_SEARCHES
    if stats.n_dispatched >= escalation_threshold and not stats.escalated:
        stats.escalated = True

    if stats.n_valid_draws == 0:
        return True

    try:
        step = _STEP_BY_RULE[ep.adaptive_stopping_rule]
    except KeyError:
        raise ValueError(f"unknown adaptive_stopping_rule {ep.adaptive_stopping_rule!r}") from None
    return step(stats, ep.adaptive_epsilon, ep.adaptive_scope, params.control.refine_thr)


class AdaptiveSearchSession:
    """Own the per-`ShapeID` state machine for one KMC step's adaptive search.

    `KMC.adaptive_event_search` dispatches searches and reports results here
    as they stream in; this class tracks which shapes remain dispatchable,
    their rediscovery tallies, and escalation, so that state machine lives in
    one place instead of being managed inline in `KMC`.

    Every shape dispatched here is already resolved before this session is
    built (`KMC.resolve_new_shapes()`'s eager per-atom classification), so
    a pick samples atoms already known to belong to the shape being
    searched -- there is no shared coarse pool a draw might land on
    unpredictably, and no other shape's history to keep separate from this
    one's.
    """

    def __init__(
        self,
        shapes: list[ShapeID],
        knowledge: dict[ShapeID, ShapeKnowledge] | None = None,
    ) -> None:
        knowledge = knowledge or {}
        self.stats: dict[ShapeID, ShapeSearchStats] = {
            shape: seed_stats(knowledge.get(shape, ShapeKnowledge()), shape) for shape in shapes
        }
        self.open_shapes: set[ShapeID] = set(shapes)
        self._round_index = 0
        # Dispatched but not yet resolved (advance_one'd) draws, per shape --
        # lets pick_next cap a confirming shape at its own exact remaining
        # need instead of over-dispatching a burst of redundant searches
        # while streaming keeps refilling the pool ahead of any one result
        # landing. Only ever holds non-negative counts.
        self._in_flight: dict[ShapeID, int] = {}

    def mark_no_atoms(self, shape: ShapeID) -> None:
        """Mark `shape` "skipped": no atoms left to search it with.

        Distinct from "capped" -- this shape was never actually searched, so
        it must not count toward the "hit the ceiling without converging"
        warning.
        """
        self.stats[shape].state = "skipped"
        self.open_shapes.discard(shape)

    def record_dispatch(self, shape: ShapeID) -> None:
        """Count one search as issued against `shape`."""
        self.stats[shape].n_dispatched += 1
        self._in_flight[shape] = self._in_flight.get(shape, 0) + 1

    def remaining_budget(self, shape: ShapeID, params) -> int:
        """Searches left before `shape` hits its cap."""
        return max(0, params.eventsearch.adaptive_max_searches - self.stats[shape].n_dispatched)

    def record_result(self, shape: ShapeID, idx_ref: int, k: float) -> None:
        """Update `shape`'s tally from one resolved sighting.

        `shape` is the outcome's actual `ShapeID`, `idx_ref`/`k` the
        cataloged event it resolved to (both already resolved by the caller
        via `ReferenceEventTable.resolve_forward_and_backward_rows`) -- not
        necessarily the shape a search was dispatched for, since a draw's
        outcome can turn out to confirm a different, already-known shape.
        The bucket is created lazily so a shape sighted for the first time
        mid-session (never part of this session's own dispatch pool) still
        gets tracked.
        """
        if shape not in self.stats:
            self.stats[shape] = ShapeSearchStats(shape=shape)
        record_draw(self.stats[shape], idx_ref, k)

    def credit(self, resolved_shape: ShapeID, fwd_row, bwd_row) -> None:
        """Record `fwd_row`/`bwd_row` against `resolved_shape`'s tally, if either one is actually its own row.

        `fwd_row`/`bwd_row` are one outcome's forward/backward catalogue
        rows (from `ReferenceEventTable.resolve_forward_and_backward_rows`);
        `resolved_shape` is the live atom's own current `ShapeID`. Neither
        row matching means this outcome is real knowledge about some other
        shape, not this one -- see `record_result`.
        """
        if ShapeID(fwd_row["id_initial"], int(fwd_row["sid_initial"])) == resolved_shape:
            self.record_result(resolved_shape, int(fwd_row["idx_ref"]), float(fwd_row["k"]))
        elif ShapeID(bwd_row["id_initial"], int(bwd_row["sid_initial"])) == resolved_shape:
            self.record_result(resolved_shape, int(bwd_row["idx_ref"]), float(bwd_row["k"]))

    def pick_next(self, dispatchable: list[ShapeID], params) -> ShapeID | None:
        """Choose the single next shape to dispatch a search for.

        A shape is "confirming" if it's already past its own rule-specific
        reset point (see `ShapeSearchStats.confirming`) *and* it still has
        remaining confirmation demand net of its own currently in-flight
        (dispatched, not yet `advance_one`'d) draws -- so a shape one draw
        away from converging gets exactly the searches it still needs, never
        a whole pool-refill burst of redundant ones, and once its demand is
        fully covered by in-flight draws it drops out of this tier (freeing
        the pool for other shapes) until one of those draws resolves. A
        shape not yet confirming is "uncertain" -- picked only once no
        confirming shape remains eligible. Ties within either tier rotate
        via `self._round_index` so repeated calls don't always favor the
        same shape.

        Parameters
        ----------
        dispatchable : list[ShapeID]
            Shapes eligible to be searched (already filtered for having at
            least one eligible atom by the caller).
        params : Parameters
            Needs `.eventsearch.adaptive_epsilon`, `.eventsearch.adaptive_stopping_rule`,
            `.eventsearch.adaptive_scope`, `.control.refine_thr`.

        Returns
        -------
        ShapeID | None
            The next shape to dispatch a search for, or `None` if nothing in
            `dispatchable` still has budget left to search.

        """
        eligible = [shape for shape in dispatchable if self.remaining_budget(shape, params) > 0]
        if not eligible:
            return None

        confirming = [
            shape
            for shape in eligible
            if self.stats[shape].confirming
            and remaining_demand(self.stats[shape], params) > self._in_flight.get(shape, 0)
        ]
        pool = confirming or eligible
        self._round_index += 1
        return pool[self._round_index % len(pool)]

    def advance_one(self, shape: ShapeID, params) -> bool:
        """Advance `shape`'s state by one result.

        Called exactly once per dispatched draw, once its outcome has been
        credited (or found not to apply) to `shape` -- including a search
        that produced no usable evidence at all, so a shape still gets
        checked against its own dispatch cap even on a string of pure
        failures. Also releases this draw's own in-flight slot (see
        `record_dispatch`), so `pick_next` sees this shape's remaining
        confirmation demand as available again.

        Parameters
        ----------
        shape : ShapeID
            The shape to re-evaluate.
        params : Parameters
            Needs `.eventsearch.adaptive_max_searches`, `.eventsearch.adaptive_epsilon`,
            `.eventsearch.adaptive_stopping_rule`, `.eventsearch.adaptive_scope`,
            `.control.refine_thr`.

        Returns
        -------
        bool
            True iff this call is the moment `shape` crossed into escalation
            (for the caller to log) -- `False` otherwise, including every
            later call once it's already flagged.

        """
        self._in_flight[shape] = max(0, self._in_flight.get(shape, 0) - 1)
        stats = self.stats[shape]
        was_escalated = stats.escalated
        still_open = step_stats(stats, params)
        if not still_open:
            self.open_shapes.discard(shape)
        return not was_escalated and stats.escalated

    def finalize(self, mark_completed: Callable[[ShapeID], None]) -> None:
        """Write back terminal per-shape outcomes to the persistent record.

        Call once the dispatch loop has fully drained (`open_shapes`
        empty). Only "converged"/"capped" -- a shape that actually went
        through its own dispatched session to a real stopping point -- earns
        persistent `"completed"` status. "skipped" (never had an eligible
        atom this step) and "open" (shouldn't occur once drained) leave the
        persistent status untouched, so the shape is reconsidered whenever
        it does become live/dispatchable.

        `mark_completed` (typically `ShapeTable.mark_shape_completed`) is
        taken as a callback rather than the raw `shape_knowledge` dict so
        this module never has to touch that dict's representation directly.
        """
        for shape, stats in self.stats.items():
            if stats.state in ("converged", "capped"):
                mark_completed(shape)


class AdaptiveSearchResult(NamedTuple):
    """Return value of `KMC.adaptive_event_search`.

    A NamedTuple rather than a plain tuple so a call site can use either
    positional unpacking or named attribute access.

    Attributes
    ----------
    search_results : list[EventSearchOutput]
        Flattened successful search outputs across the whole session.
    raw_results : list[Result]
        Raw per-search Result list (successes and failures), for
        `get_info_reference_event_searches`.
    valid_results : list[Result]
        `is_valid_new_event` results, same order/length as `search_results`.
    stats : dict[ShapeID, ShapeSearchStats]
        Final per-shape stats, for `get_info_adaptive_search`.

    """

    search_results: list[EventSearchOutput]
    raw_results: list[Result]
    valid_results: list[Result]
    stats: dict[ShapeID, ShapeSearchStats]
