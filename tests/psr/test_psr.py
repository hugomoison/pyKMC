"""Tests for the PointSetRegistration facade.

Uses a single fake, registered strategy rather than IRA so these tests
exercise only the facade's own logic (delegation, create(), the
match_and_check threshold behaviour) independently of any registration
algorithm.
"""

import pytest

from pykmc.psr import PointSetRegistration
from pykmc.psr.result import PSROutput, PSRError
from pykmc.psr.strategies import PSRStrategy
from pykmc.result import Err, ErrorInfo, Ok


class FakeStrategy(PSRStrategy):
    """Reports a fixed matching score, or fails to match if score is None."""

    name = "fake_test_strategy"

    def __init__(self, matching_score: float | None = 0.0):
        self.matching_score = matching_score

    def match(self, nat1, typ1, coords1, nat2, typ2, coords2):
        if self.matching_score is None:
            return Err(ErrorInfo(type=PSRError.NO_MATCH_FOUND, message="no match"))
        return Ok(PSROutput(matching_score=self.matching_score))


def test_match_delegates_to_the_strategy() -> None:
    psr = PointSetRegistration(FakeStrategy(matching_score=0.0))

    result = psr.match(1, ["X"], [], 1, ["X"], [])

    assert result.is_ok()
    assert result.ok_value().matching_score == 0.0


def test_create_wires_the_registered_strategy_by_name() -> None:
    psr = PointSetRegistration.create("fake_test_strategy")

    assert isinstance(psr._strategy, FakeStrategy)
    assert psr.match(1, ["X"], [], 1, ["X"], []).is_ok()


def test_create_with_unknown_style_raises() -> None:
    with pytest.raises(ValueError):
        PointSetRegistration.create("not-a-real-style")


def test_match_and_check_accepts_score_within_threshold() -> None:
    psr = PointSetRegistration(FakeStrategy(matching_score=0.05))

    result = psr.match_and_check(1, ["X"], [], 1, ["X"], [], matching_score_thr=0.1)

    assert result.is_ok()
    assert result.ok_value().matching_score == 0.05


def test_match_and_check_rejects_score_above_threshold() -> None:
    psr = PointSetRegistration(FakeStrategy(matching_score=0.5))

    result = psr.match_and_check(1, ["X"], [], 1, ["X"], [], matching_score_thr=0.1)

    assert not result.is_ok()
    err = result.err_value()
    assert err.type is PSRError.MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD
    assert err.variables["matching_score"] == pytest.approx(0.5)


def test_match_and_check_passes_through_a_failed_match_unchanged() -> None:
    psr = PointSetRegistration(FakeStrategy(matching_score=None))

    result = psr.match_and_check(1, ["X"], [], 1, ["X"], [], matching_score_thr=0.1)

    assert not result.is_ok()
    assert result.err_value().type is PSRError.NO_MATCH_FOUND
