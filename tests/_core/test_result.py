"""Tests for pykmc._core.result."""

from enum import auto

from pykmc._core.result import Err, ErrorCode, Ok


class PSRError(ErrorCode):
    """Stand-in for a module-declared enumeration."""

    NOT_FOUND = auto()


class BasinError(ErrorCode):
    """Deliberately reuses a member name declared by `PSRError`."""

    NOT_FOUND = auto()


def test_ok_wraps_its_value():
    assert Ok(42).ok_value() == 42
    assert Ok(42).is_ok() is True


def test_err_wraps_its_error():
    info = object()
    assert Err(info).err_value() is info
    assert Err(info).is_ok() is False


def test_key_is_prefixed_by_the_owning_enum():
    assert PSRError.NOT_FOUND.key == "PSRError.NOT_FOUND"
