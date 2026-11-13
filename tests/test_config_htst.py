"""Tests for the [RateConstant] keys of style = htst and the [ActiveVolume] radii."""

import pytest
from pydantic import ValidationError

from pykmc.config import ActiveVolume, RateConstantConfig


def test_htst_defaults() -> None:
    config = RateConstantConfig(style="htst")

    assert config.free_radius == 8.0
    assert config.shell_width is None
    assert config.hessian_dir is None
    assert config.fd_step == 0.01
    assert config.max_full_hessian_mb == 1000.0


@pytest.mark.parametrize(
    ("value", "expected"), [("full", "full"), (" FULL ", "full"), ("6.5", 6.5)]
)
def test_free_radius_is_a_radius_or_full(value: str, expected: object) -> None:
    assert RateConstantConfig(style="htst", free_radius=value).free_radius == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "all"])
def test_free_radius_refuses_other_values(value: str) -> None:
    with pytest.raises(ValidationError, match="free_radius must be a positive"):
        RateConstantConfig(style="htst", free_radius=value)


def test_k0_in_hz_is_refused_under_htst_only() -> None:
    assert RateConstantConfig(style="constant", k0=1e13).k0 == 1e13
    with pytest.raises(ValidationError, match="k0 is in ps\\^-1"):
        RateConstantConfig(style="htst", k0=1e13)


@pytest.mark.parametrize("key", ["premin", "nu0_zone_radius", "nu0_min_THz"])
def test_removed_keys_are_refused_by_name(key: str) -> None:
    with pytest.raises(ValidationError, match=f"{key} is no longer supported"):
        RateConstantConfig(style="htst", **{key: "1"})
    with pytest.raises(ValidationError, match="style = rpa is no longer supported"):
        RateConstantConfig(style="rpa")


def test_active_volume_refuses_rmov_above_ract() -> None:
    assert ActiveVolume(ract=6.0, rmov=6.0).rmov == 6.0
    with pytest.raises(ValidationError, match="rmov = 7 A must not be larger"):
        ActiveVolume(ract=6.0, rmov=7.0)
