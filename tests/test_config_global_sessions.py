"""n_global_sessions config field parsing."""
import os

from pykmc import Config

DATA_INI = os.path.join(os.path.dirname(__file__), "data", "input.in")


def test_n_global_sessions_defaults_to_none():
    """input.in does not set the key -> defaults to None (legacy: global on all cores)."""
    config = Config.from_ini_file(DATA_INI)
    assert config.control.n_global_sessions is None


def test_n_global_sessions_parsed_from_ini(tmp_path):
    """Setting n_global_sessions under [Control] is read back as an int."""
    text = open(DATA_INI).read().replace(
        "[Control]", "[Control]\nn_global_sessions = 4", 1
    )
    ini = tmp_path / "in.ini"
    ini.write_text(text)
    config = Config.from_ini_file(str(ini))
    assert config.control.n_global_sessions == 4
