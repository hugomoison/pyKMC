from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import concurrent.futures

import numpy as np
import pandas as pd
import pytest

from pykmc.config import Config
from pykmc.enginemanager.lmpi.lammps_operations import setup_otf_cycle
from pykmc.enginemanager.lmpi.pool import Manager
from pykmc.eventsearch import EventSearch
from pykmc.kmc import KMC
from pykmc.otfml import OTFMLController
from pykmc.refinement import Refinement
from pykmc.result import (
    Err,
    ErrorInfo,
    ErrorType,
    EventRefinementOutput,
    EventSearchOutput,
    Ok,
)


class DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def progress_bar(self, *_args, **_kwargs):
        return None

    def is_enabled_for(self, *_args, **_kwargs):
        return False


class SearchManagerStub:
    def __init__(self, result_batches):
        self.result_batches = list(result_batches)

    def partn_search(self, **_kwargs):
        batch = self.result_batches.pop(0)
        futures = []
        for result in batch:
            future = concurrent.futures.Future()
            future.set_result(result)
            futures.append(future)
        return futures


def build_search_output(atom_index: int, barrier: float) -> Ok:
    positions = np.zeros((3, 3), dtype=float)
    positions[0] = np.array([1.0, 1.0, 1.0])
    return Ok(
        EventSearchOutput(
            central_atom_index=atom_index,
            min1_positions=positions.copy(),
            saddle_positions=positions.copy(),
            min2_positions=positions.copy(),
            dE_forward=barrier,
            dE_backward=barrier + 0.1,
            move_atom_index=0,
        )
    )


def make_otfml_stub(enabled: bool = False) -> OTFMLController:
    """A disabled-by-default OTFMLController, just used for its _collect_extrapolation_retry_ids helper."""
    return OTFMLController(
        SimpleNamespace(
            config=SimpleNamespace(control=SimpleNamespace(otfml=enabled), otfml=None)
        )
    )


@pytest.fixture
def otf_ini_template():
    return """
[Control]
initial_config = ./initial_config.xyz
n_steps = 1
engine = lammps

[Lammps]
pair_style = pair_style_cmd
pair_coeff = pair_coeff_cmd

[AtomicEnvironment]
style = cna/graph
rnei = 2.8
rcut = 6.5

[EventSearch]
style = partn
nsearch = 1

[pARTn]
path_artnso = /tmp/libartn.so

[RateConstant]
style = constant
k0 = 1.0
T = 300.0

[PSR]
style = ira

[IRA]

[OTFML]
retrain_command = true
potential_file = potential.almtp
training_set_file = train.cfg
gamma_tolerance = 1.2
gamma_max = 25.0
enabled_phases = [search, refine, minimize]
"""


def test_otfml_config_parses_from_ini(tmp_path, otf_ini_template):
    ini_path = tmp_path / "input.in"
    ini_path.write_text(otf_ini_template, encoding="utf-8")

    config = Config.from_ini_file(str(ini_path))

    assert config.control.otfml is False
    assert config.otfml.retrain_command == "true"
    assert config.otfml.enabled_phases == ["search", "refine", "minimize"]


def test_otfml_rejects_active_volume(tmp_path, otf_ini_template):
    ini_path = tmp_path / "input.in"
    ini_text = (
        otf_ini_template
        + """
[ActiveVolume]
ract = 8.0
rmov = 4.0
"""
    )
    ini_text = ini_text.replace(
        "engine = lammps", "engine = lammps\nactive_volume = True\notfml = True"
    )
    ini_path.write_text(ini_text, encoding="utf-8")

    with pytest.raises(ValueError, match="OTFML does not support active_volume=True"):
        Config.from_ini_file(str(ini_path))


def test_setup_otf_cycle_reissues_pair_style_and_pair_coeff():
    commands = []
    engine = SimpleNamespace(
        engine_id=7,
        command=commands.append,
        lmp=SimpleNamespace(set_internal_variable=lambda *_: None),
    )
    config = SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="pair_style_cmd",
            pair_coeff="pair_coeff_cmd",
        )
    )

    setup_otf_cycle(engine, config)

    assert commands == [
        "pair_style pair_style_cmd",
        "pair_coeff pair_coeff_cmd",
        "fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1",
    ]


def test_manager_setup_otf_cycle_calls_each_session():
    sessions = [Mock(), Mock()]
    manager = Manager(sessions=sessions, global_session=Mock())
    config = Mock()

    manager.setup_otf_cycle(config)

    for session in sessions:
        session.setup_otf_cycle.assert_called_once_with(config)


def test_event_search_retries_only_extrapolating_jobs():
    system = SimpleNamespace(
        positions=np.zeros((3, 3), dtype=float),
        cell=np.eye(3) * 10.0,
        types=np.array(["Ni", "Ni", "Ni"]),
    )
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        atomicenvironment=SimpleNamespace(rcut=6.5),
    )
    manager = SearchManagerStub(
        [
            [build_search_output(5, 0.2), build_search_output(6, 0.3)],
            [build_search_output(5, 0.9)],
        ]
    )
    event_search = EventSearch(config, system, manager, DummyLogger())
    otfml = make_otfml_stub()

    event_search.execute([5, 6])
    event_search.results[0] = Err(
        ErrorInfo(
            type=ErrorType.EXTRAPOLATION,
            message="search extrapolated",
            variables={"central_atom_index": 5},
        )
    )
    retry_task_ids = otfml._collect_extrapolation_retry_ids(event_search.results)
    assert retry_task_ids == [0]

    event_search.retry(retry_task_ids)

    assert len(event_search.results) == 2
    by_atom = {
        result.ok_value().central_atom_index: result for result in event_search.results
    }
    assert by_atom[5].ok_value().dE_forward == pytest.approx(0.9)
    assert by_atom[6].ok_value().dE_forward == pytest.approx(0.3)


def test_event_search_retry_removes_the_extrapolating_duplicate():
    system = SimpleNamespace(
        positions=np.zeros((3, 3), dtype=float),
        cell=np.eye(3) * 10.0,
        types=np.array(["Ni", "Ni", "Ni"]),
    )
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        atomicenvironment=SimpleNamespace(rcut=6.5),
    )
    manager = SearchManagerStub(
        [
            [
                build_search_output(5, 0.2),
                build_search_output(6, 0.3),
                build_search_output(5, 0.4),
            ],
            [build_search_output(5, 0.9)],
        ]
    )
    event_search = EventSearch(config, system, manager, DummyLogger())
    otfml = make_otfml_stub()

    event_search.execute([5, 6, 5])
    event_search.results[2] = Err(
        ErrorInfo(
            type=ErrorType.EXTRAPOLATION,
            message="search extrapolated",
            variables={"central_atom_index": 5},
        )
    )

    retry_task_ids = otfml._collect_extrapolation_retry_ids(event_search.results)
    event_search.retry(retry_task_ids)

    barriers_for_atom_5 = [
        result.ok_value().dE_forward
        for result in event_search.results
        if result.is_ok() and result.ok_value().central_atom_index == 5
    ]
    assert barriers_for_atom_5 == [pytest.approx(0.2), pytest.approx(0.9)]


def test_refinement_retry_replaces_only_matching_job(monkeypatch):
    """Refinement.retry() re-runs exactly the extrapolating symmetry job and
    leaves the other reference event's result untouched."""
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False, refine_thr=0.9999),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05),
        psr=SimpleNamespace(matching_score_thr=0.1),
    )
    atoms_by_event_id = {1: [10], 2: [11]}
    refinement = Refinement(
        config=config,
        loggers=DummyLogger(),
        system=None,
        neighbors_list=None,
        atomic_environment=SimpleNamespace(
            get_atoms_with_id=lambda event_id: atoms_by_event_id[event_id]
        ),
        manager=None,
    )

    # dfevent's "energy_barrier" is set equal to the eventual E_saddle so
    # check_refinement_energy's mismatch check passes (total_energy=0.0 below).
    dfevent1 = pd.Series(
        {
            "idx_ref": 1,
            "k": 1.0,
            "event_id": 1,
            "energy_barrier": 9.0,
            "sym_matrix": [None],
        }
    )
    dfevent2 = pd.Series(
        {
            "idx_ref": 2,
            "k": 2.0,
            "event_id": 2,
            "energy_barrier": 2.0,
            "sym_matrix": [None],
        }
    )
    df_reference_events = pd.DataFrame([dfevent1, dfevent2])

    call_log = []

    def fake_refine_single(
        at_idx, dfevent, total_energy, future_context, e_thr, only_symmetry_index=None
    ):
        call_log.append((at_idx, int(dfevent["idx_ref"]), only_symmetry_index))
        f = concurrent.futures.Future()
        future_context[f] = {
            "min2_positions": np.zeros((1, 3)),
            "num_reference_event": dfevent["idx_ref"],
            "reference_energy_barrier": dfevent["energy_barrier"],
            "neighbors": np.array([0]),
            "at_idx": at_idx,
            "dfevent": dfevent,
            "symmetry_index": 0,
        }
        if at_idx == 10 and only_symmetry_index is None:
            # First pass: atom 10's job extrapolates.
            f.set_result(
                Err(
                    ErrorInfo(
                        type=ErrorType.EXTRAPOLATION, message="refine extrapolated"
                    )
                )
            )
        else:
            energy = 9.0 if at_idx == 10 else 2.0
            f.set_result(
                Ok(
                    EventRefinementOutput(
                        central_atom_index=at_idx,
                        saddle_positions=np.zeros((1, 3)),
                        E_saddle=energy,
                    )
                )
            )
        return f

    monkeypatch.setattr(refinement, "refine_single", fake_refine_single)
    otfml = make_otfml_stub()

    refinement.execute(df_reference_events, total_energy=0.0)
    retry_task_ids = otfml._collect_extrapolation_retry_ids(refinement.results)
    assert retry_task_ids == [0]

    refinement.retry(retry_task_ids)

    assert len(refinement.results) == 2
    by_atom = {
        r.ok_value().central_atom_index: r for r in refinement.results if r.is_ok()
    }
    assert by_atom[10].ok_value().E_saddle == pytest.approx(9.0)
    assert by_atom[11].ok_value().E_saddle == pytest.approx(2.0)
    # atom 11's job was never resubmitted during retry.
    assert call_log == [(10, 1, None), (11, 2, None), (10, 1, 0)]


def test_otfml_controller_retrains_reloads_and_retries_search(monkeypatch):
    config = SimpleNamespace(
        control=SimpleNamespace(otfml=True),
        otfml=SimpleNamespace(
            retrain_command="true",
            potential_file="potential.almtp",
            training_set_file="train.cfg",
            gamma_tolerance=1.2,
            gamma_max=25.0,
            launcher="nested",
            batch_args=None,
            runner_args="--oversubscribe",
            extra_args=None,
            sequential_eval=False,
            enabled_phases=["search", "refine", "minimize"],
        ),
    )
    session0 = Mock(session_id=1)
    session1 = Mock(session_id=2)
    global_session = Mock(session_id=0)
    manager = SimpleNamespace(
        using_global=False,
        sleeping_workers=Mock(return_value=nullcontext()),
        sessions=[session0, session1],
        setup_otf_cycle=Mock(),
        global_session=global_session,
        use_global=Mock(),
        use_local=Mock(),
        set_all_positions=Mock(),
    )
    kmc = SimpleNamespace(
        config=config,
        manager=manager,
        loggers=DummyLogger(),
        system=SimpleNamespace(positions=np.zeros((2, 3), dtype=float)),
    )
    minimize_mock = Mock()
    kmc._minimize_system = minimize_mock
    monkeypatch.setattr("pykmc.otfml.subprocess.run", Mock())

    event_search = SimpleNamespace(
        results=[
            Err(
                ErrorInfo(
                    type=ErrorType.EXTRAPOLATION,
                    message="search extrapolated",
                    variables={"central_atom_index": 5},
                )
            )
        ],
    )

    def retry(task_ids):
        event_search.results = [build_search_output(5, 0.9)]

    event_search.retry = Mock(side_effect=retry)
    controller = OTFMLController(kmc)

    controller.retry_extrapolating("search", event_search)

    retried_task_ids = event_search.retry.call_args.args[0]
    assert retried_task_ids == [0]
    manager.setup_otf_cycle.assert_called_once_with(config)
    minimize_mock.assert_called_once()
    assert manager.use_local.call_count == 1
    assert manager.use_global.call_count == 0
    manager.set_all_positions.assert_called_once_with(kmc.system.positions)


def test_kmc_minimize_system_is_unchanged_when_otfml_disabled(
    config_system_single_type, monkeypatch
):
    kmc = KMC(config_system_single_type)
    minimize_once = Mock()
    monkeypatch.setattr(kmc, "_minimize_system", minimize_once)

    kmc.minimize_system()

    minimize_once.assert_called_once_with(positions=None, types=None)
