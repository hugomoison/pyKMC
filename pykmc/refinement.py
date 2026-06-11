"""Module implementing the Refinement class that deals with the event refinement procedure."""

from .result import Result, EventRefinementOutput, ErrorInfo, ErrorType, Err, Ok
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
from .config import Config
from .system import System
from .neighbors_list import NeighborsList
from .log import LogKMC
from .atomic_environment import AtomicEnvironment
from .enginemanager.lmpi.pool import Manager
import ase.geometry
import numpy as np
import pandas as pd
import concurrent.futures


class Refinement:
    """Perfrom event refinements and deal with results.

    Parameters
    ----------
    config : Config
        The configuration of the simulation.
    loggers : LogKMC
        The logger of the KMC simulation.
    system : System
        The atomic system.
    neighbors_list : NeighborsList
        The neighbors lists of the system.
    atomic_environment : AtomicEnvironment
        The atomic environment of the system.
    engine : Engine
        The engine to use for the refinement.

    """

    def __init__(
        self,
        config: Config,
        loggers: LogKMC,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment: AtomicEnvironment,
        manager: Manager,
    ) -> None:
        self.config = config
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.atomic_environment = atomic_environment
        self.manager = manager
        self.results = None

    def execute(
        self,
        df_reference_events: pd.DataFrame,
        total_energy,
        existing_pairs: set[tuple[int, int]] | None = None,
    ) -> None:
        """Execute event refinements for each reference event in the df_reference_events dataframe.

        It stores the results of the event refinements in self.results.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe of reference events to refine.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            active event table (carried over from the previous step). These
            are skipped during refinement.

        """
        existing_pairs = existing_pairs or set()
        self.results = []

        total_refinements, supposed_ktot = self.get_total_refinements_todo(df_reference_events)
        e_thr = self.get_energy_thr_refine(df_reference_events, supposed_ktot)
        self.loggers.info("log", "\t :=> Refining {} events".format(total_refinements))

        all_futures = []
        future_context = {}  # mapping future -> contexte

        #Launch all refine Jobs
        for idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            atoms_refine_idx = self.atomic_environment.get_atoms_with_id(dfevent["event_id"])
            ref_idx = int(dfevent["idx_ref"])

            for at_idx in atoms_refine_idx:
                if (at_idx, ref_idx) in existing_pairs:
                    continue
                ###=>refine single generic
                futures = self.refine_single(at_idx, dfevent, total_energy, future_context, e_thr)
                if isinstance(futures,list): #If symmetries
                    all_futures.extend(futures)
                else:
                    all_futures.append(futures)

        #Get results and update values : 
        for f in all_futures:
            #get results
            res = f.result()
            #get specific results info
            ctx = future_context[f]
            _ = future_context.pop(f)

            self.loggers.progress_bar("progress", total_refinements-len(future_context), total_refinements)

            #update result 
            if res.is_ok() : 
                res.ok_value().min2_positions = ctx["min2_positions"]
                res.ok_value().num_reference_event = ctx["num_reference_event"]
                res.ok_value().saddle_positions = res.ok_value().saddle_positions[ctx["neighbors"]]
                #Now check if energy barrier consistent with generic one
                #TODO partn should not return different things depending on AV or not. We get the total energy at the saddle point or dE, but not both.
                #TODO and to be consistent, you should modify res.ok_value().E_saddle.
                if self.config.control.active_volume==True:
                    res.ok_value().dE_forward = res.ok_value().E_saddle
                else:
                    res.ok_value().dE_forward = res.ok_value().E_saddle - total_energy
                res = self.check_refinement_energy(res,abs(res.ok_value().dE_forward- ctx["reference_energy_barrier"]),self.config.eventsearch.refined_energy_thr,)

            else : 
                err = res.err_value()
                if not isinstance(err.variables, dict):
                    err.variables = {}
                err.variables["n_ref_event"] = ctx["num_reference_event"]


            self.results.append(res)

    def refine_single(
        self,
        at_idx: int,
        dfevent: pd.Series,
        total_energy: float,
        future_context: dict, 
        e_thr: float
    ) -> list[Result[EventRefinementOutput, ErrorInfo]]:
        """Perform a single reference event refinement.

        If a reference event has symmetries, it also refine those symmetric events.

        Parameters
        ----------
        at_idx : int
            index of the central atom for which we perform the refinement.
        dfevent : pd.Series
            a Series of the reference event to refine.
        count : int
            number of refinements already done.

        Returns
        -------
        list[Result[EventRefinementOutput, ErrorInfo]]
            list of Result of the refinements.
        """

        ##=>PSR between generic event and at_idx environments
        result_psr = PointSetRegistration(
            self.config, self.system, dfevent, self.neighbors_list, at_idx
        ).match()

        ##=>Check results if match or match < matching_score
        result_psr = check_match(result_psr, self.config.psr.matching_score_thr)
        if not result_psr.is_ok():
            f = concurrent.futures.Future()
            f.set_result(result_psr)
            future_context[f] = {"num_reference_event": dfevent["idx_ref"]}
            return f

        else:

        ##=> Get Saddle positions to refine 

            output_psr = result_psr.ok_value()

            displacement_saddle = dfevent.at["saddle_positions"].copy() - dfevent.at["initial_positions"].copy()
            displacement_final = dfevent.at["final_positions"].copy() - dfevent.at["initial_positions"].copy()

            #all_results = []
            futures = []

            ###=>Apply symmetries 

            current_positions = self.system.positions.copy() #save to restore system after

            for sym_matrix, perm_matrix in zip(dfevent.at["sym_matrix"], dfevent.at["sym_perm"], strict=False):
                
                ###=> Apply symmetries to displacements
                new_displacement_saddle = geometry.transform_positions(displacement_saddle, sym_matrix, 0, perm_matrix)
                new_displacement_final = geometry.transform_positions(displacement_final, sym_matrix, 0, perm_matrix)

                ###=> Get symmetric saddle and final positions
                saddle_positions = dfevent.at["initial_positions"].copy() + new_displacement_saddle
                final_positions = dfevent.at["initial_positions"].copy() + new_displacement_final

                ###=> Apply PSR to the saddle and final positions do get specific saddle and final positions (before refinement)
                new_positions_saddle = geometry.transform_positions(saddle_positions,output_psr.rotation_matrix,output_psr.translation_matrix,output_psr.permutation_matrix)
                new_positions_final = geometry.transform_positions(final_positions, output_psr.rotation_matrix, output_psr.translation_matrix, output_psr.permutation_matrix)
                neighbors = self.neighbors_list.get_neighbors("rcut", at_idx).copy()

                ###=> move the system to the saddle point
                self.system.update_positions(new_positions=new_positions_saddle, atom_idx=neighbors)
                if dfevent.at["energy_barrier"] > e_thr : #We dont refine, we use generic date 
                    #create a fake future to store the result
                    f = concurrent.futures.Future()
                    #TODO I don't like that we don't gibe the same information to E_saddle depending on AV or not
                    f.set_result(Ok(EventRefinementOutput(
                        central_atom_index=at_idx,
                        saddle_positions=self.system.positions.copy(),
                        E_saddle=dfevent["energy_barrier"] if self.config.control.active_volume else total_energy + dfevent["energy_barrier"] ,
                        refined='F'
                    )))

                else : #we refine
                    #TODO : same here, we should send the same information to the partn_refine function 
                    #TODO : when AV, partn_refine needs the minimum positions to compute the initial energy with AV
                    #TODO : but this is the third parameter here, and without AV, the third parameter is the saddle positions. 
                    #TODO : and with AV, you only need to send saddle positions in the rcut, while without we send all saddle positions, this is just too confusing
                    if self.config.control.active_volume==True:
                        # add a job to manager queue
                        f = self.manager.partn_refine(self.config, at_idx,
                                                      current_positions.copy(),
                                                      self.system.cell,
                                                      self.system.types.copy(),
                                                      neighbors.copy(),
                                                      self.system.positions.copy()[neighbors.copy()])  # send copy not reference !
                    else:
                    #add a job to manager queue
                        f = self.manager.partn_refine(self.config, at_idx, self.system.positions.copy(), types=self.system.types.copy()) #send copy not reference !
                futures.append(f)


                #NOTE: TEMPORARY, NEED TO FIND A BETTER WAY
                future_context[f] = {
                    "min2_positions": ase.geometry.wrap_positions(new_positions_final, cell = self.system.cell, pbc=self.system.pbc if self.system.pbc is not None else True),
                    "num_reference_event": dfevent["idx_ref"],
                    "reference_energy_barrier": dfevent["energy_barrier"],
                    "neighbors": neighbors.copy()
                }


                #=> Restore the system to its initial state 
                self.system.update_positions(current_positions)
            return futures



    def check_refinement_energy(
        self,
        result_refine: Result[EventRefinementOutput, ErrorInfo],
        energy_mismatch: float,
        refined_energy_thr: float,
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Check if the energy barrier of the refinement correspond the one of the reference event.

        Parameters
        ----------
        result_refine : Result[EventRefinementOutput, ErrorInfo]
            Results of the refinement procedure.
        energy_mismatch : float
            Difference between the reference event energy barrier and the refine one.
        refined_energy_thr : float
            maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            list of results of the procedure.

        """
        if energy_mismatch > refined_energy_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
                    message="refinement energy barrier does not match reference one",
                )
            )
        else:
            return result_refine

    def get_total_refinements_todo(self, df_reference_events: pd.DataFrame) -> int:
        """Give the total number of refinements to do.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe with reference events to refine.

        Returns
        -------
        int
            total number of refinements to do.

        """
        total = 0
        supposed_ktot = 0
        for _idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            n_atoms = len(
                self.atomic_environment.get_atoms_with_id(dfevent["event_id"])
            ) * len(dfevent["sym_matrix"])
            total += n_atoms 
            supposed_ktot += dfevent.at["k"]*n_atoms
        return total, supposed_ktot
    
    def get_energy_thr_refine(self, df_reference_events, supposed_ktot) : 
        tol = self.config.control.refine_thr
        k_thr = supposed_ktot*tol
        
        #get energy corresponding to the first k value just under k_thr
        mask = df_reference_events["k"] <  k_thr
        if mask.any():
            e_value = df_reference_events.loc[mask].sort_values("k").iloc[-1]["energy_barrier"]
        else: #refine no event
            e_value = 0.0 
        e_value += 0.1 #to be sure want using condition 
        return e_value

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return successful results.

        Returns
        -------
        list[EventRefinementOutput]
            list of EventRefinementOutpout dataclass with refine event's informations.

        """
        return [e.ok_value() for e in self.results if e.is_ok()]
