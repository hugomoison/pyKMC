"""Contain a LogManager for handling multiple loggers and a LogKMC class tailored for KMC-specific logging configured with the LOGGING_CONFIG dictionary.

It also contains custom handlers/formatters for diverse console and file output
"""

import sys
import logging
import logging.config
from typing import Any, ClassVar, TextIO
from enum import Enum
import re
from .parameters import Parameters

DISPLAYED_HASH_LENGTH = 8


def fmt_hash(value: str | None, length: int = DISPLAYED_HASH_LENGTH) -> str:
    """Return the human-readable hash prefix used in log output."""
    if value is None:
        return "?"
    return value[:length]


class LogManager:
    """Manage the configuration and usage of multiple standard Python loggers.

    It is setup via dictionary configuration and provides convenience
    methods for sending messagers to specific loggers.

    Attributes
    ----------
    config_dict (dict[str, Any], optional):
        Configuration dictionary for loggers in the format expected by logging.config.dictConfig.
        No configuration is applied if None.

    """

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        self._logger = {}  # Stores logger instances by name
        # Configure loggers
        if config_dict is not None:
            self.configure_from_dict(config_dict)

    def configure_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Configure logger instances from a dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary with the loggers settings.

        Raises
        ------
        Exception
            If loggers configuration fails.

        """
        try:
            logging.config.dictConfig(config_dict)
            # Retrieve and store configured loggers
            for logger_name in config_dict.get("loggers", {}):
                self._logger[logger_name] = logging.getLogger(logger_name)

        except Exception as e:
            raise Exception("Unable to setup loggers from dictionary") from e

    def _get_active_logger(self, logger_name: str) -> logging.Logger:
        """Retrieve a logger instance by its name.

        Parameters
        ----------
        logger_name : str
            Logger name.

        Returns
        -------
        logging.Logger
            The requested logger instance.

        Raises
        ------
        Exception
            ValueError: If the logger name is not configured within this LogManager.

        """
        if logger_name not in self._logger:
            raise ValueError(f"Logger '{logger_name}' is not configured.")
        else:
            logger = self._logger.get(logger_name)

        return logger

    def debug(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a message with level DEBUG using the specified logger.

        This is a convenience method that wraps the underlying `logging.Logger.debug()` call.
        Refer to `logging.Logger.debug()` documentation for `*args` and `**kwargs` usage.

        Parameters
        ----------
        logger_name : str
            Logger name.
        msg : str
            Log message.
        *args : Any
            Positional arguments forwarded to the logger.
        **kwargs : Any
            Keyword arguments forwarded to the logger.

        """
        self._get_active_logger(logger_name).debug(msg, *args, **kwargs)

    def info(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Similar to `debug()`, but for the INFO level."""
        self._get_active_logger(logger_name).info(msg, *args, **kwargs)

    def warning(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Similar to `debug()`, but for the WARNING level."""
        self._get_active_logger(logger_name).warning(msg, *args, **kwargs)

    def error(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Similar to `debug()`, but for the ERROR level."""
        self._get_active_logger(logger_name).error(msg, *args, **kwargs)

    def critical(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Similar to `debug()`, but for the CRITICAL level."""
        self._get_active_logger(logger_name).critical(msg, *args, **kwargs)

    def is_enabled_for(self, logger_name: str, level: int) -> bool:
        """Return whether the specified logger would emit records at ``level``."""
        return self._get_active_logger(logger_name).isEnabledFor(level)


class Colors(Enum):
    """An enumeration of ANSI escape codes for common text colors and styles.

    Access the color string using the .value attribute (e.g., Colors.RED.value).
    """

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    WHITE = "\x1b[37m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"


class CustomFormatter(logging.Formatter):
    """Custom Formatter to display different style messages depending on their level."""

    def format(self, record: logging.LogRecord) -> str:
        """Format dynamically log records based on their severity level.

        For INFO and DEBUG level records, only the log message is displayed.
        For other levels (WARNING, ERROR, CRITICAL), the level name is
        included, formatted in red for easy visibility, followed by the message.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            The formatted message.

        """
        if record.levelno == logging.INFO or record.levelno == logging.DEBUG:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = (
                f"{Colors.RED.value}%(levelname)-1s{Colors.RESET.value} : %(message)s"
            )
        return super().format(record)


class ProgressHandler(logging.StreamHandler):
    """A custom logging handler for displaying single-line, dynamic updates in the console.

    It uses a carriage return to overwrite the previous line.

    Attributes
    ----------
    stream : sys.stdout, optional
        The output stream, defaults to sys.stdout.

    """

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        super().__init__(stream)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, overwriting the current console line.

        Parameters
        ----------
        record : logging.LogRecord
            The record to emit.

        """
        try:
            msg = self.format(record)
            self.stream.write("\r" + msg)
            self.stream.flush()
        except Exception:
            self.handleError(record)


ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class AnsiStrippingFormatter(CustomFormatter):
    """Custom Formatter to remove Ansi caracter.

    Used when logging a colored message in the stdout and a file.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log records to remove Ansi caracters.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be fromatted.

        Returns
        -------
        str
            The formatted message.

        """
        formatted_message = super().format(record)
        clean_message = ANSI_ESCAPE_PATTERN.sub("", formatted_message)
        clean_message = clean_message.replace("\r", "")
        return clean_message


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default_formatter": {
            "()": CustomFormatter,
        },
        "file_formatter": {
            "()": AnsiStrippingFormatter,
        },
    },
    "handlers": {
        "log_file": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.log",
            "mode": "a",
        },
        "console_output_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "general_output_file": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.out",
            "mode": "a",
        },
        "step_informations": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.info",
            "mode": "a",
        },
        "events_output": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.events",
            "mode": "a",
        },
        "reference_table_output": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.reference_table",
            "mode": "w",
        },
        "progress_bar_handler": {
            "class": "pykmc.log.ProgressHandler",
            "formatter": "default_formatter",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "log": {
            "handlers": ["log_file", "console_output_handler"],
            "propagate": False,
        },
        "output": {
            "handlers": ["general_output_file"],
        },
        "info": {"handlers": ["step_informations"]},
        "events": {"handlers": ["events_output"]},
        "reference_table": {"handlers": ["reference_table_output"]},
        "progress": {
            # "handlers": ["log_file", "progress_bar_handler"],
            "handlers": ["progress_bar_handler"],
        },
    },
}


class LogKMC(LogManager):
    """Manage logging for the KMC, offering dynamic verbosity control.

    Extends `LogManager` to adjust logging levels for specified loggers
    (e.g., 'log', 'output') based on a simple verbosity setting (0-2).
    Provides convenience methods for KMC-specific log messages.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary for loggers in the format expected by logging.config.dictConfig.
    verbosity : int, optional
        Defines the loggers level (0=WARNING, 1=INFO, 2=DEBUG). Defaults to 1.

    """

    OUTPUT_TABLE_COLUMNS: ClassVar[tuple[tuple[int, str, str], ...]] = (
        (10, "n", "Step"),
        (18, ".4f", "E(eV)"),
        (14, ".6f", "Ea(eV)"),
        (14, ".6e", "dT(s)"),
        (14, ".6e", "k_evt(ps-1)"),
        (14, ".6e", "T(s)"),
        (14, ".6e", "k_tot(ps-1)"),
        (14, "d", "Ref event"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "event_id"),
        (14, ".6e", "Cpu time(s)"),
        (14, ".6e", "Wall time(s)"),
    )

    EVENTS_TABLE_COLUMNS: ClassVar[tuple[tuple[int, str, str], ...]] = (
        (5, "d", "#"),
        (7, "s", "Types"),
        (13, "d", "CentralAtom"),
        (10, "d", "RefEvent"),
        (14, ".6f", "dE_forward"),
        (14, ".6f", "dE_backward"),
        (14, ".6f", "dE_asym"),
        (14, ".6e", "k"),
        (14, ".6f", "dra_i"),
        (14, ".6f", "dra_f"),
        (9, "s", "Refined"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "event_id"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_initial"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_saddle"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_final"),
    )

    REFERENCE_TABLE_COLUMNS: ClassVar[tuple[tuple[int, str, str], ...]] = (
        (10, "d", "idx_ref"),
        (14, ".6f", "dE_forward"),
        (14, ".6f", "dE_backward"),
        (14, ".6e", "k"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "event_id"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_initial"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_saddle"),
        (DISPLAYED_HASH_LENGTH + 4, "s", "id_final"),
        (14, "d", "move_atom_idx"),
        (14, "d", "idx_backward"),
        (14, ".6f", "dra"),
    )

    def __init__(self, config_dict: dict[str, Any], verbosity: int = 1) -> None:
        super().__init__(config_dict)
        self._verbosity = verbosity
        # apply verbosity option modifying logger and handlers level
        self._apply_verbosity_level()

    # TODO : set level should be more robust, especially for the progress bar
    def _apply_verbosity_level(self) -> None:
        """Modify loggers and their handlers levels.

        Raises
        ------
        ValueError
           if verbosity value is not 0, 1 or 2.

        """
        if self._verbosity == 0:
            level = logging.WARNING
        elif self._verbosity == 1:
            level = logging.INFO
        elif self._verbosity == 2:
            level = logging.DEBUG
        else:
            raise ValueError("verbosity should be 0, 1 or 2")

        for logger_name in self._logger:
            logger = self._get_active_logger(logger_name)
            logger.setLevel(level)
            if (
                logger_name == "progress" and self._verbosity >= 2
            ):  # To pass debug level for progress bar
                logger.setLevel(logging.DEBUG)
            for handler in logger.handlers:
                if (
                    logger_name == "progress"
                    and isinstance(handler, ProgressHandler)
                    and self._verbosity >= 2
                ):
                    handler.setLevel(
                        logging.DEBUG
                    )  # always display bar in stdout bug only debug level for log_file
                else:
                    handler.setLevel(level)

    def title(self, logger_name: str) -> None:
        """Display pyKMC title to the logger.

        Parameters
        ----------
        logger_name : str
            the logger name.

        """
        self.info(logger_name, "          |              ")
        self.info(logger_name, ",---.,   .|__/ ,-.-.,---.")
        self.info(logger_name, "|   ||   ||  \\ | | ||    ")
        self.info(logger_name, "|---'`---|`   `` ' '`---'")
        self.info(logger_name, "|    `---              ")
        self.info(logger_name, "\n")

    def write_parameters(
        self, logger_name: str, params: Parameters, width: int = 60, indent: int = 4
    ) -> None:
        """Write simulation parameter to the logger.

        Parameters
        ----------
        logger_name : str
            The logger name.
        params : Parameters
            The configuration Object
        width : int, optional
            Width of the lines, by default 60.
        indent : int, optional
            How many space define the indentation, by default 4.

        """
        centered_title = "SIMULATION PARAMETERS".center(width)
        separator = "=" * width
        self.info(
            logger_name, "\n{}\n{}\n{}".format(separator, centered_title, separator)
        )

        max_key_len = 0
        for section, model in params:
            if model is not None:
                for key, value in model:
                    max_key_len = max(max_key_len, len(str(key)))

        for section, model in params:
            self.info(logger_name, section)
            if model is not None:
                for key, value in model:
                    self.info(
                        logger_name,
                        "{}{:<{}} : {}".format(" " * indent, key, max_key_len, value),
                    )
        self.new_line(logger_name)

    def output_file_header(self, logger_name: str) -> None:
        """Write the header of the output file.

        Parameters
        ----------
        logger_name: str
            The logger name.

        """
        # Information header :
        self.info(logger_name, "# Simulation Progress Tracking File")
        self.new_line(logger_name)
        self.info(logger_name, "# Column Details:")
        self.info(logger_name, "\t# Step          : Simulation step number.")
        self.info(logger_name, "\t# E(eV)         : Energy of the system.")
        self.info(logger_name, "\t# Ea(eV)        : Event activation energy barrier.")
        self.info(
            logger_name, "\t# dT(s)         : Time elapsed for this specific step."
        )
        self.info(
            logger_name, "\t# k_evt(ps-1)   : Rate constant of the selected event."
        )
        self.info(logger_name, "\t# T(s)          : Total time since simulation start.")
        self.info(
            logger_name,
            "\t# k_tot(ps-1)   : Total rate constant of all possible events at this step.",
        )
        self.info(
            logger_name,
            "\t# Ref event     : Index in the reference table of the selected event.",
        )
        self.info(
            logger_name,
            f"\t# event_id      : First {DISPLAYED_HASH_LENGTH} characters of the selected event's combined topology ID.",
        )
        self.info(logger_name, "\t# Cpu time(s)   : Cpu time in seconds.")
        self.info(logger_name, "\t# Wall time(s)  : Wall time in seconds.")
        self.new_line(logger_name)
        # First line of the table
        self.info(logger_name, self._format_output_table_header())
        self.info(logger_name, "-" * len(self._format_output_table_header()))

    def table_line_info_kmc(self, logger_name: str, *args: int | float) -> None:
        """Write a formatted line of simulation output values into the output table.

        Parameters
        ----------
        logger_name : str
            The logger name.
        *args : int | float
            Values representing the columns of the simulation output progress table.
            These should correspond to:
                - Step Number
                - Total energy (E in eV)
                - Event energy barrier (Ea in eV)
                - Time elapsed for this step (dT in s)
                - Rate constant of current event (k_evt in ps-1)
                - Total cumulative time (T in s)
                - Total rate constant of all events (k_tot in ps-1)
                - Index in the reference table of the selected event
                - Truncated display form of the event's combined topology ID (event_id)
                - Cpu time (s)
                - Wall time (s)

        """
        self.info(logger_name, self._format_output_table_row(*args))

    @classmethod
    def _format_output_table_header(cls) -> str:
        return " ".join(
            f"{name:<{width}s}" for width, _, name in cls.OUTPUT_TABLE_COLUMNS
        )

    @classmethod
    def _format_output_table_row(cls, *values: int | float) -> str:
        cells: list[str] = []
        for idx, (width, value_fmt, _) in enumerate(cls.OUTPUT_TABLE_COLUMNS):
            value = values[idx] if idx < len(values) else None
            if value is None:
                cells.append(" " * width)
                continue
            cells.append(f"{value:<{width}{value_fmt}}")
        return " ".join(cells)

    @classmethod
    def _format_events_table_header(cls) -> str:
        return " ".join(
            f"{name:<{width}s}" for width, _, name in cls.EVENTS_TABLE_COLUMNS
        )

    @classmethod
    def _format_events_table_row(cls, *values) -> str:
        cells: list[str] = []
        for idx, (width, value_fmt, _) in enumerate(cls.EVENTS_TABLE_COLUMNS):
            value = values[idx] if idx < len(values) else None
            if value is None:
                cells.append(" " * width)
                continue
            cells.append(f"{value:<{width}{value_fmt}}")
        return " ".join(cells)

    @classmethod
    def _format_reference_table_header(cls) -> str:
        return " ".join(
            f"{name:<{width}s}" for width, _, name in cls.REFERENCE_TABLE_COLUMNS
        )

    @classmethod
    def _format_reference_table_row(cls, *values) -> str:
        cells: list[str] = []
        for idx, (width, value_fmt, _) in enumerate(cls.REFERENCE_TABLE_COLUMNS):
            value = values[idx] if idx < len(values) else None
            if value is None:
                cells.append(" " * width)
                continue
            cells.append(f"{value:<{width}{value_fmt}}")
        return " ".join(cells)

    def events_file_header(self, logger_name: str) -> None:
        """Write header of the events file

        Parameters
        ----------
        logger_name: str
            The logger name.
        """
        self.info(logger_name, "#Actif Events Informations File")
        self.info(logger_name, "\t #Type: The central atom's type.")
        self.info(
            logger_name, "\t #Central Atom: Index of the central atom of the event."
        )
        self.info(
            logger_name,
            "\t #Ref Event: Index of reference event in the reference table.",
        )
        self.info(
            logger_name, "\t #dE forward: Energy barrier of the forward reaction (eV)."
        )
        self.info(
            logger_name,
            "\t #dE backward: Energy barrier of the backward reaction (eV).",
        )
        self.info(logger_name, "\t #dE asym: |dE forward - dE backward| (eV).")
        self.info(logger_name, "\t #k: rate of the forward reaction (ps-1)")
        self.info(
            logger_name,
            "\t #dra_i: displacement between the initial positions and the saddle positions.",
        )
        self.info(
            logger_name,
            "\t #dra_f: displacement between the final positions and the saddle positions.",
        )
        self.info(logger_name, "\t #Refined: - T : The event has been refined.")
        self.info(logger_name, "\t #         - F : The event has not been refined.")
        self.new_line(logger_name)
        self.info(logger_name, self._format_events_table_header())
        self.info(logger_name, "-" * len(self._format_events_table_header()))

    def reference_table_file_header(self, logger_name: str) -> None:
        """Write the header of the reference table file."""
        self.info(logger_name, "#Reference Event Table")
        self.info(logger_name, "\t #idx_ref      : Index of the reference event.")
        self.info(logger_name, "\t #dE_forward   : Forward energy barrier (eV).")
        self.info(logger_name, "\t #dE_backward  : Backward energy barrier (eV).")
        self.info(
            logger_name,
            "\t #k            : Rate constant of the forward reaction (ps-1).",
        )
        self.info(
            logger_name,
            f"\t #event_id     : First {DISPLAYED_HASH_LENGTH} characters of the combined topology ID (ini+sad+fin).",
        )
        self.info(
            logger_name,
            f"\t #id_initial   : First {DISPLAYED_HASH_LENGTH} characters of the initial topology ID.",
        )
        self.info(
            logger_name,
            f"\t #id_saddle    : First {DISPLAYED_HASH_LENGTH} characters of the saddle topology ID.",
        )
        self.info(
            logger_name,
            f"\t #id_final     : First {DISPLAYED_HASH_LENGTH} characters of the final topology ID.",
        )
        self.info(
            logger_name,
            "\t #move_atom_idx: Index of the moving atom in the environment.",
        )
        self.info(
            logger_name, "\t #idx_backward : Index of the corresponding backward event."
        )
        self.info(
            logger_name,
            "\t #dra          : Displacement between initial and saddle positions.",
        )
        self.new_line(logger_name)
        self.info(logger_name, self._format_reference_table_header())
        self.info(logger_name, "-" * len(self._format_reference_table_header()))

    def reference_table_write(self, logger_name: str, reference_table) -> None:
        """Write a snapshot of the reference event table, overwriting on each call."""
        logger = self._get_active_logger(logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.stream:
                handler.stream.seek(0)
                handler.stream.truncate()
                break
        tbl = reference_table.table
        self.reference_table_file_header(logger_name)
        self.info(
            logger_name, "========== Reference Events ({}) ==========".format(len(tbl))
        )
        for i in range(len(tbl)):
            row = tbl.iloc[i]
            self.info(
                logger_name,
                self._format_reference_table_row(
                    int(row["idx_ref"]),
                    float(row["dE_forward"]),
                    float(row["dE_backward"]),
                    float(row["k"]),
                    fmt_hash(row["event_id"]),
                    fmt_hash(row["id_initial"]),
                    fmt_hash(row["id_saddle"]),
                    fmt_hash(row["id_final"]),
                    int(row["move_atom_idx"]),
                    int(row["idx_backward"]),
                    float(row["dra"]),
                ),
            )

    def events_write(self, logger_name: str, events_info) -> None:
        """Write formatted event rows to the events file."""
        for i in range(len(events_info.types)):
            self.info(
                logger_name,
                self._format_events_table_row(
                    i,
                    str(events_info.types[i]),
                    int(events_info.central_atom[i]),
                    int(events_info.reference_events[i]),
                    float(events_info.dE_forward[i]),
                    float(events_info.dE_backward[i]),
                    float(events_info.dE_asym[i]),
                    float(events_info.k[i]),
                    float(events_info.dra_i[i]),
                    float(events_info.dra_f[i]),
                    str(events_info.refined[i]),
                    fmt_hash(events_info.event_id[i]),
                    fmt_hash(events_info.id_initial[i]),
                    fmt_hash(events_info.id_saddle[i]),
                    fmt_hash(events_info.id_final[i]),
                ),
            )

    def events_file_step_first_line(self, logger_name: str, step: int) -> None:
        """Write the first line with step informations

        Parameters
        ----------
        logger_name: str
            The logger name.
        """
        self.info(logger_name, "#Step: {}".format(step))

    def events_applicable_info_line(
        self, logger_name: str, selected_event: int
    ) -> None:
        self.info(
            logger_name,
            "========== Applicable Events (Selected={}) ==========".format(
                selected_event
            ),
        )

    def events_basin_info_line(self, logger_name: str, selected_event: int) -> None:

        self.info(
            logger_name,
            "========== Basin Exit Events (Selected={}) ==========".format(
                selected_event
            ),
        )

    def new_line(self, logger_name: str) -> None:
        """Write a new line in the logger.

        Parameters
        ----------
        logger_name : str
           The logger name.

        """
        self.info(logger_name, "")

    def progress_bar(
        self,
        logger_name: str,
        current_step: int,
        total_steps: int,
        bar_length: int = 40,
    ) -> None:
        """Display a progression bar.

        Parameters
        ----------
        logger_name : str
            The logger name.
        current_step : int
            Current step of the process.
        total_steps : int
            Total steps of the process.
        bar_length : int, optional
            Lenght of the progress bar, by default 40.

        """
        # Compute percentage
        percent = (current_step / total_steps) * 100

        # Dynamical bar colors
        bar_fill_color = Colors.WHITE.value

        if percent < 30:
            bar_fill_color = Colors.RED.value
        elif percent < 70:
            bar_fill_color = Colors.YELLOW.value
        else:
            bar_fill_color = Colors.GREEN.value

        # bar fill lenght
        filled_length = int(bar_length * current_step / total_steps)
        # all bar
        bar_segment = "#" * filled_length + "-" * (bar_length - filled_length)
        progress_message = (
            f"\r\t Progression: "
            f"{bar_fill_color}[{bar_segment}]{Colors.RESET.value}"
            f" {percent:.1f}% "
        )

        # Envoi du message via le logger
        self.info(logger_name, progress_message)
        if bar_length == filled_length:
            self.info(logger_name, "\n")
