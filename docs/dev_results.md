# Result type

> **Refactoring in progress.** `ErrorType` is the legacy shared enumeration. It is emptied module by module and will be removed once every module declares its own `ErrorCode` subclass. See Migration status.

## Why

During a KMC simulation, many operations can fail without the failure being fatal: an event search finds nothing, a refinement produces an invalid barrier, a point set registration finds no match. These are somewhat expected outcomes, not bugs, the simulation carries on, counts them, and reports them.

Raising an exception for each forces a `try`/`except` around every call, and it conflates "this attempt did not succeed" with "the code is broken". `pykmc._core.result` therefore exposes a `Result` type, so that an operation can report failure as a value that the caller must look at.

The point is not only to keep the simulation running, but to keep enough structured information to tell the user afterwards what went wrong and how often for analysis.

## Structure

An operation that can fail returns a `Result`, which is either:
- `Ok` : the operation's output, usually a dataclass 
- `Err` : an `ErrorInfo` describing the failure

```
Result
├── Ok(value)          value: the operation's output dataclass
└── Err(error)         error: ErrorInfo
                                ├── type      : an ErrorCode member  (what failed)
                                ├── message   : str                  (human readable)
                                ├── details   : str | None           (optional context)
                                └── variables : dict | None          (diagnostic values)
```

`Ok` and `Err` know nothing about each other, and `ErrorInfo` knows nothing about the error codes a given module declares. 
## Producing a result

Every exit point states explicitly whether it succeeded:

```python
from pykmc._core.result import Err, ErrorInfo, Ok, Result
from pykmc.my_module.results import MyModuleError, MyModuleOutput


def operation(...) -> Result[MyModuleOutput, ErrorInfo]:
    """Run the operation this module is responsible for."""
    if <input is unusable>:
        return Err(
            ErrorInfo(
                type=MyModuleError.INVALID_INPUT,
                message="<what went wrong, in plain words>",
                variables={"<name>": <value>},
            )
        )

    result = <do the work>

    if <result is not acceptable>:
        return Err(
            ErrorInfo(
                type=MyModuleError.ABOVE_THRESHOLD,
                message="<what went wrong, in plain words>",
                variables={"value": result, "threshold": threshold},
            )
        )

    return Ok(MyModuleOutput(...))
```

Fill in `variables` generously for diagnosing.
## Consuming a result

Test the class with `isinstance`, then unwrap:

```python
res = operation(...)

if isinstance(res, Ok):
    output = res.ok_value()
    ...
else:
    err_info = res.err_value()
    ...
```

To unwrap a result, use `isinstance`. It narrows the type in both branches, so a type checker flags `ok_value()` called on what may be an `Err`.

To merely test a result without unwrapping it, `is_ok()` is fine and reads better, e.g. :

```python
if res.is_ok():
    logger.debug("operation succeeded")
```

When an intermediate function cannot handle the failure itself, it forwards the `Err` untouched rather than unwrapping it:

```python
res = operation(...)
if not isinstance(res, Ok):
    return res
output = res.ok_value()
```

## Per-module `results.py`

A module that corresponds to a single operation declares its own outputs and error codes in a `results.py`, next to the code that produces them, e.g. :

```python
"""Outputs and error codes for <this module>."""

from dataclasses import dataclass
from enum import auto

from pykmc._core.result import ErrorCode


class MyModuleError(ErrorCode):
    """Failures this module can report."""

    INVALID_INPUT = auto()
    ABOVE_THRESHOLD = auto()


@dataclass
class MyModuleOutput:
    """Store the result of a successful operation."""

    ...
```


## Migration status

`ErrorType`, the single project-wide enumeration, is being replaced by per-module `ErrorCode` subclasses. While both coexist:

- `ErrorInfo.type` is annotated `ErrorType | ErrorCode`.
- Modules not yet refactored keep raising `ErrorType` members and keying their counters with `.name`, migrated ones use their own enumeration and `.key`.
- Counters and YAML reports may therefore mix both key formats (`EVENT_NOT_FOUND` and `MyModuleError.INVALID_INPUT`). Analysis scripts should tolerate both until the migration completes.

When a module is refactored, its members are removed from `ErrorType` and redeclared in its own subclass. Once `ErrorType` is empty it is deleted, and `ErrorInfo.type` narrows to `ErrorCode`.

The same applies to the output dataclasses, which currently still live in `pykmc/result.py`. That module is a temporary bridge, it re-exports the `Result` infrastructure from `_core` and holds the dataclasses that have not moved yet.


