from pykmc._core import autodiscover

from .base import PSRStrategy

PSRStrategy._import_errors = autodiscover(__name__, __path__)

__all__ = ["PSRStrategy"]
