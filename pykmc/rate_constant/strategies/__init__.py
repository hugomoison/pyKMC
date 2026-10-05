from pykmc._core import autodiscover

from .base import RateConstantStrategy

RateConstantStrategy._import_errors = autodiscover(__name__, __path__)

__all__ = ["RateConstantStrategy"]
