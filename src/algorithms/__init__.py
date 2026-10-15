"""Algorithm registry. A new algorithm registers itself with @register("name");
main.py never needs to change."""
from typing import Callable, Dict, Type

from ..core.search import LocalSearch

ALGORITHMS: Dict[str, Type[LocalSearch]] = {}


def register(name: str) -> Callable[[Type[LocalSearch]], Type[LocalSearch]]:
    def decorator(cls: Type[LocalSearch]) -> Type[LocalSearch]:
        cls.name = name
        ALGORITHMS[name] = cls
        return cls
    return decorator


# Import modules for their @register side effect.
from . import baseline  # noqa: E402,F401
from . import hill_climbing  # noqa: E402,F401
from . import simulated_annealing  # noqa: E402,F401
