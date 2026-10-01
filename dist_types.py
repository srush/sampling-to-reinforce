"""Array containers for the introductory exercises; no arithmetic or lineage."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _values(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("Values must be a nonempty finite 1D array")
    result.setflags(write=False)
    return result


def _probabilities(prob: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    result = np.array(prob, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Probabilities must be finite, nonnegative, and match the values")
    if not np.isclose(result.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError("Probabilities must sum to 1")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False, slots=True, init=False)
class Var:
    values: NDArray[np.float64]
    probs: NDArray[np.float64]

    def __init__(self, values: ArrayLike, probs: ArrayLike):
        values = _values(values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probs", _probabilities(probs, values.shape))

    def prob(self, value: float) -> float:
        """Probability of a value, including any repeated entries in the support."""
        return float(self.probs[self.values == value].sum())

    def log_prob(self, value: float) -> float:
        return float(np.log(self.prob(value)))


@dataclass(frozen=True, eq=False, slots=True, init=False)
class Joint:
    _x: NDArray[np.float64]
    _y: NDArray[np.float64]
    probs: NDArray[np.float64]

    def __init__(self, x: ArrayLike, y: ArrayLike, probs: ArrayLike):
        x, y = _values(x), _values(y)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_y", y)
        object.__setattr__(self, "probs", _probabilities(probs, (len(x), len(y))))


def _joint(x: ArrayLike, y: ArrayLike, prob: ArrayLike) -> Joint:
    return Joint(x, y, prob)
