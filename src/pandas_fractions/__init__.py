import numpy as np
import pandas as pd
import fractions as fr
from pandas._typing import type_t


@pd.api.extensions.register_extension_dtype
class FractionDType(pd.api.extensions.ExtensionDtype):
    """A pandas dtype for fractions."""

    # must be class attributes so that pandas can look up the dtype by name
    name = "frac"
    type = fr.Fraction

    @property
    def _is_numeric(self):
        return True

    def construct_array_type(self) -> type_t["FractionArray"]:
        return FractionArray


def _to_parts(value) -> tuple[int, int]:
    """Convert a scalar to a (numerator, denominator) pair. NA becomes (0, 0)."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return 0, 0
    f = fr.Fraction(value)
    return f.numerator, f.denominator


class FractionArray(pd.api.extensions.ExtensionArray):
    """
    An array of fractions, stored as two int64 arrays for numerators and denominators.

    Fractions are always normalized (reduced, positive denominator).
    Missing values are marked by a denominator of 0.
    """

    def __init__(self, numerators: np.ndarray, denominators: np.ndarray):
        self._num = np.asarray(numerators, dtype=np.int64)
        self._den = np.asarray(denominators, dtype=np.int64)
        if self._num.shape != self._den.shape or self._num.ndim != 1:
            raise ValueError("numerators and denominators must be 1D and of equal length")

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(scalars, cls):
            return scalars.copy() if copy else scalars
        parts = [_to_parts(x) for x in scalars]
        num = np.array([n for n, _ in parts], dtype=np.int64)
        den = np.array([d for _, d in parts], dtype=np.int64)
        return cls(num, den)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values)

    def __getitem__(self, item):
        if pd.api.types.is_integer(item):
            den = self._den[item]
            if den == 0:
                return self.dtype.na_value
            return fr.Fraction(int(self._num[item]), int(den))
        item = pd.api.indexers.check_array_indexer(self, item)
        return type(self)(self._num[item], self._den[item])

    def __len__(self) -> int:
        return len(self._num)

    def __eq__(self, other):
        if isinstance(other, (pd.Series, pd.Index, pd.DataFrame)):
            return NotImplemented
        if isinstance(other, FractionArray):
            if len(other) != len(self):
                raise ValueError("Lengths must match to compare")
            onum, oden = other._num, other._den
        elif pd.api.types.is_scalar(other):
            if pd.isna(other):
                return np.zeros(len(self), dtype=bool)
            onum, oden = _to_parts(other)
        else:
            return self == type(self)._from_sequence(other)
        # NA (den == 0) never compares equal
        return (self._num == onum) & (self._den == oden) & (self._den != 0)

    @property
    def dtype(self) -> FractionDType:
        return FractionDType()

    @property
    def nbytes(self) -> int:
        return self._num.nbytes + self._den.nbytes

    def isna(self) -> np.ndarray:
        return self._den == 0

    def take(self, indices, *, allow_fill=False, fill_value=None):
        if allow_fill:
            fill_num, fill_den = _to_parts(fill_value)
        else:
            fill_num, fill_den = None, None
        take = pd.api.extensions.take
        num = take(self._num, indices, allow_fill=allow_fill, fill_value=fill_num)
        den = take(self._den, indices, allow_fill=allow_fill, fill_value=fill_den)
        return type(self)(num, den)

    def copy(self):
        return type(self)(self._num.copy(), self._den.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(
            np.concatenate([a._num for a in to_concat]),
            np.concatenate([a._den for a in to_concat]),
        )
