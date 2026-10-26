import operator

import numpy as np
import pandas as pd
import fractions as fr
from pandas._typing import type_t
from pandas.core import roperator
from pandas.core.arraylike import OpsMixin


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


# checked int64 arithmetic
# ------------------------
# numpy integer arithmetic wraps around silently, so every operation that can
# overflow is checked and raises an OverflowError instead.

_INT_MIN = np.iinfo(np.int64).min


def _overflow_if(bad):
    if np.any(bad):
        raise OverflowError("fraction arithmetic overflowed int64")


def _checked_mul(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(y, np.int64))
    with np.errstate(over="ignore", divide="ignore"):
        r = x * y
        # a wrapped product never divides back to the original factor
        bad = (x != 0) & (r // np.where(x == 0, 1, x) != y)
    _overflow_if(bad | ((x == -1) & (y == _INT_MIN)) | ((y == -1) & (x == _INT_MIN)))
    return r


def _checked_add(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(y, np.int64))
    with np.errstate(over="ignore"):
        r = x + y
    # overflow iff both summands have the same sign and the result does not
    _overflow_if(((x >= 0) == (y >= 0)) & ((r >= 0) != (x >= 0)))
    return r


def _checked_neg(x):
    _overflow_if(x == _INT_MIN)
    return -x


# operations on (numerator, denominator) parts
# --------------------------------------------
# Inputs must be normalized and contain no NA (denominator 0).
# Results are normalized again; the gcd reductions follow CPython's Fraction
# and keep intermediate values small.


def _add(a, b, c, d):
    g = np.gcd(b, d)
    s = b // g
    t = _checked_add(_checked_mul(a, d // g), _checked_mul(c, s))
    g2 = np.gcd(t, g)
    return t // g2, _checked_mul(s, d // g2)


def _sub(a, b, c, d):
    return _add(a, b, _checked_neg(c), d)


def _mul(a, b, c, d):
    g1 = np.gcd(a, d)
    g2 = np.gcd(c, b)
    return _checked_mul(a // g1, c // g2), _checked_mul(b // g2, d // g1)


def _truediv(a, b, c, d):
    if np.any(c == 0):
        raise ZeroDivisionError("division by zero")
    sign = np.where(c < 0, -1, 1)
    return _mul(a, b, _checked_mul(sign, d), _checked_mul(sign, c))


def _floordiv(a, b, c, d):
    n, m = _truediv(a, b, c, d)
    return n // m, np.ones_like(n)


def _mod(a, b, c, d):
    # x mod y = x - floor(x / y) * y
    q, one = _floordiv(a, b, c, d)
    return _sub(a, b, *_mul(q, one, c, d))


def _compare(op, a, b, c, d):
    """Compare a/b with c/d elementwise. Requires positive denominators."""
    # a/b < c/d iff a*d < c*b
    try:
        return op(_checked_mul(a, d), _checked_mul(c, b))
    except OverflowError:
        # compare exactly with Python ints
        obj = lambda x: np.asarray(x).astype(object)
        return np.asarray(op(obj(a) * obj(d), obj(c) * obj(b)), dtype=bool)


def _min(a, b, c, d):
    left = _compare(operator.le, a, b, c, d)
    return np.where(left, a, c), np.where(left, b, d)


def _max(a, b, c, d):
    left = _compare(operator.ge, a, b, c, d)
    return np.where(left, a, c), np.where(left, b, d)


def _tree_reduce(f, num, den, neutral):
    """
    Reduce non-empty arrays of fractions with a binary operation on parts.

    Combines the two halves of the arrays elementwise until one element is left,
    so that each step is vectorized.
    An odd element is padded with `neutral`, or with itself if `neutral` is None.
    """
    while len(num) > 1:
        if len(num) % 2 == 1:
            pad_num, pad_den = (num[-1], den[-1]) if neutral is None else neutral
            num, den = np.append(num, pad_num), np.append(den, pad_den)
        h = len(num) // 2
        num, den = f(num[:h], den[:h], num[h:], den[h:])
    return num, den


# reduction: (binary operation, neutral element)
_TREE_REDUCTIONS = {
    "sum": (_add, (0, 1)),
    "prod": (_mul, (1, 1)),
    "min": (_min, None),
    "max": (_max, None),
}


_ARITH_OPS = {
    operator.add: _add,
    operator.sub: _sub,
    operator.mul: _mul,
    operator.truediv: _truediv,
    operator.floordiv: _floordiv,
    operator.mod: _mod,
}

# reversed operators and their forward counterpart
_REVERSED_OPS = {
    roperator.radd: operator.add,
    roperator.rsub: operator.sub,
    roperator.rmul: operator.mul,
    roperator.rtruediv: operator.truediv,
    roperator.rfloordiv: operator.floordiv,
    roperator.rmod: operator.mod,
    roperator.rdivmod: divmod,
}


class FractionArray(OpsMixin, pd.api.extensions.ExtensionArray):
    """
    An array of fractions, stored as two int64 arrays for numerators and denominators.

    Fractions are always normalized (reduced, positive denominator).
    Missing values are marked by a denominator of 0.

    Arithmetic with fractions and integers is exact and vectorized,
    and raises an OverflowError if an intermediate value exceeds int64.
    Arithmetic with floats returns a float ndarray, like Fraction does with float.
    Powers are computed elementwise with Python's Fraction.
    """

    def __init__(self, numerators: np.ndarray, denominators: np.ndarray):
        self._num = np.asarray(numerators, dtype=np.int64)
        self._den = np.asarray(denominators, dtype=np.int64)
        if self._num.shape != self._den.shape or self._num.ndim != 1:
            raise ValueError(
                "numerators and denominators must be 1D and of equal length"
            )

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

    # arithmetic and comparison
    # -------------------------

    def _coerce_other(self, other):
        """
        Convert the other operand of a binary operation.

        Returns a (numerators, denominators) pair of arrays of the same length as self,
        a float scalar or array for float operands, or NotImplemented.
        """
        n = len(self)
        if isinstance(other, FractionArray):
            if len(other) != n:
                raise ValueError("Lengths must match")
            return other._num, other._den
        if pd.api.types.is_scalar(other):
            if pd.isna(other):
                return np.zeros(n, np.int64), np.zeros(n, np.int64)
            if isinstance(other, (int, np.integer, fr.Fraction)):
                num, den = _to_parts(other)
                return np.full(n, num, np.int64), np.full(n, den, np.int64)
            if isinstance(other, (float, np.floating)):
                return float(other)
            return NotImplemented
        if isinstance(other, list):
            other = np.asarray(other)
        if not isinstance(other, (np.ndarray, pd.api.extensions.ExtensionArray)):
            return NotImplemented
        if len(other) != n:
            raise ValueError("Lengths must match")
        if pd.api.types.is_float_dtype(other.dtype):
            return np.asarray(other, dtype=float, copy=None)
        if isinstance(other, np.ndarray) and other.dtype.kind in "iub":
            return other.astype(np.int64), np.ones(n, np.int64)
        try:
            other = type(self)._from_sequence(other)
        except TypeError, ValueError:
            return NotImplemented
        return other._num, other._den

    def _to_float(self) -> np.ndarray:
        out = np.full(len(self), np.nan)
        valid = self._den != 0
        out[valid] = self._num[valid] / self._den[valid]
        return out

    def _arith_method(self, other, op):
        if op in (operator.pow, roperator.rpow):
            return self._pointwise(other, op)
        coerced = self._coerce_other(other)
        if coerced is NotImplemented:
            return NotImplemented
        if not isinstance(coerced, tuple):
            return op(self._to_float(), coerced)

        na = (self._den == 0) | (coerced[1] == 0)
        # replace NA by 1 so that it doesn't trigger division by zero
        x = (np.where(na, 1, self._num), np.where(na, 1, self._den))
        y = (np.where(na, 1, coerced[0]), np.where(na, 1, coerced[1]))
        if op in _REVERSED_OPS:
            x, y = y, x
            op = _REVERSED_OPS[op]

        if op is divmod:
            return (
                self._with_na(_floordiv(*x, *y), na),
                self._with_na(_mod(*x, *y), na),
            )
        return self._with_na(_ARITH_OPS[op](*x, *y), na)

    def _with_na(self, parts, na):
        num, den = parts
        return type(self)(np.where(na, 0, num), np.where(na, 0, den))

    def _pointwise(self, other, op):
        """Apply op elementwise using Python's Fraction."""
        if pd.api.types.is_list_like(other):
            if len(other) != len(self):
                raise ValueError("Lengths must match")
        else:
            other = [other] * len(self)
        results = [
            None if pd.isna(x) or pd.isna(y) else op(x, y) for x, y in zip(self, other)
        ]
        if any(isinstance(r, float) for r in results):
            return np.array([np.nan if r is None else r for r in results], dtype=float)
        return type(self)._from_sequence(results)

    def _cmp_method(self, other, op):
        coerced = self._coerce_other(other)
        if coerced is NotImplemented:
            # incomparable values are never equal
            if op is operator.eq:
                return np.zeros(len(self), dtype=bool)
            if op is operator.ne:
                return np.ones(len(self), dtype=bool)
            raise TypeError(f"Cannot compare {self.dtype} with {type(other).__name__}")
        if not isinstance(coerced, tuple):
            return op(self._to_float(), coerced)

        a, b = self._num, self._den
        c, d = coerced
        result = _compare(op, a, b, c, d)
        # NA behaves like NaN: only != is true
        return np.where((b == 0) | (d == 0), op is operator.ne, result)

    def __neg__(self):
        return type(self)(_checked_neg(self._num), self._den.copy())

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        _overflow_if(self._num == _INT_MIN)
        return type(self)(np.abs(self._num), self._den.copy())

    # reductions
    # ----------

    def _reduce(self, name: str, *, skipna=True, keepdims=False, **kwargs):
        result = self._reduce_scalar(name, skipna=skipna, **kwargs)
        if not keepdims:
            return result
        if isinstance(result, (bool, np.bool_, float)):
            return np.array([result])
        return type(self)._from_sequence([result])

    def _reduce_scalar(self, name, *, skipna, min_count=0, ddof=1, **kwargs):
        na = self.isna()
        if name in ("any", "all"):
            # like floats, NA is truthy if it is not skipped
            truthy = (self._num != 0) | na
            values = truthy[~na] if skipna else truthy
            return bool(values.any() if name == "any" else values.all())
        if name not in (
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "median",
            "var",
            "std",
            "sem",
        ):
            raise TypeError(
                f"'{type(self).__name__}' does not support reduction '{name}'"
            )

        if not skipna and na.any():
            return np.nan if name in ("std", "sem") else self.dtype.na_value
        num, den = self._num[~na], self._den[~na]
        n = len(num)

        if name in _TREE_REDUCTIONS:
            f, neutral = _TREE_REDUCTIONS[name]
            if n == 0 or n < min_count:
                if neutral is None or n < min_count:
                    return self.dtype.na_value
                return fr.Fraction(*neutral)
            return _part_to_scalar(*_tree_reduce(f, num, den, neutral))
        if n == 0:
            return np.nan if name in ("std", "sem") else self.dtype.na_value
        if name == "mean":
            return _part_to_scalar(*self._mean_parts(num, den))
        if name == "median":
            # sorting is done with Python's Fraction, which is exact
            values = sorted(fr.Fraction(int(a), int(b)) for a, b in zip(num, den))
            mid = n // 2
            return values[mid] if n % 2 == 1 else (values[mid - 1] + values[mid]) / 2

        # var, std, sem
        if n - ddof <= 0:
            return np.nan if name in ("std", "sem") else self.dtype.na_value
        mean_num, mean_den = self._mean_parts(num, den)
        dev = _sub(num, den, np.full(n, mean_num[0]), np.full(n, mean_den[0]))
        sq_num, sq_den = _tree_reduce(_add, *_mul(*dev, *dev), (0, 1))
        var = _part_to_scalar(*_truediv(sq_num, sq_den, np.array([n - ddof]), 1))
        if name == "var":
            return var
        std = float(np.sqrt(float(var)))
        return std if name == "std" else std / np.sqrt(n)

    @staticmethod
    def _mean_parts(num, den):
        sum_num, sum_den = _tree_reduce(_add, num, den, (0, 1))
        return _truediv(sum_num, sum_den, np.array([len(num)]), 1)


def _part_to_scalar(num, den):
    """Convert a length-1 (numerators, denominators) pair to a Fraction."""
    return fr.Fraction(int(num[0]), int(den[0]))
