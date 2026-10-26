import operator

import numpy as np
import pandas as pd
import fractions as fr
from pandas._typing import type_t
from pandas.api.extensions import no_default
from pandas.core import arraylike, roperator


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

    def _get_common_dtype(self, dtypes):
        # fractions represent all integers exactly, so they absorb integer dtypes
        # (e.g. when concatenating), but not floats or anything else
        if all(isinstance(t, FractionDType) or t.kind in "iu" for t in dtypes):
            return self
        return None


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


def _exact_order(num, den):
    """Return the indices that sort the fractions num/den exactly (stable)."""
    # Integers up to 2^53 are exact floats, and then num / den is correctly rounded,
    # so sorting by float gives the right order, except between distinct fractions
    # that round to the same float. These would be adjacent after sorting.
    limit = 2**53
    if np.all((num >= -limit) & (num <= limit) & (den <= limit)):
        floats = num / den
        order = np.argsort(floats, kind="stable")
        f, n, d = floats[order], num[order], den[order]
        if not np.any((f[1:] == f[:-1]) & ((n[1:] != n[:-1]) | (d[1:] != d[:-1]))):
            return order
    # otherwise sort exactly with Python's Fraction
    key = lambda i: fr.Fraction(int(num[i]), int(den[i]))
    return np.array(sorted(range(len(num)), key=key), dtype=np.intp)


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


def _scan(f, num, den):
    """
    Compute the prefix reductions (inclusive scan) of arrays of fractions.

    Uses the Hillis-Steele algorithm: in step k, every element is combined
    with the element 2^k positions before it, so each step is vectorized.
    """
    num, den = num.copy(), den.copy()
    k = 1
    while k < len(num):
        # the right-hand side is computed before assigning, so it uses the old values
        num[k:], den[k:] = f(num[:-k], den[:-k], num[k:], den[k:])
        k *= 2
    return num, den


# exact unary ufuncs on parts
# ---------------------------
# Like the binary operations, these expect normalized inputs without NA.


def _floor(num, den):
    return num // den, np.ones_like(num)


def _ceil(num, den):
    return _checked_neg(_checked_neg(num) // den), np.ones_like(num)


def _trunc(num, den):
    return np.where(num < 0, _ceil(num, den)[0], num // den), np.ones_like(num)


def _rint(num, den):
    # round half to even, like np.rint and Python's round()
    q, r = np.divmod(num, den)
    _overflow_if(r > np.iinfo(np.int64).max // 2)
    twice = 2 * r
    up = (twice > den) | ((twice == den) & (q % 2 == 1))
    return q + up, np.ones_like(num)


def _sign(num, den):
    return np.sign(num), np.ones_like(num)


def _square(num, den):
    return _mul(num, den, num, den)


def _reciprocal(num, den):
    return _truediv(np.ones_like(num), np.ones_like(num), num, den)


_UNARY_UFUNCS = {
    "floor": _floor,
    "ceil": _ceil,
    "trunc": _trunc,
    "rint": _rint,
    "sign": _sign,
    "square": _square,
    "reciprocal": _reciprocal,
}

# binary ufuncs: (operation, whether NA is skipped instead of propagated)
_BINARY_UFUNCS = {
    "minimum": (_min, False),
    "maximum": (_max, False),
    "fmin": (_min, True),
    "fmax": (_max, True),
}

# ufunc methods that map to reductions and accumulations
_UFUNC_REDUCTIONS = {
    "add": "sum",
    "multiply": "prod",
    "minimum": "min",
    "maximum": "max",
}
_UFUNC_ACCUMULATIONS = {
    "add": "cumsum",
    "multiply": "cumprod",
    "minimum": "cummin",
    "maximum": "cummax",
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


class FractionArray(arraylike.OpsMixin, pd.api.extensions.ExtensionArray):
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
    def _from_scalars(cls, scalars, *, dtype):
        # Strict version of _from_sequence, used by pandas to decide whether the results
        # of a pointwise operation (e.g. Series.combine) still have this dtype.
        # Only fractions, integers, and NA qualify, so that e.g. bools and floats don't.
        for x in scalars:
            exact = isinstance(x, (fr.Fraction, int, np.integer))
            if isinstance(x, (bool, np.bool_)) or not (exact or pd.isna(x)):
                raise TypeError(f"{type(x).__name__} is not a valid {dtype}")
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype, copy=False):
        # Fraction parses strings like "3", "-1/2", or "0.25"; empty strings are NA
        return cls._from_sequence(
            [None if isinstance(s, str) and not s.strip() else s for s in strings]
        )

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values)

    @property
    def _readonly(self) -> bool:
        # stored as the writeable flag of the numpy arrays,
        # so that views (e.g. slices) of read-only arrays are read-only as well
        return not self._num.flags.writeable

    @_readonly.setter
    def _readonly(self, value: bool):
        self._num.flags.writeable = not value
        self._den.flags.writeable = not value

    def __getitem__(self, item):
        if pd.api.types.is_integer(item):
            den = self._den[item]
            if den == 0:
                return self.dtype.na_value
            return fr.Fraction(int(self._num[item]), int(den))
        item = pd.api.indexers.check_array_indexer(self, item)
        return type(self)(self._num[item], self._den[item])

    def __setitem__(self, key, value):
        if self._readonly:
            raise ValueError("Cannot modify read-only array")
        key = pd.api.indexers.check_array_indexer(self, key)
        if pd.api.types.is_scalar(value):
            num, den = _to_parts(value)
        else:
            if not isinstance(value, FractionArray):
                value = type(self)._from_sequence(value)
            num, den = value._num, value._den
        # convert both parts before assigning,
        # so that an overflow can't leave the array half-modified
        num, den = np.asarray(num, np.int64), np.asarray(den, np.int64)
        self._num[key] = num
        self._den[key] = den

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

    # conversion
    # ----------

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        # the conversion always creates a new array, so `copy` doesn't matter
        dtype = np.dtype(object if dtype is None else dtype)
        na = self.isna()
        if dtype.kind == "f":
            result = self._to_float().astype(dtype, copy=False)
        else:
            result = np.empty(len(self), dtype=object)
            valid = ~na
            result[valid] = [
                fr.Fraction(int(a), int(b))
                for a, b in zip(self._num[valid], self._den[valid])
            ]
            result[na] = self.dtype.na_value
        if na_value is not no_default:
            result[na] = na_value
        return result.astype(dtype, copy=False)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "A FractionArray can't be converted to numpy without a copy"
            )
        return self.to_numpy(dtype=dtype)

    def astype(self, dtype, copy=True):
        dtype = pd.api.types.pandas_dtype(dtype)
        if isinstance(dtype, FractionDType):
            return self.copy() if copy else self
        if isinstance(dtype, np.dtype) and dtype.kind in "fO":
            return self.to_numpy(dtype=dtype)
        if isinstance(dtype, np.dtype) and dtype.kind in "iu":
            if self.isna().any():
                raise ValueError("Cannot convert NA to integer")
            # truncate towards zero, like int(Fraction)
            return _trunc(self._num, self._den)[0].astype(dtype)
        return super().astype(dtype, copy=copy)

    # sorting and factorizing
    # -----------------------

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return integer ranks that sort exactly like the fractions.

        Equal fractions get equal ranks, missing values the highest rank.
        """
        na = self.isna()
        num, den = self._num[~na], self._den[~na]
        order = _exact_order(num, den)
        # normalized fractions are equal iff their parts are equal
        n, d = num[order], den[order]
        new_value = np.ones(len(order), dtype=bool)
        new_value[1:] = (n[1:] != n[:-1]) | (d[1:] != d[:-1])
        valid_ranks = np.empty(len(order), dtype=np.int64)
        valid_ranks[order] = np.cumsum(new_value) - 1
        ranks = np.full(len(self), len(order), dtype=np.int64)
        ranks[~na] = valid_ranks
        return ranks

    def factorize(self, use_na_sentinel=True):
        # normalized fractions are equal iff their parts are equal,
        # so the unique values are the unique (numerator, denominator) rows
        pairs = np.stack([self._num, self._den], axis=1)
        if len(pairs) == 0:
            return np.array([], dtype=np.intp), self.copy()
        uniques, first, inverse = np.unique(
            pairs, axis=0, return_index=True, return_inverse=True
        )
        # like pandas, number the unique values in order of their first appearance
        order = np.argsort(first)
        codes = np.empty(len(order), dtype=np.intp)
        codes[order] = np.arange(len(order))
        codes = codes[inverse.reshape(-1)]
        uniques = uniques[order]

        na = self.isna()
        if use_na_sentinel and na.any():
            na_code = codes[np.argmax(na)]
            codes = np.where(na, -1, codes - (codes > na_code))
            uniques = np.delete(uniques, na_code, axis=0)
        return codes, type(self)(uniques[:, 0], uniques[:, 1])

    def unique(self):
        # in order of appearance, including NA, like pandas' unique
        return self.factorize(use_na_sentinel=False)[1]

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
        # e.g. roots are floats, and roots of negative numbers are complex
        for kind in (complex, float):
            if any(isinstance(r, kind) for r in results):
                return np.array(
                    [np.nan if r is None else r for r in results], dtype=kind
                )
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
        # decide by name, not by result, since a missing fraction (NaN) is a float
        if name in ("std", "sem", "any", "all"):
            # a numpy array, so that e.g. df.std() has dtype float64
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

    # accumulations
    # -------------

    def _accumulate(self, name: str, *, skipna=True, **kwargs):
        if name not in ("cumsum", "cumprod", "cummin", "cummax"):
            raise TypeError(
                f"'{type(self).__name__}' does not support accumulation '{name}'"
            )
        f, neutral = _TREE_REDUCTIONS[name.removeprefix("cum")]
        na = self.isna()
        if not na.any():
            return type(self)(*_scan(f, self._num, self._den))
        if not skipna:
            # everything from the first NA on is NA
            na = np.logical_or.accumulate(na)
        if na.all():
            zeros = np.zeros(len(self), np.int64)
            return type(self)(zeros, zeros.copy())
        # fill NA with a value that doesn't change the result:
        # the neutral element, or the first valid value for min and max
        if neutral is None:
            first = np.argmin(na)
            neutral = (self._num[first], self._den[first])
        num = np.where(na, neutral[0], self._num)
        den = np.where(na, neutral[1], self._den)
        return self._with_na(_scan(f, num, den), na)

    # numpy ufuncs
    # ------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # defer to pandas containers, which unwrap themselves and call this again
        out = kwargs.get("out", ())
        if any(
            isinstance(x, (pd.Series, pd.Index, pd.DataFrame)) for x in inputs + out
        ):
            return NotImplemented

        # operators: np.add(x, y) -> x + y, np.negative(x) -> -x, etc.
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        kwargs = arraylike._standardize_out_kwarg(**kwargs)
        if "out" in kwargs:
            # compute without out, then copy the result into out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        name = ufunc.__name__
        single = len(inputs) == 1 and inputs[0] is self
        # like numpy, ufunc methods don't skip NA
        if method == "reduce" and single and name in _UFUNC_REDUCTIONS:
            if kwargs.get("axis", 0) in (0, None) and not kwargs.get("keepdims"):
                return self._reduce(_UFUNC_REDUCTIONS[name], skipna=False)
        if method == "accumulate" and single and name in _UFUNC_ACCUMULATIONS:
            if kwargs.get("axis", 0) == 0:
                return self._accumulate(_UFUNC_ACCUMULATIONS[name], skipna=False)

        if method == "__call__" and not kwargs:
            if single and name in _UNARY_UFUNCS:
                na = self.isna()
                num, den = np.where(na, 1, self._num), np.where(na, 1, self._den)
                return self._with_na(_UNARY_UFUNCS[name](num, den), na)
            if len(inputs) == 2 and name in _BINARY_UFUNCS:
                result = self._binary_ufunc(name, *inputs)
                if result is not NotImplemented:
                    return result

        # fallback: compute with floats, e.g. np.sqrt or np.exp
        float_inputs = [
            x._to_float() if isinstance(x, FractionArray) else x for x in inputs
        ]
        return getattr(ufunc, method)(*float_inputs, **kwargs)

    def _binary_ufunc(self, name, left, right):
        """Exact elementwise minimum and maximum."""
        parts = []
        for x in (left, right):
            if x is self:
                parts.append((self._num, self._den))
            else:
                coerced = self._coerce_other(x)
                if not isinstance(coerced, tuple):  # float or unsupported operand
                    return NotImplemented
                parts.append(coerced)
        (a, b), (c, d) = parts
        f, skip_na = _BINARY_UFUNCS[name]
        na_left, na_right = b == 0, d == 0
        na = (na_left & na_right) if skip_na else (na_left | na_right)
        # replace a missing operand by the other one (only matters for fmin/fmax)
        a, b = np.where(na_left, c, a), np.where(na_left, d, b)
        c, d = np.where(na_right, a, c), np.where(na_right, b, d)
        a, b, c, d = (np.where(na, 1, x) for x in (a, b, c, d))
        return self._with_na(f(a, b, c, d), na)


def _part_to_scalar(num, den):
    """Convert a length-1 (numerators, denominators) pair to a Fraction."""
    return fr.Fraction(int(num[0]), int(den[0]))
