# pandas_fractions

A pandas extension dtype (`"frac"`) for exact rational numbers.

```python
from fractions import Fraction
import pandas as pd
import pandas_fractions

s = pd.Series([Fraction(1, 2), 3, "2/6", None], dtype="frac")
s + Fraction(1, 6)  # [2/3, 19/6, 1/2, NaN], dtype: frac
```

This document describes the design of `FractionArray`, in particular its arithmetic.

## Representation

A `FractionArray` stores its values in two `int64` arrays, `_num` (numerators) and `_den` (denominators).

- **Normalized values.**
  Every fraction is kept in lowest terms with a positive denominator, just like Python's `Fraction`.
  Every constructor and operation preserves this invariant.
  As a consequence, each value has exactly one representation,
  so equality is a plain comparison of the two arrays.
- **Missing values** are marked by a denominator of 0 (stored as `0/0`).
  Since no valid fraction has a zero denominator, no separate mask is needed:
  `isna()` is just `_den == 0`.
  Reading a missing element returns the dtype's `na_value` (`NaN`).
- **Bounded integers.**
  Unlike Python's `Fraction`, numerators and denominators are limited to `int64`.
  Values that don't fit raise an `OverflowError`,
  both when constructing an array and during arithmetic (see below).

## General design of the operations

### Vectorized and exact

Operations work directly on the numerator and denominator arrays using numpy,
instead of converting each element to a Python `Fraction`.
The results are exact, and always equal to what Python's `Fraction` would compute,
as long as no value exceeds `int64`.
The only exception is `**`, which is computed elementwise (see [Powers](#powers)).

Operators are provided through pandas' `OpsMixin` (from `pandas.core.arraylike`).
It defines all Python operator methods (`__add__`, `__radd__`, `__lt__`, ...)
and forwards them to two methods:

- `_arith_method(other, op)` for arithmetic,
- `_cmp_method(other, op)` for comparisons,

where `op` is a function from `operator` (e.g. `operator.add`)
or `pandas.core.roperator` for reversed operations (e.g. `roperator.radd` for `1 + x`).
The mixin also returns `NotImplemented` when the other operand is a `Series`, `Index`, or `DataFrame`,
so that pandas can unwrap it and call the operation again on the underlying array.
Note that `OpsMixin` and `roperator` are not part of pandas' public API.

### Overflow handling

numpy's integer arithmetic wraps around silently on overflow,
which would produce wrong fractions without any warning.
Silently wrong results are worse than an error,
so all integer operations that can overflow go through checked helpers
that raise an `OverflowError` instead:

| Helper          | Detects overflow by                                                        |
|-----------------|----------------------------------------------------------------------------|
| `_checked_mul`  | dividing the product by one factor and comparing it to the other factor    |
| `_checked_add`  | both summands having the same sign, but the result having a different sign |
| `_checked_neg`  | the argument being the smallest `int64` (whose negation doesn't exist)     |

All three checks are exact: there are no false positives.
`_checked_mul` additionally handles the case `-1 * INT64_MIN`,
where the check itself would overflow.

To avoid unnecessary overflows, operations reduce common factors *before* multiplying,
following the implementation of CPython's `Fraction`.
This keeps intermediate values as small as possible,
so that an operation only fails if its inputs or result are actually too large.
For example, `2**62 * (1/2)` works, although `2**62 * 1` already uses most of the `int64` range.

Comparisons are handled differently, see [Comparisons](#comparisons).

### Missing values

Missing values propagate like `NaN` (or pandas' nullable types):

- in arithmetic, the result is missing if either operand is missing;
- in comparisons, the result is `False`, except for `!=`, which is `True`.

To implement this, `_arith_method` first computes the mask of missing results,
then replaces the missing entries in *both* operands by `1/1`,
computes the operation on all elements,
and finally sets the masked entries of the result back to `0/0`.
Replacing missing values by `1` rather than `0` is important,
because a missing divisor must produce a missing result, not a `ZeroDivisionError`.
The replacement also keeps all inputs normalized,
which the operations rely on.

A missing scalar operand (`None`, `np.nan`, `pd.NA`) is treated as an array of missing values,
so `x + None` is entirely missing.

### Division by zero

Dividing by an actual (non-missing) zero raises a `ZeroDivisionError`, just like Python's `Fraction`.
This applies to `/`, `//`, `%`, and `divmod`,
and to any element of the divisor, not just scalars.
pandas' nullable integer types return a missing value or infinity instead,
but zero denominators are reserved for missing values here,
and silently turning a division by zero into a missing value would hide errors.

### Operand types

`_coerce_other` converts the other operand of a binary operation into one of three forms:

1. **A pair of numerator and denominator arrays** of the same length as `self`.
   This is the case for `FractionArray`s, scalar `int`s and `Fraction`s (which are broadcast),
   integer or boolean numpy arrays (with denominator 1),
   and any other sequence that `FractionArray._from_sequence` accepts (e.g. lists of `Fraction`s).
2. **A float scalar or array**, for `float` operands and float arrays.
   The operation then converts `self` to floats (with missing values as `NaN`)
   and returns a float64 numpy array.
   This mirrors Python, where `Fraction(1, 2) + 0.5` is the float `1.0`.
   Converting the float to an exact fraction instead would usually produce huge denominators
   (`Fraction(0.1) == Fraction(3602879701896397, 36028797018963968)`) and overflow.
3. **`NotImplemented`**, for anything else (e.g. strings),
   which lets Python try the reversed operation and finally raise a `TypeError`.

Operands of a different length raise a `ValueError`.

## Details on specific operations

All of the following functions take the numerators and denominators of both operands
as four arrays `a, b, c, d`, representing `a/b` and `c/d`.
They expect normalized inputs without missing values (`b, d > 0`)
and return normalized results.

### Addition and subtraction

Instead of computing `(a*d + c*b) / (b*d)` and reducing afterwards,
`_add` divides out the common factor of the denominators first:

```
g  = gcd(b, d)
s  = b / g
t  = a * (d / g) + c * s
g2 = gcd(t, g)
result = (t / g2) / (s * (d / g2))
```

The result is already in lowest terms, since any common factor of `t` and the denominator
must divide `g`.
In particular, the denominator never exceeds `lcm(b, d)`,
whereas the naive formula produces `b*d`.

`_sub` adds the negated second operand.

### Multiplication and division

`_mul` cancels common factors crosswise before multiplying:

```
g1 = gcd(a, d)
g2 = gcd(c, b)
result = ((a / g1) * (c / g2)) / ((b / g2) * (d / g1))
```

Since both inputs are in lowest terms, this result is in lowest terms as well.

`_truediv` checks the divisor for zeros, multiplies by its reciprocal `d/c`,
and moves the sign of `c` to the numerator to keep the denominator positive.

### Floor division, modulo, and divmod

These follow Python's semantics for fractions (rounding towards negative infinity):

- `x // y` computes `x / y` with `_truediv` and floor-divides its numerator by its denominator.
  The result is returned as a `FractionArray` with denominator 1
  (Python returns an `int`, but a `FractionArray` keeps the dtype and supports missing values).
- `x % y` is computed as `x - (x // y) * y`, so the result has the sign of `y`, as in Python.
- `divmod(x, y)` returns the tuple `(x // y, x % y)` of two `FractionArray`s.

### Reversed operations

For reversed operations like `1 - x`, `_arith_method` swaps the two operands
and looks up the corresponding forward operation in `_REVERSED_OPS` (e.g. `rsub` → `sub`).
Thus, each operation is only implemented once.
For float operands, the reversed operator is applied directly to the float array,
as `roperator` already takes care of the operand order.

### Powers

`**` (and its reverse) is the only operation that is computed elementwise
with Python's `Fraction`, in `_pointwise`.
Its result type depends on the values:
integer exponents produce fractions,
but other exponents usually produce irrational numbers
(`Fraction(2) ** Fraction(1, 2)` is a float),
so a vectorized implementation would gain little.
If any element produces a float, the whole result is returned as a float64 numpy array,
otherwise as a `FractionArray`, where too large values raise an `OverflowError` during construction.
Missing values propagate as usual.

### Comparisons

Since denominators are positive, `a/b < c/d` holds exactly if `a*d < c*b`,
and similarly for the other comparison operators.
`_cmp_method` compares these cross products with checked multiplication.
Unlike arithmetic, a comparison always has a representable result,
so instead of raising an `OverflowError`,
it falls back to computing the cross products with Python integers (numpy `object` arrays).
This is slower, but exact.

Comparisons with floats convert the fractions to floats.
Comparisons with incomparable values (e.g. strings) return all `False` for `==`,
all `True` for `!=`, and raise a `TypeError` for ordering comparisons, consistent with Python.

### Unary operations

`-x` and `abs(x)` negate the numerators with an overflow check
(the denominators are unaffected),
and `+x` returns a copy.
Missing values stay missing, since their denominator remains 0.

## Reductions

Reductions like `s.sum()` or `df.mean()` are implemented in `_reduce`,
the interface pandas uses for all reductions of extension arrays.
`_reduce` computes a scalar in `_reduce_scalar`,
and wraps it in an array of length 1 if pandas asks for `keepdims=True`
(which it does for `DataFrame` reductions, so that the result column keeps the `frac` dtype).

### Result types

Reductions whose result is always rational return an exact `Fraction`:
`sum`, `prod`, `min`, `max`, `mean`, `median`, and `var`.
`std` and `sem` involve a square root, so they return a `float`, computed from the exact variance.
`any` and `all` return a `bool`.
`skew` and `kurt` are not supported and raise a `TypeError`.

### Missing values and empty inputs

- With `skipna=True` (the default), missing values are ignored.
- With `skipna=False`, any missing value makes the result missing
  (`NaN` for the float-valued `std` and `sem`).
- If no values are left, `sum` and `prod` return their neutral elements `0` and `1`,
  and all other reductions return a missing value.
  `min_count` works as for other dtypes: `sum` and `prod` return a missing value
  if fewer than `min_count` values are present.
- `var`, `std`, and `sem` are missing if there are no more values than `ddof` (default 1).
- `any` and `all` treat missing values like `NaN`:
  they are skipped with `skipna=True` and count as true (nonzero) otherwise.

### Vectorized tree reduction

`sum`, `prod`, `min`, and `max` reuse the binary operations on parts (`_add`, `_mul`, `_min`, `_max`)
in `_tree_reduce`:
it combines the first half of the array with the second half elementwise,
and repeats this until a single value is left.
This needs only about log₂(n) vectorized steps instead of n Python-level steps.
If the remaining length is odd, the array is padded with the neutral element of the operation
(`0/1` for `sum`, `1/1` for `prod`), or with a copy of the last element for `min` and `max`.

Overflows are handled exactly like in arithmetic, since the same checked operations are used.
Note that `sum` (and therefore `mean` and `var`) can overflow quite easily:
the denominator of a sum is the least common multiple of all denominators,
which exceeds `int64` for many unrelated denominators, even if the individual values are small.
Values that share a common grid of denominators (like note durations or onsets) are no problem.
`min` and `max` use the same exact comparison as the comparison operators,
which falls back to Python integers instead of overflowing.

### Other reductions

- `mean` is the sum divided by the number of values.
- `median` sorts the values as Python `Fraction`s, which is exact but not vectorized,
  and averages the two middle values for an even number of values.
- `var` computes the exact squared deviations from the exact mean,
  sums them with the tree reduction, and divides by `n - ddof`.
- `std` is the square root of `var` as a float, and `sem` is `std / sqrt(n)`.

## Not yet implemented

- numpy ufuncs (via `__array_ufunc__`), e.g. `np.add(x, y)`.
- Cumulative operations like `cumsum` (via `_accumulate`).
