import io
import operator
import statistics
from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

from pandas_fractions import FractionArray, FractionDType  # registers the "frac" dtype


@pytest.fixture
def s():
    return pd.Series([F(1, 2), 3, "2/6", None, 0.25], dtype="frac")


def test_construction(s):
    assert str(s.dtype) == "frac"
    assert s.tolist()[:3] == [F(1, 2), F(3), F(1, 3)]
    assert s.iloc[4] == F(1, 4)


def test_isna_and_nbytes(s):
    assert s.isna().tolist() == [False, False, False, True, False]
    assert s.nbytes == 80


def test_indexing(s):
    assert s[1:3].tolist() == [F(3), F(1, 3)]
    assert s.iloc[[0, 2]].tolist() == [F(1, 2), F(1, 3)]


def test_eq(s):
    assert (s == F(1, 3)).tolist() == [False, False, True, False, False]


def test_reindex_fills_na(s):
    result = s.reindex([0, 9])
    assert result.iloc[0] == F(1, 2)
    assert pd.isna(result.iloc[1])


def test_concat_and_copy(s):
    assert pd.concat([s, s]).shape == (10,)
    assert s.copy().equals(s)


def test_unique_and_value_counts(s):
    uniques = s.unique()
    assert len(uniques) == 5
    assert uniques.isna().tolist() == [False, False, False, True, False]
    counts = s.value_counts()
    assert counts.tolist() == [1, 1, 1, 1]
    assert np.all(counts.index.isin([F(1, 2), F(3), F(1, 3), F(1, 4)]))


def test_groupby_first(s):
    df = pd.DataFrame({"a": s, "k": list("xxyyx")})
    result = df.groupby("k")["a"].first()
    assert result.dtype == s.dtype
    assert result.to_dict() == {"x": F(1, 2), "y": F(1, 3)}


# arithmetic
# ----------

BINARY_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
]


def random_fractions(rng, n, nonzero=False, max_den=1000):
    num = rng.integers(-1000, 1000, n)
    if nonzero:
        num[num == 0] = 1
    den = rng.integers(1, max_den, n)
    return [F(int(a), int(b)) for a, b in zip(num, den)]


def assert_normalized(arr):
    valid = arr._den != 0
    assert np.all(arr._den[valid] > 0)
    assert np.all(np.gcd(arr._num[valid], arr._den[valid]) == 1)


@pytest.mark.parametrize("op", BINARY_OPS)
def test_arith_matches_python(op):
    rng = np.random.default_rng(42)
    xs = random_fractions(rng, 200)
    ys = random_fractions(rng, 200, nonzero=True)
    x = FractionArray._from_sequence(xs)
    y = FractionArray._from_sequence(ys)
    result = op(x, y)
    assert result.tolist() == [op(a, b) for a, b in zip(xs, ys)]
    assert_normalized(result)


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("scalar", [3, -2, F(-5, 7)])
def test_arith_with_scalar(op, scalar):
    xs = [F(1, 2), F(-7, 3), F(5, 4), F(1)]
    x = FractionArray._from_sequence(xs)
    assert op(x, scalar).tolist() == [op(a, scalar) for a in xs]
    assert op(scalar, x).tolist() == [op(scalar, a) for a in xs]


def test_arith_with_int_array():
    x = FractionArray._from_sequence([F(1, 2), F(1, 3)])
    assert (x + np.array([1, 2])).tolist() == [F(3, 2), F(7, 3)]
    assert (np.array([1, 2]) - x).tolist() == [F(1, 2), F(5, 3)]


def test_divmod():
    xs = [F(7, 2), F(-7, 2), F(1, 3)]
    x = FractionArray._from_sequence(xs)
    q, r = divmod(x, F(2, 3))
    assert list(zip(q.tolist(), r.tolist())) == [divmod(a, F(2, 3)) for a in xs]
    q, r = divmod(5, x)
    assert list(zip(q.tolist(), r.tolist())) == [divmod(5, a) for a in xs]


def test_series_arith(s):
    assert (s + s).tolist()[:3] == [F(1), F(6), F(2, 3)]
    assert (s * 2).dtype == s.dtype
    assert (1 / s.dropna()).tolist() == [F(2), F(1, 3), F(3), F(4)]
    assert (s.dropna() - s.dropna()).tolist() == [F(0)] * 4


def test_arith_propagates_na(s):
    for result in [s + 1, 1 - s, s * s, s / F(1, 2), s // 1, s % 1, -s]:
        assert result.isna().tolist() == [False, False, False, True, False]
    assert (s + None).isna().all()
    # NA in the divisor is NA, not a division by zero
    assert (s.array / s.array).isna().tolist() == [False, False, False, True, False]


def test_division_by_zero():
    x = FractionArray._from_sequence([F(1, 2), F(1, 3)])
    with pytest.raises(ZeroDivisionError):
        x / 0
    with pytest.raises(ZeroDivisionError):
        x // FractionArray._from_sequence([1, 0])
    with pytest.raises(ZeroDivisionError):
        1 % FractionArray._from_sequence([0, 1])


def test_overflow():
    big = 2**62
    x = FractionArray._from_sequence([big])
    with pytest.raises(OverflowError):
        x * 2
    with pytest.raises(OverflowError):
        x + x
    with pytest.raises(OverflowError):
        FractionArray._from_sequence([F(1, big)]) + F(1, 3)
    # reducing before multiplying avoids overflow when the result fits
    assert (x * F(1, 2)).tolist() == [F(2**61)]
    assert (FractionArray._from_sequence([F(1, big)]) + F(1, big)).tolist() == [
        F(1, 2**61)
    ]


def test_float_operands(s):
    result = s + 0.5
    assert result.dtype == np.float64
    assert result.tolist()[:3] == [1.0, 3.5, 1 / 3 + 0.5]
    assert np.isnan(result.iloc[3])
    assert (s.array * np.array([2.0] * 5)).dtype == np.float64


def test_pow():
    xs = [F(1, 2), F(-2, 3), F(4)]
    x = FractionArray._from_sequence(xs)
    assert (x**2).tolist() == [F(1, 4), F(4, 9), F(16)]
    assert (x**-1).tolist() == [F(2), F(-3, 2), F(1, 4)]
    assert (x ** np.array([0, 1, 2])).tolist() == [F(1), F(-2, 3), F(16)]
    assert (2**x).dtype == np.float64
    assert (abs(x) ** F(1, 2)).dtype == np.float64  # roots are irrational


def test_unary():
    x = FractionArray._from_sequence([F(1, 2), F(-2, 3), None])
    assert (-x).tolist()[:2] == [F(-1, 2), F(2, 3)]
    assert abs(x).tolist()[:2] == [F(1, 2), F(2, 3)]
    assert (+x).tolist()[:2] == [F(1, 2), F(-2, 3)]
    assert (-x).isna().tolist() == [False, False, True]


# comparison
# ----------

CMP_OPS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


@pytest.mark.parametrize("op", CMP_OPS)
def test_cmp_matches_python(op):
    rng = np.random.default_rng(0)
    xs = random_fractions(rng, 200)
    ys = xs[:50] + random_fractions(rng, 150)  # include some equal pairs
    result = op(FractionArray._from_sequence(xs), FractionArray._from_sequence(ys))
    assert result.dtype == bool
    assert result.tolist() == [op(a, b) for a, b in zip(xs, ys)]


def test_cmp_na(s):
    assert (s == F(1, 3)).tolist() == [False, False, True, False, False]
    assert (s != 3).tolist() == [True, False, True, True, True]
    assert (s < 1).tolist() == [True, False, True, False, True]
    assert (s >= None).tolist() == [False] * 5


def test_cmp_with_float_and_incomparable(s):
    assert (s > 0.4).tolist() == [True, True, False, False, False]
    assert (s == "a").tolist() == [False] * 5
    assert (s != "a").tolist() == [True] * 5
    with pytest.raises(TypeError):
        s < "a"


def test_cmp_large_values():
    # cross-multiplication overflows int64, comparison falls back to Python ints
    big = 2**62
    x = FractionArray._from_sequence([F(big - 1, big)])
    y = FractionArray._from_sequence([F(big - 2, big - 1)])
    assert (x > y).tolist() == [True]
    assert (x == y).tolist() == [False]


# reductions
# ----------


@pytest.mark.parametrize("name", ["sum", "min", "max", "mean", "median", "var"])
def test_reduce_matches_python(name):
    rng = np.random.default_rng(1)
    for n in [1, 2, 7, 64]:
        # small denominators keep the exact result within int64
        xs = random_fractions(rng, n, max_den=13)
        expected = {
            "sum": lambda: sum(xs, F(0)),
            "min": lambda: min(xs),
            "max": lambda: max(xs),
            "mean": lambda: statistics.mean(xs),
            "median": lambda: statistics.median(xs),
            "var": lambda: statistics.variance(xs) if n > 1 else None,
        }[name]()
        result = getattr(pd.Series(xs, dtype="frac"), name)()
        if expected is None:
            assert pd.isna(result)
        else:
            assert result == expected
            assert isinstance(result, F)


def test_reduce_prod():
    xs = [F(1, 2), F(-2, 3), F(9, 4), F(5)]
    assert pd.Series(xs, dtype="frac").prod() == F(-15, 4)


def test_reduce_float_results(s):
    # s without NA: 1/2, 3, 1/3, 1/4
    var = float(s.var())
    assert s.std() == pytest.approx(var**0.5)
    assert s.sem() == pytest.approx((var / 4) ** 0.5)
    assert s.var(ddof=0) == statistics.pvariance([F(1, 2), F(3), F(1, 3), F(1, 4)])


def test_reduce_na(s):
    assert s.sum() == F(49, 12)
    assert pd.isna(s.sum(skipna=False))
    assert pd.isna(s.max(skipna=False))
    assert np.isnan(s.std(skipna=False))
    assert pd.isna(s.sum(min_count=5))
    assert s.sum(min_count=4) == F(49, 12)


def test_reduce_empty():
    empty = pd.Series([], dtype="frac")
    assert empty.sum() == 0
    assert empty.prod() == 1
    for name in ["min", "max", "mean", "median", "var"]:
        assert pd.isna(getattr(empty, name)())
    assert np.isnan(empty.std())


def test_reduce_any_all():
    s = pd.Series([F(0), F(1, 2), None], dtype="frac")
    assert s.any()
    assert not s.all()
    assert not pd.Series([F(0), None], dtype="frac").any()
    assert pd.Series([F(1), None], dtype="frac").all()
    # unskipped NA is truthy, like NaN
    assert pd.Series([F(0), None], dtype="frac").any(skipna=False)


def test_reduce_overflow():
    big = 2**62
    with pytest.raises(OverflowError):
        pd.Series([big, big], dtype="frac").sum()
    # min and max compare exactly instead of overflowing
    s = pd.Series([F(big - 1, big), F(big - 2, big - 1)], dtype="frac")
    assert s.max() == F(big - 1, big)


def test_reduce_unsupported(s):
    with pytest.raises(TypeError):
        s.skew()


def test_dataframe_reductions(s):
    df = pd.DataFrame({"a": s, "b": s * 2})
    result = df.sum()
    assert result.dtype == s.dtype
    assert result.tolist() == [F(49, 12), F(49, 6)]
    assert df.agg(["min", "max"])["b"].tolist() == [F(1, 2), F(6)]


# accumulations
# -------------

ACCUMULATIONS = {
    "cumsum": operator.add,
    "cumprod": operator.mul,
    "cummin": min,
    "cummax": max,
}


def python_accumulate(f, xs, skipna):
    """Reference implementation with pandas' NA semantics."""
    result, acc, seen_na = [], None, False
    for x in xs:
        seen_na = seen_na or x is None
        if x is None or (seen_na and not skipna):
            result.append(None)
            continue
        acc = x if acc is None else f(acc, x)
        result.append(acc)
    return result


@pytest.mark.parametrize("name", list(ACCUMULATIONS))
@pytest.mark.parametrize("skipna", [True, False])
def test_accumulate_matches_python(name, skipna):
    rng = np.random.default_rng(2)
    for n in [1, 2, 7, 33]:
        xs = random_fractions(rng, n, nonzero=True, max_den=7)
        if name == "cumprod":
            xs = [x / abs(x.numerator) for x in xs]  # keep products small
        for i in rng.integers(0, n, n // 5):
            xs[i] = None
        result = getattr(pd.Series(xs, dtype="frac"), name)(skipna=skipna)
        assert result.dtype == "frac"
        expected = python_accumulate(ACCUMULATIONS[name], xs, skipna)
        assert [None if pd.isna(x) else x for x in result] == expected


def test_accumulate_edge_cases():
    empty = pd.Series([], dtype="frac")
    assert empty.cumsum().tolist() == []
    all_na = pd.Series([None, None], dtype="frac")
    assert all_na.cumsum().isna().all()
    leading_na = pd.Series([None, F(1, 2), F(1, 3)], dtype="frac")
    assert leading_na.cummin().tolist()[1:] == [F(1, 2), F(1, 3)]
    assert leading_na.cumsum(skipna=False).isna().all()


def test_accumulate_overflow():
    with pytest.raises(OverflowError):
        pd.Series([2**62, 2**62], dtype="frac").cumsum()


def test_accumulate_dataframe():
    df = pd.DataFrame({"a": pd.Series([F(1, 2), F(1, 3)], dtype="frac")})
    result = df.cumsum()
    assert result["a"].dtype == "frac"
    assert result["a"].tolist() == [F(1, 2), F(5, 6)]


# numpy ufuncs
# ------------


@pytest.fixture
def x():
    return FractionArray._from_sequence([F(7, 2), F(-5, 2), F(5, 2), F(-1, 3), None])


def test_ufunc_operators(x):
    assert np.add(x, 1).tolist()[:4] == [F(9, 2), F(-3, 2), F(7, 2), F(2, 3)]
    assert np.subtract(1, x).tolist()[:4] == [F(-5, 2), F(7, 2), F(-3, 2), F(4, 3)]
    assert np.true_divide(x, 2).dtype == "frac"
    assert np.less(x, 0).tolist() == [False, True, False, True, False]
    assert np.negative(x).tolist()[:4] == [F(-7, 2), F(5, 2), F(-5, 2), F(1, 3)]
    assert np.absolute(x).tolist()[:4] == [F(7, 2), F(5, 2), F(5, 2), F(1, 3)]


@pytest.mark.parametrize("ufunc", [np.floor, np.ceil, np.trunc, np.rint, np.sign])
def test_ufunc_rounding_matches_numpy(ufunc):
    rng = np.random.default_rng(3)
    xs = random_fractions(rng, 200, max_den=8) + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2)]
    result = ufunc(FractionArray._from_sequence(xs))
    assert result.dtype == "frac"
    assert result.tolist() == ufunc(np.array([float(v) for v in xs])).tolist()


def test_ufunc_square_reciprocal(x):
    assert np.square(x).tolist()[:4] == [F(49, 4), F(25, 4), F(25, 4), F(1, 9)]
    assert np.reciprocal(x).tolist()[:4] == [F(2, 7), F(-2, 5), F(2, 5), F(-3)]
    assert np.isnan(np.square(x).astype(float))[4]
    with pytest.raises(ZeroDivisionError):
        np.reciprocal(FractionArray._from_sequence([0]))


def test_ufunc_minimum_maximum(x):
    y = FractionArray._from_sequence([1, F(-3), None, F(-1, 2), 2])
    assert np.minimum(x, y).tolist()[:2] == [F(1), F(-3)]
    assert np.minimum(x, y).isna().tolist() == [False, False, True, False, True]
    assert np.fmin(x, y).tolist() == [F(1), F(-3), F(5, 2), F(-1, 2), F(2)]
    assert np.fmax(x, y).tolist() == [F(7, 2), F(-5, 2), F(5, 2), F(-1, 3), F(2)]
    assert np.maximum(0, x).tolist()[:4] == [F(7, 2), F(0), F(5, 2), F(0)]
    # float operands fall back to floats
    assert np.maximum(x, 0.5).dtype == np.float64


def test_ufunc_reduce_accumulate(x):
    valid = x[:4]
    assert np.add.reduce(valid) == F(19, 6)
    assert np.multiply.reduce(valid) == F(175, 24)
    assert np.minimum.reduce(valid) == F(-5, 2)
    assert np.maximum.reduce(valid) == F(7, 2)
    # like numpy with NaN, NA is not skipped
    assert pd.isna(np.add.reduce(x))
    assert np.add.accumulate(valid).tolist() == [F(7, 2), F(1), F(7, 2), F(19, 6)]
    assert np.add.accumulate(x).isna().tolist() == [False] * 4 + [True]


def test_ufunc_float_fallback(x):
    result = np.sqrt(abs(x))
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[:4], np.sqrt([3.5, 2.5, 2.5, 1 / 3]))
    assert np.isnan(result[4])
    assert np.isnan(x).tolist() == [False] * 4 + [True]
    out = np.empty(5)
    np.exp(x, out=out)
    np.testing.assert_allclose(out[:4], np.exp([3.5, -2.5, 2.5, -1 / 3]))


def test_ufunc_series(x):
    s = pd.Series(x)
    assert isinstance(np.floor(s), pd.Series)
    assert np.floor(s).dtype == "frac"
    assert np.add(s, s).dtype == "frac"
    assert np.sqrt(abs(s)).dtype == np.float64


# item assignment
# ---------------


def test_setitem_keys():
    x = FractionArray._from_sequence([F(1, 2), 3, None, F(1, 4)])
    x[0] = F(2, 3)
    x[1:3] = [1, "5/6"]
    x[np.array([False, False, False, True])] = 0.5
    x[[0, 1]] = FractionArray._from_sequence([F(-1, 3), None])
    assert x.tolist()[2:] == [F(5, 6), F(1, 2)]
    assert x[0] == F(-1, 3)
    assert x.isna().tolist() == [False, True, False, False]
    assert_normalized(x)


def test_setitem_na():
    x = FractionArray._from_sequence([F(1, 2), F(1, 3), F(1, 4)])
    x[0] = None
    x[1] = pd.NA
    x[2] = np.nan
    assert x.isna().all()


def test_setitem_invalid_leaves_array_unchanged():
    x = FractionArray._from_sequence([F(1, 2), F(1, 3)])
    for value in [2**70, F(1, 2**70), "abc"]:
        with pytest.raises((OverflowError, ValueError)):
            x[0] = value
    with pytest.raises(ValueError):
        x[:] = [1, 2, 3]
    assert x.tolist() == [F(1, 2), F(1, 3)]


def test_setitem_slice_is_view():
    x = FractionArray._from_sequence([F(1, 2), F(1, 3), F(1, 4)])
    view = x[1:]
    view[0] = 7
    assert x[1] == F(7)


def test_setitem_series():
    s = pd.Series([F(1, 2), None, 3], dtype="frac")
    assert s.fillna(0).tolist() == [F(1, 2), F(0), F(3)]
    assert s.where(s.isna() | (s > 1), F(-1)).tolist()[::2] == [F(-1), F(3)]
    s.iloc[0] = 7
    s.loc[1] = F(1, 3)
    s[s > 5] = F(5)
    assert s.tolist() == [F(5), F(1, 3), F(3)]
    assert s.dtype == "frac"


def test_ufunc_out_fraction_array():
    x = FractionArray._from_sequence([F(7, 2), F(-5, 2), None])
    np.floor(x, out=x)
    assert x.tolist()[:2] == [F(3), F(-3)]
    assert x.isna()[2]


# strict scalars, strings, and powers
# -----------------------------------


def test_combine_keeps_result_dtype():
    s = pd.Series([F(1, 2), F(3, 2)], dtype="frac")
    assert s.combine(s, lambda a, b: a <= b).dtype == bool
    assert s.combine(s, lambda a, b: float(a)).dtype == np.float64
    assert s.combine(s, operator.add).dtype == "frac"
    with pytest.raises(TypeError):
        FractionArray._from_scalars([True, F(1)], dtype=FractionDType())


def test_read_csv():
    csv = io.StringIO("a\n1/2\n\n3\n-0.25\n")
    df = pd.read_csv(csv, dtype={"a": "frac"}, skip_blank_lines=False)
    assert df["a"].dtype == "frac"
    assert df["a"].isna().tolist() == [False, True, False, False]
    assert df["a"].dropna().tolist() == [F(1, 2), F(3), F(-1, 4)]


def test_pow_float():
    s = pd.Series([F(-1, 4), F(1, 4), None], dtype="frac")
    result = s ** F(1, 2)
    assert result.dtype == np.float64
    # like float Series, roots of negative numbers are NaN
    assert np.isnan(result[0]) and result[1] == 0.5 and np.isnan(result[2])
    # fraction exponents give floats, even if they are integers
    assert (s ** F(2)).dtype == np.float64
    assert (s**s).dtype == np.float64


def test_pow_type_stable():
    # the result type depends only on the exponent's type, not on the values
    empty = pd.Series([], dtype="frac")
    assert (empty**2).dtype == "frac"
    assert (empty ** F(1, 2)).dtype == np.float64
    assert (2**empty).dtype == np.float64


def test_pow_integer_exponents():
    rng = np.random.default_rng(5)
    xs = random_fractions(rng, 100, nonzero=True, max_den=5)
    exponents = rng.integers(-5, 6, 100)
    x = FractionArray._from_sequence(xs)
    assert (x**exponents).tolist() == [a ** int(e) for a, e in zip(xs, exponents)]
    assert (x**-3).tolist() == [a**-3 for a in xs]
    with_na = pd.Series([F(1, 2), None, F(2)], dtype="frac")
    exps = pd.array([2, 3, None], dtype="Int64")
    assert (with_na**exps).isna().tolist() == [False, True, True]
    with pytest.raises(ZeroDivisionError):
        FractionArray._from_sequence([0]) ** -1
    big = FractionArray._from_sequence([F(1, 3)])
    assert (big**39).tolist() == [F(1, 3) ** 39]
    with pytest.raises(OverflowError):
        big**40


def test_count():
    s = pd.Series([F(1, 2), None, F(2)], dtype="frac")
    assert s.array._reduce("count") == 2
    assert s.array._reduce("count", skipna=False) == 2
    assert pd.DataFrame({"a": s}).count().tolist() == [2]


def test_json_roundtrip():
    s = pd.Series([F(1, 2), F(-2, 3), None, F(4)], dtype="frac")
    assert s.to_json() == '{"0":"1\\/2","1":"-2\\/3","2":null,"3":"4"}'
    result = pd.read_json(io.StringIO(s.to_json()), typ="series", dtype="frac")
    pd.testing.assert_series_equal(result, s)


# read-only arrays
# ----------------


def test_readonly():
    x = FractionArray._from_sequence([F(1, 2), None, F(1, 3)])
    assert not x._readonly
    x._readonly = True
    with pytest.raises(ValueError, match="read-only"):
        x[0] = 1
    with pytest.raises(ValueError, match="read-only"):
        x.fillna(0, copy=False)
    assert x[1:]._readonly  # views stay read-only
    assert not x.copy()._readonly
    assert not x[[0, 1]]._readonly  # selections are copies
    assert x.fillna(0).tolist() == [F(1, 2), F(0), F(1, 3)]
    assert x.isna().tolist() == [False, True, False]


# conversion
# ----------


def test_keepdims_reductions():
    s = pd.Series([F(1, 2), F(3, 2)], dtype="frac")
    df = pd.DataFrame({"a": s, "b": s})
    assert df.std().dtype == np.float64
    assert df.sum().dtype == "frac"
    # a missing sum is still a fraction, not a float
    assert df.sum(min_count=3).dtype == "frac"
    assert df.sum(min_count=3).isna().all()


def test_concat_common_dtype():
    s = pd.Series([F(1, 2)], dtype="frac")
    assert pd.concat([s, pd.Series([1, 2])]).dtype == "frac"
    assert pd.concat([s, pd.Series([1, None], dtype="Int64")]).tolist()[:2] == [
        F(1, 2),
        F(1),
    ]
    assert pd.concat([s, pd.Series([1.5])]).dtype == object
    assert pd.concat([s, pd.Series([True])]).dtype == object


def test_to_numpy_and_astype():
    x = FractionArray._from_sequence([F(7, 2), F(-7, 2), None])
    assert x.to_numpy().dtype == object
    assert x.to_numpy().tolist()[:2] == [F(7, 2), F(-7, 2)]
    assert x.to_numpy(na_value=None)[2] is None
    np.testing.assert_array_equal(x.astype(float), [3.5, -3.5, np.nan])
    np.testing.assert_array_equal(np.asarray(x, dtype=np.float32), [3.5, -3.5, np.nan])
    assert x.astype("frac") is not x
    assert x.astype("frac", copy=False) is x
    assert x[:2].astype(int).tolist() == [3, -3]  # truncated, like int(Fraction)
    with pytest.raises(ValueError):
        x.astype(int)
    assert x.astype("string").tolist()[:2] == ["7/2", "-7/2"]
    with pytest.raises(ValueError):
        np.asarray(x, copy=False)


# sorting and factorizing
# -----------------------


def test_sort_exact():
    # distinct fractions that are equal as floats
    big = 2**60
    xs = [F(big + 1, big), F(1), F(big - 1, big), F(big + 1, big), F(-1, 2)]
    assert float(xs[0]) == float(xs[1]) == float(xs[2])
    s = pd.Series(xs, dtype="frac")
    assert s.sort_values().tolist() == sorted(xs)
    assert s.argmax() == 0 and s.argmin() == 4
    assert s.rank().tolist() == [4.5, 3.0, 2.0, 4.5, 1.0]


def test_sort_matches_python():
    rng = np.random.default_rng(4)
    xs = random_fractions(rng, 200, max_den=10)  # with many duplicates
    s = pd.Series(xs + [None], dtype="frac")
    assert s.sort_values().tolist()[:-1] == sorted(xs)
    assert pd.isna(s.sort_values().iloc[-1])
    assert s.sort_values(ascending=False).tolist()[:-1] == sorted(xs, reverse=True)


def test_factorize():
    s = pd.Series([F(1, 2), F(3, 2), None, F(1, 2), F(-1, 3), None], dtype="frac")
    codes, uniques = s.factorize()
    assert codes.tolist() == [0, 1, -1, 0, 2, -1]
    assert uniques.tolist() == [F(1, 2), F(3, 2), F(-1, 3)]
    codes, uniques = s.factorize(use_na_sentinel=False)
    assert codes.tolist() == [0, 1, 2, 0, 3, 2]
    assert uniques.isna().tolist() == [False, False, True, False]
    assert pd.Series([], dtype="frac").factorize()[0].tolist() == []


def test_factorize_based_methods():
    s = pd.Series([F(1, 2), F(3, 2), None, F(1, 2)], dtype="frac")
    assert s.value_counts().to_dict() == {F(1, 2): 2, F(3, 2): 1}
    assert s.duplicated().tolist() == [False, False, False, True]
    df = pd.DataFrame({"k": s, "v": [1, 2, 3, 4]})
    assert df.groupby("k")["v"].sum().to_dict() == {F(1, 2): 5, F(3, 2): 2}
