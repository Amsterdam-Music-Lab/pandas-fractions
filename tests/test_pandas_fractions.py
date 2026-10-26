import operator
import statistics
from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

from pandas_fractions import FractionArray  # also registers the "frac" dtype


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
