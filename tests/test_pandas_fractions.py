from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

import pandas_fractions  # noqa: F401  (registers the "frac" dtype)


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
