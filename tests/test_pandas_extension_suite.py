"""
Run pandas' test suite for extension arrays against FractionArray.

The suite checks that FractionArray implements the ExtensionArray interface
consistently with pandas' expectations. It is configured using the fixtures and
the overridden methods of TestFractionArray below.
"""

from fractions import Fraction as F

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.tests.extension import base

from pandas_fractions import FractionArray, FractionDType


def frac(values):
    return FractionArray._from_sequence(values)


@pytest.fixture
def dtype():
    return FractionDType()


@pytest.fixture
def data():
    # 10 nonzero values with small denominators,
    # so that arithmetic neither overflows nor divides by zero
    values = [F(k % 7 + 1, k % 4 + 1) * (-1) ** k for k in range(10)]
    values[0], values[1] = F(1, 2), F(3, 2)
    return frac(values)


@pytest.fixture
def data_for_twos():
    return frac([2] * 10)


@pytest.fixture
def data_missing():
    return frac([None, F(1, 2)])


@pytest.fixture
def data_for_sorting():
    return frac([F(1, 2), F(3, 2), F(-1, 3)])


@pytest.fixture
def data_missing_for_sorting():
    return frac([F(1, 2), None, F(-1, 3)])


@pytest.fixture
def data_for_grouping():
    a, b, c = F(-1, 3), F(1, 2), F(3, 2)
    return frac([b, b, None, None, a, a, b, c])


@pytest.fixture
def na_cmp():
    return lambda x, y: pd.isna(x) and pd.isna(y)


class TestFractionArray(base.ExtensionTests):
    # supported operations (the suite assumes none by default)

    def _get_expected_exception(self, op_name, obj, other):
        return None

    def _supports_reduction(self, ser, op_name):
        return op_name not in ("skew", "kurt")

    def _supports_accumulation(self, ser, op_name):
        return True

    def _cast_pointwise_result(self, op_name, obj, other, pointwise_result):
        # The suite only uses fraction exponents, for which FractionArray returns floats,
        # with NaN for roots of negative numbers (like float Series).
        # pandas' pointwise reference uses Python's Fraction, which can mix
        # fractions, floats, and complex numbers (for roots of negative numbers).
        if "pow" in op_name:
            to_float = lambda v: np.nan if complex(v).imag != 0 else complex(v).real
            return pointwise_result.map(to_float, na_action="ignore").astype(float)
        return pointwise_result

    # The suite computes expected reductions and accumulations with float64,
    # so compare the exact results approximately.

    def check_reduce(self, ser, op_name, skipna):
        if op_name == "count":
            return super().check_reduce(ser, op_name, skipna)
        result = getattr(ser, op_name)(skipna=skipna)
        expected = getattr(ser.astype("float64"), op_name)(skipna=skipna)
        if pd.isna(expected):
            assert pd.isna(result)
        else:
            if isinstance(result, F):
                result = float(result)
            tm.assert_almost_equal(result, expected)

    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ("std", "sem"):
            # The suite expects an ExtensionArray from _reduce(..., keepdims=True),
            # but pandas doesn't unwrap a NumpyExtensionArray in DataFrame reductions,
            # so df.std() would get the dtype NumpyEADtype('float64') instead of float64.
            # FractionArray returns a numpy array instead.
            mark = pytest.mark.xfail(reason="float reductions return numpy arrays")
            request.applymarker(mark)
        super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def _get_expected_reduction_dtype(self, arr, op_name, skipna):
        return np.dtype("float64") if op_name in ("std", "sem") else arr.dtype

    def check_accumulate(self, ser, op_name, skipna):
        result = getattr(ser, op_name)(skipna=skipna)
        expected = getattr(ser.astype("float64"), op_name)(skipna=skipna)
        tm.assert_series_equal(result.astype("float64"), expected, check_exact=False)
