import pandas as pd
import fractions as fr
from pandas._typing import type_t


@pd.api.extensions.register_extension_dtype
class FractionDType(pd.api.extensions.ExtensionDtype):
    """A pandas dtype for fractions."""

    @property
    def name(self) -> str:
        return "frac"

    @property
    def type(self) -> type_t[fr.Fraction]:
        return fr.Fraction

    @property
    def _is_numeric(self):
        return True

    def construct_array_type(self) -> type_t[FractionArray]:
        return FractionArray


class FractionArray(pd.api.extensions.ExtensionArray):
    pass
