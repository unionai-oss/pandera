# pylint: skip-file
"""Regression coverage for Series annotations outside DataFrameModel (#2018)."""

from pandera.typing import Series


class Wrapper:
    a: Series

    def __init__(self, a: Series) -> None:
        self.a = a
