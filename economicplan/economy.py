from __future__ import annotations

from pathlib import Path
from typing import Any

import dill
from pydantic import BaseModel, field_validator
from numpy.typing import NDArray
from scipy.sparse import spmatrix


class ShapesNotEqualError(ValueError):
    """The shapes of the matrices difer."""

    def __init__(self) -> None:
        super().__init__("The shapes of the matrices in the same list differ.\n\n")


class ShapeError(ValueError):
    """The shapes of different matrices of the economy don't match."""

    def __init__(self, shape: tuple[int, int], desired_shape: tuple[int, int]) -> None:
        message = f"Shape is {shape}, instead of {desired_shape}.\n\n"
        super().__init__(message)


class Economy(BaseModel):
    model_config = dict(arbitrary_types_allowed=True, extra="allow")

    supply: list[NDArray | spmatrix]
    use_domestic: list[NDArray | spmatrix]
    use_import: list[NDArray | spmatrix]
    depreciation: list[NDArray | spmatrix]
    final_domestic: list[NDArray | spmatrix]
    final_export: list[NDArray | spmatrix]
    prices_import: list[NDArray | spmatrix]
    prices_export: list[NDArray | spmatrix]
    worked_hours: list[NDArray | spmatrix]

    @field_validator("*")
    def assert_equal_shapes(cls, matrices):
        """Assert that all the inputed matrices have the same shape."""
        shapes = [matrix.shape for matrix in matrices]
        if not all([shape == shapes[0] for shape in shapes]):
            raise ShapesNotEqualError
        return matrices

    def model_post_init(self, __context: Any) -> None:
        """Run after initial validation. Validates that the shapes of the
        matrices are compatible with each other (same number of products
        and sectors).
        """
        self.products = self.supply[0].shape[0]
        self.sectors = self.supply[0].shape[1]

        self.validate_matrix_shape(
            self.use_domestic[0], self.use_import[0], shape=(self.products, self.sectors)
        )
        self.validate_matrix_shape(self.depreciation[0], shape=(self.products, self.products))
        self.validate_matrix_shape(
            self.final_domestic[0],
            self.final_export[0],
            self.prices_import[0],
            self.prices_export[0],
            shape=(self.products,),
        )
        self.validate_matrix_shape(self.worked_hours[0], shape=(self.sectors,))

    @staticmethod
    def validate_matrix_shape(*matrices, desired_shape):
        """Assert that all the inputed matrices have the same shape."""
        for matrix in matrices:
            if matrix.shape != desired_shape:
                raise ShapeError(matrix.shape, desired_shape)

    @property
    def products(self) -> int:
        """Number of products in the economy."""
        return self._products

    @products.setter
    def products(self, val: int):
        self._products = val

    @property
    def sectors(self) -> int:
        """Number of products in the economy."""
        return self._sectors

    @sectors.setter
    def sectors(self, val: int):
        self._sectors = val

    @property
    def product_names(self):
        return self._product_names

    @product_names.setter
    def product_names(self, names: list[str]):
        if len(names) != self.products:
            raise ValueError(f"List of names must be of length {self.products}.\n")
        self._product_names = names

    @property
    def sector_names(self):
        return self._sector_names

    @sector_names.setter
    def sector_names(self, names: list[str]):
        if len(names) != self.sectors:
            raise ValueError(f"List of names must be of length {self.sectors}.\n")
        self._sector_names = names

    def save_file(self, path: Path):
        with path.open(mode="wb") as f:
            dill.dump(self, f)


class SpanishEconomy(Economy):
    @classmethod
    def load_excel(
        cls,
        sheet_path: str,
        sheet_name: str,
        min_row: int,
        min_col: int,
        max_row: int,
        max_col: int,
    ) -> Economy:
        """Save the supply-use tables and other data needed for planning."""
        pass
