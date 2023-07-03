"""Dataclasses to save the economy and the planned economy returned by the optimizer.
The Economy class is implemented using Pydantic to perform certain checks in
the data, which will normally come from a database, making it prone to mistakes
when loading the data.

Classes:
    Economy
    TargetEconomy
    PlannedEconomy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from ._exceptions import ShapeError, ShapesNotEqualError


ECONOMY_FIELDS = {
    "supply",
    "use_domestic",
    "use_import",
    "depreciation",
    "prices_import",
    "prices_export",
    "worked_hours",
}


MatrixList = list[NDArray] | list[spmatrix]


class Economy(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    supply: list[NDArray] | list[spmatrix]
    use_domestic: list[NDArray] | list[spmatrix]
    use_import: list[NDArray] | list[spmatrix]
    depreciation: list[NDArray] | list[spmatrix]
    prices_import: list[NDArray] | list[spmatrix]
    prices_export: list[NDArray] | list[spmatrix]
    worked_hours: list[NDArray] | list[spmatrix]
    product_names: list[str] | None = None
    sector_names: list[str] | None = None

    @field_validator(*ECONOMY_FIELDS)
    def equal_shapes(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        """Assert that all the inputed matrices have the same shape."""
        shapes = [matrix.shape for matrix in matrices]
        if not all([shape == shapes[0] for shape in shapes]):
            raise ShapesNotEqualError
        logging.info(f"{info.field_name} has shape {shapes[0]}")
        return matrices

    @field_validator(*ECONOMY_FIELDS)
    def equal_periods(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        if "supply" in info.data and len(matrices) != len(info.data["supply"]):
            raise ValueError(
                f"\n{info.field_name} and supply don't have the same number of time periods.\n\n"
            )
        return matrices

    def __post_init__(self) -> None:
        """Run after initial validation. Validates that the shapes of the
        matrices are compatible with each other (same number of products
        and sectors).
        """
        self.validate_matrix_shape(
            self.use_domestic[0], self.use_import[0], shape=(self.products, self.sectors)
        )
        self.validate_matrix_shape(self.depreciation[0], shape=(self.products, self.products))
        self.validate_matrix_shape(
            self.prices_import[0],
            self.prices_export[0],
            shape=(self.products,),
        )
        self.validate_matrix_shape(self.worked_hours[0], shape=(self.sectors,))

        if self.product_names is not None and len(self.product_names) != self.products:
            raise ValueError(f"\nList of PRODUCT names must be of length {self.products}.\n\n")

        if self.sector_names is not None and len(self.product_names) != self.products:
            raise ValueError(f"\nList of SECTOR names must be of length {self.sectors}.\n\n")

    @staticmethod
    def validate_matrix_shape(*matrices: MatrixList, shape: tuple[int, int]) -> None:
        """Assert that all the inputed matrices have the same shape."""
        for matrix in matrices:
            if matrix.shape != shape:
                raise ShapeError(shape, matrix.shape)

    @property
    def products(self) -> int:
        """Number of products in the economy."""
        return self.supply[0].shape[0]

    @property
    def sectors(self) -> int:
        """Number of products in the economy."""
        return self.supply[0].shape[1]

    @property
    def periods(self) -> int:
        """Number of products in the economy."""
        return len(self.supply)


class TargetEconomy(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    domestic: list[NDArray] | list[spmatrix]
    exports: list[NDArray] | list[spmatrix]
    imports: list[NDArray] | list[spmatrix]

    @field_validator("domestic", "exports", "imports")
    def equal_periods(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        if "domestic" in info.data and len(matrices) != len(info.data["domestic"]):
            raise ValueError(
                f"\n{info.field_name} and supply don't have the same number of time periods.\n\n"
            )
        return matrices

    @field_validator("domestic", "exports", "imports")
    def equal_shapes(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        """Assert that all the inputed matrices have the same size."""
        sizes = [matrix.size for matrix in matrices]
        if not all([size == sizes[0] for size in sizes]):
            raise ShapesNotEqualError
        logging.info(f"{info.field_name} has shape {sizes[0]}")
        return matrices

    @field_validator("domestic", "exports", "imports")
    def consistent_shapes(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        if "domestic" in info.data and matrices[0].size != info.data["domestic"][0].size:
            raise ValueError(f"\n{info.field_name} and domestic don't have the same size.\n\n")
        return matrices

    @property
    def products(self) -> int:
        """Number of products in the economy."""
        return self.domestic[0].shape[0]

    @property
    def periods(self) -> int:
        """Number of products in the economy."""
        return len(self.domestic)


@dataclass
class PlannedEconomy:
    """Dataclass that stores the whole planned economy.

    Args:
        activity (list[NDArray]): list with the planned activity for all sectors
            in each period.
        production (list[NDArray]): list with the planned production for all product
            in each period.
        surplus (list[NDArray]): The surplus production at the end of each period.
        total_import (list[NDArray]): list of total imports in each period.
        export_deficit (list[float]): list export deficit at the end of each period.
        worked_hours (list[float]): list of total worked hours in each period.
    """

    activity: list[NDArray] = field(default_factory=list)
    production: list[NDArray] = field(default_factory=list)
    surplus: list[NDArray] = field(default_factory=list)
    total_import: list[NDArray] = field(default_factory=list)
    export_deficit: list[float] = field(default_factory=list)
    worked_hours: list[float] = field(default_factory=list)
