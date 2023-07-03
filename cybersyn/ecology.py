"""
Dataclasses to save the economy and the planned economy returned by the optimizer.
The Economy class is implemented using Pydantic to perform certain checks in
the data, which will normally come from a database, making it prone to mistakes
when loading the data.

Classes:
    ShapesNotEqualError
    ShapeError
    Economy
    PlannedEconomy
"""

from __future__ import annotations

import logging

from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from ._exceptions import ShapesNotEqualError


MatrixList = list[NDArray] | list[spmatrix]

ECOLOGY_FIELDS = {
    "pollutants_sector",
}


class Ecology(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    pollutants_sector: list[NDArray] | list[spmatrix]
    pollutant_names: list[str] | None = None

    @property
    def num_pollutants(self) -> int:
        """Number of products in the economy."""
        return self.pollutants_sector[0].shape[0]


class TargetEcology(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    pollutants: list[NDArray] | list[spmatrix]

    @field_validator("pollutants")
    def equal_periods(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        if "pollutants" in info.data and len(matrices) != len(info.data["pollutants"]):
            raise ValueError(
                f"\n{info.field_name} and supply don't have the same number of time periods.\n\n"
            )
        return matrices

    @field_validator("pollutants")
    def equal_sizes(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        """Assert that all the inputed matrices have the same size."""
        sizes = [matrix.size for matrix in matrices]
        if not all([size == sizes[0] for size in sizes]):
            raise ShapesNotEqualError
        logging.info(f"{info.field_name} has size {sizes[0]}")
        return matrices

    @field_validator("pollutants")
    def consistent_shapes(cls, matrices: MatrixList, info: FieldValidationInfo) -> MatrixList:
        if "pollutants" in info.data and matrices[0].size != info.data["pollutants"][0].size:
            raise ValueError(f"\n{info.field_name} and pollutants don't have the same size.\n\n")
        return matrices

    @property
    def num_pollutants(self) -> int:
        """Number of products in the economy."""
        return self.pollutants[0].size

    @property
    def periods(self) -> int:
        """Number of products in the economy."""
        return len(self.pollutants)
