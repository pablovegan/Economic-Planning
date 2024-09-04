"""Dataclasses to save the ecology and the planned ecology.

Classes:
    Ecology
    PlannedEcology

# TODO:
    Add PlannedEcology class or merge it with planned economy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from ._exceptions import ShapesNotEqualError


MatrixList = list[NDArray] | list[spmatrix]

ECOLOGY_FIELDS = {
    "pollutant_sector",
}


class Ecology(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    pollutant_sector: list[NDArray] | list[spmatrix]
    pollutant_names: list[str] | None = None

    @property
    def num_pollutants(self) -> int:
        """Number of products in the economy."""
        return self.pollutant_sector[0].shape[0]


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


@dataclass
class PlannedEcology:
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

    pollutants: list[NDArray] = field(default_factory=list)
