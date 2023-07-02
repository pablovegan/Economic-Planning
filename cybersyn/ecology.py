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

from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pydantic import BaseModel


ECOLOGY_FIELDS = {
    "pollutants",
}


MatrixList = list[NDArray] | list[spmatrix]


class Ecology(BaseModel):
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    pollutants: list[NDArray] | list[spmatrix]
    target_pollutants: list[NDArray] | list[spmatrix]
    pollutant_names: list[str] | None = None

    @property
    def num_pollutants(self) -> int:
        """Number of products in the economy."""
        return self.pollutants[0].shape[0]
