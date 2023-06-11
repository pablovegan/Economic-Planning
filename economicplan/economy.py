from __future__ import annotations

from typing import Any

from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo


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
    """Dataclass with validations that stores the whole economy's information."""

    model_config = dict(arbitrary_types_allowed=True)

    supply: list[NDArray] | list[spmatrix]
    use_domestic: list[NDArray] | list[spmatrix]
    use_import: list[NDArray] | list[spmatrix]
    depreciation: list[NDArray] | list[spmatrix]
    final_domestic: list[NDArray] | list[spmatrix]
    final_export: list[NDArray] | list[spmatrix]
    prices_import: list[NDArray] | list[spmatrix]
    prices_export: list[NDArray] | list[spmatrix]
    worked_hours: list[NDArray] | list[spmatrix]
    product_names: list[str] | None = None
    sector_names: list[str] | None = None

    @field_validator(
        "supply",
        "use_domestic",
        "use_import",
        "depreciation",
        "final_domestic",
        "final_export",
        "prices_import",
        "prices_export",
        "worked_hours",
    )
    def validate_equal_shapes(cls, matrices, info: FieldValidationInfo):
        """Assert that all the inputed matrices have the same shape."""
        shapes = [matrix.shape for matrix in matrices]
        if not all([shape == shapes[0] for shape in shapes]):
            raise ShapesNotEqualError
        print(f"{info.field_name} has shape {shapes[0]}")
        return matrices

    # ! VALIDATE THAT ALL LISTS ARE THE SAME LENGTH root_validator

    def model_post_init(self, __context: Any) -> None:
        """Run after initial validation. Validates that the shapes of the
        matrices are compatible with each other (same number of products
        and sectors).
        """

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

        if self.product_names is not None and len(self.product_names) != self.products:
            raise ValueError(f"List of PRODUCT names must be of length {self.products}.\n\n")

        if self.sector_names is not None and len(self.product_names) != self.products:
            raise ValueError(f"List of SECTOR names must be of length {self.sectors}.\n\n")

    @staticmethod
    def validate_matrix_shape(*matrices, shape):
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


class PlannedEconomy(BaseModel):
    """Dataclass that stores the whole planned economy."""

    model_config = dict(arbitrary_types_allowed=True)

    activity_planned: list[NDArray]
    production_planned: list[NDArray]
    surplus_planned: list[NDArray]
    final_import_planned: list[NDArray]
    export_deficit: list[NDArray]
    worked_hours: list[NDArray]
