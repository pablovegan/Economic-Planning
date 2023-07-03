"""Basic exception classes used to validate correct inputs in the economy.

Classes:
    ShapesNotEqualError
    ShapeError
"""

import logging


class ShapesNotEqualError(ValueError):
    """The shapes of the matrices difer."""

    def __init__(self) -> None:
        super().__init__("The shapes of the matrices in the same list differ.\n\n")


class ShapeError(ValueError):
    """The shapes of different matrices of the economy don't match."""

    def __init__(self, shape: tuple[int, int], desired_shape: tuple[int, int]) -> None:
        message = f"Shape is {shape}, instead of {desired_shape}.\n\n"
        logging.error(message)
        super().__init__(message)
