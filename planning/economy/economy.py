"""
Module docstrings
"""
from numpy import ndarray
from scipy.sparse import csr_matrix  # type: ignore
# from scipy.sparse import spmatrix


class Economy:
    """
    This class contains all the information of our economic plan
    such as supply-use tables and other constraints.
    """

    def __init__(self,
                 supply: ndarray,
                 use_domestic: ndarray,
                 use_imported: ndarray,
                 export_vector: ndarray,
                 export_prices: ndarray,
                 import_prices: ndarray,
                 deprec: ndarray,
                 cost_coefs: ndarray
                 ):
        """
        Parameters
        ----------
        horizon_periods : int
            The number of periods to plan in each iteration.
        revise_periods : int
            The number of periods after which to revise a plan.
        """
        self.supply_use = csr_matrix(supply - use_imported - use_domestic)  # sparse matrix
        self.use_imported = csr_matrix(use_imported)
        self.export_vector = export_vector
        self.export_prices = export_prices
        self.import_prices = import_prices
        self.deprec = csr_matrix(deprec)
        self.cost_coefs = cost_coefs
