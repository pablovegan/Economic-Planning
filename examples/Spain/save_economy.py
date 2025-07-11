"""Save the supply-use tables and other data needed for planning."""

from pathlib import Path
import pickle

import numpy as np
from pandas import DataFrame, read_excel
from scipy.sparse import csr_matrix

from planning import Economy, TargetEconomy


def load_excel(
    sheet_path: str, sheet_name: str, min_row: int, min_col: int, max_row: int, max_col: int
) -> DataFrame:
    """Load an excel sheet given the rows and columns."""
    data = read_excel(
        sheet_path,
        sheet_name=sheet_name,
        usecols=range(min_col - 1, max_col),
        skiprows=range(min_row - 2),
        nrows=max_row - min_row + 1,
    )
    data = data.fillna(0)
    return data.to_numpy()


if __name__ == "__main__":
    print("Starting...")

    supply, use_domestic, use_import = [], [], []
    target_export, target_domestic, target_import = [], [], []
    prices_export, prices_import = [], []
    worked_hours = []

    MAIN_PATH = Path("examples", "Spain", "data")

    excel_names = ["cne_tod_16_en", "cne_tod_17_en", "cne_tod_18_en", "cne_tod_19_en"]

    for name in excel_names:
        excel_path = Path(MAIN_PATH, name + ".xlsx")

        supply_matrix = csr_matrix(load_excel(excel_path, "Table1", 10, 3, 119, 83))
        supply.append(supply_matrix)

        total_import_vector = load_excel(excel_path, "Table1", 10, 85, 119, 85).flatten()

        # ! This is the complete use matrix, because use_import is not available!!
        use_domestic_matrix = csr_matrix(load_excel(excel_path, "Table2", 10, 3, 119, 83))
        use_domestic.append(use_domestic_matrix)
        # ! Same note as before!!
        use_import_matrix = csr_matrix(np.zeros(use_domestic_matrix.shape))
        use_import.append(use_import_matrix)

        target_export_vector = load_excel(excel_path, "Table2", 10, 92, 119, 92).flatten()
        target_export.append(target_export_vector)

        target_import.append(np.zeros_like(target_export_vector))

        # ? Check this is correct in the table
        # final uses includes exports
        final_uses_vector = load_excel(excel_path, "Table2", 10, 95, 119, 95).flatten()
        target_domestic.append(final_uses_vector - target_export_vector)

        prices_export.append(np.ones(supply_matrix.shape[0]))
        prices_import.append(np.ones(supply_matrix.shape[0]))

        worked_hours.append(load_excel(excel_path, "Table2", 134, 3, 134, 83).flatten())

    # Interpolate years to have more data
    for i in range(len(excel_names) - 1):
        idx = 2 * i + 1
        supply.insert(idx, (supply[i + 1] + supply[i]) / 2)
        use_domestic.insert(idx, (use_domestic[i + 1] + use_domestic[i]) / 2)
        use_import.insert(idx, (use_import[i + 1] + use_import[i]) / 2)

        target_domestic.insert(idx, (target_domestic[i + 1] + target_domestic[i]) / 2)
        target_export.insert(idx, (target_export[i + 1] + target_export[i]) / 2)
        target_import.insert(idx, (target_import[i + 1] + target_import[i]) / 2)

        prices_export.insert(idx, (prices_export[i + 1] + prices_export[i]) / 2)
        prices_import.insert(idx, (prices_import[i + 1] + prices_import[i]) / 2)

        worked_hours.insert(idx, (worked_hours[i + 1] + worked_hours[i]) / 2)

    # ! Depreciation matrix != Id may lead to infeasible solutions
    # depreciation = 0.95 * csr_matrix(np.eye(supply_use[0].shape[0]))
    depreciation = [csr_matrix(np.eye(supply[0].shape[0]))] * len(supply)
    """
    depreciation[59, 59] = 1  # Suppose CO2 is not reabsorbed
    for i in range(27, 59):
        depreciation[i, i] = 0.3  # Human services cannot be stored for the next period
    """

    # Now we save the names of each product/sector
    excel_path = Path(MAIN_PATH, "cne_tod_19_en" + ".xlsx")
    product_names = read_excel(
        excel_path,
        sheet_name="Table2",
        usecols=range(2 - 1, 2),
        skiprows=range(10 - 2),
        nrows=119 - 10 + 1,
    )
    product_names = product_names.squeeze().tolist()
    sector_names = read_excel(
        excel_path,
        sheet_name="Table2",
        usecols=range(3 - 1, 83),
        skiprows=range(8 - 2),
        nrows=8 - 8 + 1,
    )
    sector_names = sector_names.squeeze().tolist()

    economy = Economy(
        supply=supply,
        use_domestic=use_domestic,
        use_import=use_import,
        depreciation=depreciation,
        prices_import=prices_import,
        prices_export=prices_export,
        worked_hours=worked_hours,
        product_names=product_names,
        sector_names=sector_names,
    )

    target_economy = TargetEconomy(
        domestic=target_domestic,
        imports=target_import,
        exports=target_export,
    )

    with Path(MAIN_PATH, "economy.pkl").open("wb") as f:
        pickle.dump(economy.model_dump(), f)

    with Path(MAIN_PATH, "target_economy.pkl").open("wb") as f:
        pickle.dump(target_economy.model_dump(), f)

    print("Finished.")
