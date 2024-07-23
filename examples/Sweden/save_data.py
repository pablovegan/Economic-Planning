"""Save the supply-use tables and other data needed for planning."""

from pathlib import Path
import pickle

import numpy as np
from pandas import DataFrame, Series, concat, read_excel
from scipy.sparse import csr_matrix

from cybersyn import Economy, TargetEconomy


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

    sheet_names = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]

    MAIN_PATH = Path("examples", "Sweden", "data")

    for sheet_name in sheet_names:
        excel_path = Path(MAIN_PATH, "nrio_sut_181108.xlsx")

        supply_matrix = csr_matrix(load_excel(excel_path, sheet_name, 159, 3, 218, 61))
        supply.append(supply_matrix)

        use_domestic_matrix = csr_matrix(load_excel(excel_path, sheet_name, 4, 3, 63, 61))
        use_domestic.append(use_domestic_matrix)

        use_imported_matrix = csr_matrix(load_excel(excel_path, sheet_name, 94, 3, 153, 61))
        use_import.append(use_imported_matrix)

        prices_export.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())
        prices_import.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())

        target_export_vector = load_excel(excel_path, sheet_name, 4, 78, 63, 78).flatten()
        target_export.append(target_export_vector)

        target_import.append(np.zeros_like(target_export_vector))  # no target import

        final_total_vector = load_excel(excel_path, sheet_name, 4, 79, 63, 79).flatten()
        target_domestic_vector = final_total_vector - target_export_vector
        target_domestic.append(target_domestic_vector)

        worked_hours.append(load_excel(excel_path, sheet_name, 69, 3, 69, 61).flatten())

    # ! Depreciation matrix != Id may lead to infeasible solutions
    depreciation = 0.95 * csr_matrix(np.eye(supply[0].shape[0]))
    depreciation[59, 59] = 1  # Suppose CO2 is not reabsorbed
    for i in range(27, 59):
        depreciation[i, i] = 0.3  # Human services cannot be stored for the next period

    # Interpolate years to have more data
    for i in range(len(sheet_names) - 1):
        idx = 2 * i + 1
        supply.insert(idx, (supply[i + 1] + supply[i]) / 2)
        use_domestic.insert(idx, (use_domestic[i + 1] + use_domestic[i]) / 2)
        use_import.insert(idx, (use_import[i + 1] + use_import[i]) / 2)

        prices_export.insert(idx, (prices_export[i + 1] + prices_export[i]) / 2)
        prices_import.insert(idx, (prices_import[i + 1] + prices_import[i]) / 2)

        target_export.insert(idx, (target_export[i + 1] + target_export[i]) / 2)
        target_domestic.insert(idx, (target_domestic[i + 1] + target_domestic[i]) / 2)
        target_import.insert(idx, (target_import[i + 1] + target_import[i]) / 2)

        worked_hours.insert(idx, (worked_hours[i + 1] + worked_hours[i]) / 2)

        # ! Depreciation matrix != Id may lead to infeasible solutions
    # depreciation = 0.95 * csr_matrix(np.eye(supply_use[0].shape[0]))
    depreciation = [csr_matrix(np.eye(supply[0].shape[0]))] * len(supply)
    """
    depreciation[59, 59] = 1  # Suppose CO2 is not reabsorbed
    for i in range(27, 59):
        depreciation[i, i] = 0.3  # Human services cannot be stored for the next period
    """

    product_names = read_excel(
        Path(MAIN_PATH, "posternas_namn.xlsx"),
        sheet_name="SUP10",
        usecols=range(3 - 1, 3),
        skiprows=range(7 - 2),
        nrows=65 - 7 + 1,
    )
    product_names = product_names.squeeze()  # Convert pandas dataframe to series
    product_names = concat([product_names, Series(["CO2"])], ignore_index=True)  # add CO2
    product_names = product_names.tolist()

    economy = Economy(
        supply=supply,
        use_domestic=use_domestic,
        use_import=use_import,
        depreciation=depreciation,
        prices_import=prices_import,
        prices_export=prices_export,
        worked_hours=worked_hours,
        product_names=product_names,
        sector_names=None,
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
