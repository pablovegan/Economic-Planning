"""Save the supply-use tables and other data needed for planning."""

from os.path import join
from pickle import dump

import numpy as np
from pandas import DataFrame, Series, concat, read_excel
from scipy.sparse import csr_matrix


def load_excel(
    sheet_path: str, sheet_name: str, min_row: int, min_col: int, max_row: int, max_col: int
) -> DataFrame:
    """Load an excel sheet given the rows and columns."""
    df = read_excel(
        sheet_path,
        sheet_name=sheet_name,
        usecols=range(min_col - 1, max_col),
        skiprows=range(min_row - 2),
        nrows=max_row - min_row + 1,
    )
    return df.to_numpy()


if __name__ == "__main__":
    print("Starting...")

    supply, use_domestic, use_imported = [], [], []
    final_export, final_domestic = [], []
    prices_export, prices_import = [], []
    worked_hours = []

    sheet_names = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]

    MAIN_PATH = join("examples", "Sweden", "data")

    for sheet_name in sheet_names:
        excel_path = join(MAIN_PATH, "nrio_sut_181108.xlsx")

        supply_matrix = csr_matrix(load_excel(excel_path, sheet_name, 159, 3, 218, 61))
        supply.append(supply_matrix)

        use_domestic_matrix = csr_matrix(load_excel(excel_path, sheet_name, 4, 3, 63, 61))
        use_domestic.append(use_domestic_matrix)

        use_imported_matrix = csr_matrix(load_excel(excel_path, sheet_name, 94, 3, 153, 61))
        use_imported.append(use_imported_matrix)

        prices_export.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())
        prices_import.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())

        final_export_vector = load_excel(excel_path, sheet_name, 4, 78, 63, 78).flatten()
        final_export.append(final_export_vector)

        final_total_vector = load_excel(excel_path, sheet_name, 4, 79, 63, 79).flatten()
        final_domestic_vector = final_total_vector - final_export_vector
        final_domestic.append(final_domestic_vector)

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
        use_imported.insert(idx, (use_imported[i + 1] + use_imported[i]) / 2)

        prices_export.insert(idx, (prices_export[i + 1] + prices_export[i]) / 2)
        prices_import.insert(idx, (prices_import[i + 1] + prices_import[i]) / 2)

        final_export.insert(idx, (final_export[i + 1] + final_export[i]) / 2)
        final_domestic.insert(idx, (final_domestic[i + 1] + final_domestic[i]) / 2)

        worked_hours.insert(idx, (worked_hours[i + 1] + worked_hours[i]) / 2)

    economy = {}

    economy["supply"] = supply
    economy["use_domestic"] = use_domestic
    economy["use_import"] = use_imported

    economy["final_export"] = final_export
    economy["final_domestic"] = final_domestic

    economy["prices_export"] = prices_export
    economy["prices_import"] = prices_import

    economy["depreciation"] = depreciation
    economy["worked_hours"] = worked_hours

    with open(join(MAIN_PATH, "swedish_economy.pkl"), "wb") as f:
        dump(economy, f)

    # Now we save the names of each product/sector
    excel_path = join(MAIN_PATH, "posternas_namn.xlsx")
    product_names = read_excel(
        excel_path,
        sheet_name="SUP10",
        usecols=range(3 - 1, 3),
        skiprows=range(7 - 2),
        nrows=65 - 7 + 1,
    )
    # Convert pandas dataframe to series
    product_names = product_names.squeeze()
    product_names = concat([product_names, Series(["CO2"])], ignore_index=True)  # add CO2
    with open(join(MAIN_PATH, "swedish_product_names.pkl"), "wb") as f:
        dump(product_names, f)

    print("Finished.")
