"""Save the supply-use tables and other data needed for planning."""

from os.path import join
from pickle import dump

from numpy import eye, zeros
from pandas import DataFrame, read_excel
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
    df.fillna(0, inplace=True)
    return df.to_numpy()


if __name__ == "__main__":

    print("Starting...")

    use_imported, supply_use = [], []
    export_prices, import_prices = [], []
    export_output, target_output = [], []
    imported_prod = []
    worked_hours = []

    excel_names = ["cne_tod_16_en", "cne_tod_17_en", "cne_tod_18_en", "cne_tod_19_en"]

    for name in excel_names:
        excel_path = join("example", "Spain", "data", name + ".xlsx")

        supply_matrix = csr_matrix(load_excel(excel_path, "Table1", 10, 3, 119, 83))
        use_matrix = csr_matrix(load_excel(excel_path, "Table2", 10, 3, 119, 83))
        target_output.append(load_excel(excel_path, "Table2", 10, 95, 119, 95).flatten())

        supply_use.append(supply_matrix - use_matrix)

        use_imported.append(zeros(use_matrix.shape))

        imported_prod.append(load_excel(excel_path, "Table1", 10, 85, 119, 85).flatten())

        export_prices.append(zeros(110))
        import_prices.append(zeros(110))

        export_output.append(load_excel(excel_path, "Table2", 10, 92, 119, 92).flatten())
        
        worked_hours.append(load_excel(excel_path, "Table2", 134, 3, 134, 83).flatten())

    # ! Depreciation matrix != Id may lead to infeasible solutions
    # depreciation = 0.95 * csr_matrix(eye(supply_use[0].shape[0]))
    depreciation = csr_matrix(eye(supply_use[0].shape[0]))
    """
    depreciation[59, 59] = 1  # Suppose CO2 is not reabsorbed
    for i in range(27, 59):
        depreciation[i, i] = 0.3  # Human services cannot be stored for the next period
    """
    # Interpolate years to have more data
    for i in range(len(excel_names) - 1):
        idx = 2 * i + 1
        supply_use.insert(idx, (supply_use[i + 1] + supply_use[i]) / 2)
        use_imported.insert(idx, (use_imported[i + 1] + use_imported[i]) / 2)
        imported_prod.insert(idx, (imported_prod[i + 1] + imported_prod[i]) / 2)
        export_prices.insert(idx, (export_prices[i + 1] + export_prices[i]) / 2)
        import_prices.insert(idx, (import_prices[i + 1] + import_prices[i]) / 2)
        export_output.insert(idx, (export_output[i + 1] + export_output[i]) / 2)
        target_output.insert(idx, (target_output[i + 1] + target_output[i]) / 2)
        worked_hours.insert(idx, (worked_hours[i + 1] + worked_hours[i]) / 2)

    economy = {}

    economy["supply_use"] = supply_use
    economy["use_imported"] = use_imported
    economy["imported_prod"] = imported_prod
    economy["export_prices"] = export_prices
    economy["import_prices"] = import_prices
    economy["export_output"] = export_output
    economy["target_output"] = target_output
    economy["depreciation"] = depreciation
    economy["worked_hours"] = worked_hours

    with open(join("example", "Spain", "data", "spanish_economy.pkl"), "wb") as f:
        dump(economy, f)

    # Now we save the names of each product/sector
    excel_path = join("example", "Spain", "data", "cne_tod_19_en" + ".xlsx")

    product_names = read_excel(
        excel_path,
        sheet_name="Table2",
        usecols=range(2 - 1, 2),
        skiprows=range(10 - 2),
        nrows=119 - 10 + 1,
    )

    activity_names = read_excel(
        excel_path,
        sheet_name="Table2",
        usecols=range(3 - 1, 83),
        skiprows=range(8 - 2),
        nrows=8 - 8 + 1,
    )

    with open(join("example", "Spain", "data", "spanish_product_names.pkl"), "wb") as f:
        dump((product_names, activity_names), f)

    print("Finished.")
