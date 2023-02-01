"""Save the supply-use tables and other data needed for planning."""

from os.path import join
from pickle import dump

from numpy import eye, zeros
from pandas import DataFrame, Series, read_excel, concat
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

    use_imported, supply_use = [], []
    export_prices, import_prices = [], []
    export_output, target_output = [], []
    imported_prod = []
    worked_hours = []

    sheet_names = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]

    for sheet_name in sheet_names:
        excel_path = join("example", "Sweden", "data", "nrio_sut_181108.xlsx")

        supply_matrix = csr_matrix(load_excel(excel_path, sheet_name, 159, 3, 218, 61))

        use_imported_matrix = csr_matrix(load_excel(excel_path, sheet_name, 94, 3, 153, 61))
        use_imported.append(use_imported_matrix)

        use_domestic_matrix = csr_matrix(load_excel(excel_path, sheet_name, 4, 3, 63, 61))

        supply_use.append(supply_matrix - use_imported_matrix - use_domestic_matrix)

        export_prices.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())
        import_prices.append(load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten())

        target_output.append(load_excel(excel_path, sheet_name, 4, 79, 63, 79).flatten())
        export_output.append(load_excel(excel_path, sheet_name, 4, 78, 63, 78).flatten())
        worked_hours.append(load_excel(excel_path, sheet_name, 69, 3, 69, 61).flatten())

        imported_prod.append(zeros(supply_use[0].shape[0]))

    # ! Depreciation matrix != Id may lead to infeasible solutions
    depreciation = 0.95 * csr_matrix(eye(supply_use[0].shape[0]))
    depreciation[59, 59] = 1  # Suppose CO2 is not reabsorbed
    for i in range(27, 59):
        depreciation[i, i] = 0.3  # Human services cannot be stored for the next period

    # Interpolate years to have more data
    for i in range(len(sheet_names) - 1):
        idx = 2 * i + 1
        supply_use.insert(idx, (supply_use[i + 1] + supply_use[i]) / 2)
        use_imported.insert(idx, (use_imported[i + 1] + use_imported[i]) / 2)
        export_prices.insert(idx, (export_prices[i + 1] + export_prices[i]) / 2)
        import_prices.insert(idx, (import_prices[i + 1] + import_prices[i]) / 2)
        export_output.insert(idx, (export_output[i + 1] + export_output[i]) / 2)
        target_output.insert(idx, (target_output[i + 1] + target_output[i]) / 2)
        worked_hours.insert(idx, (worked_hours[i + 1] + worked_hours[i]) / 2)
        imported_prod.insert(idx, (imported_prod[i + 1] + imported_prod[i]) / 2)  # ! Unnecessary for swedish data!!

    economy = {}

    economy["supply_use"] = supply_use
    economy["use_imported"] = use_imported
    economy["export_prices"] = export_prices
    economy["import_prices"] = import_prices
    economy["export_output"] = export_output
    economy["target_output"] = target_output
    economy["depreciation"] = depreciation
    economy["worked_hours"] = worked_hours
    economy["imported_prod"] = imported_prod  # ! Unnecessary for swedish data!!

    with open(join("example", "Sweden", "data", "swedish_economy.pkl"), "wb") as f:
        dump(economy, f)

    # Now we save the names of each product/sector
    excel_path = join("example", "Sweden", "data", "posternas_namn.xlsx")
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
    with open(join("example", "Sweden", "data", "swedish_product_names.pkl"), "wb") as f:
        dump(product_names, f)

    print("Finished.")
