"""Save the supply-use tables and other data needed for planning."""

from pathlib import Path
import pickle

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, read_excel
from scipy.sparse import csr_matrix

from cybersyn import Ecology, TargetEcology


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
        engine="openpyxl",
    )
    data = data.fillna(0)
    return data.to_numpy()


idx_2 = [7, 12, 17, 19, 21, 23, 27, 30, 33, 35, 37, 43, 48, 49]
idx_3 = [9]


def deaggregate(data: NDArray) -> NDArray:
    data = list(data)
    full_data = []
    for i, row in enumerate(data):
        if i == 21 or i == 35 or i == 43 or i == 44 or i == 48:
            for _ in range(2):
                full_data.append(row / 2)
        elif i == 5 or i == 26 or i == 30 or i == 52:
            for _ in range(3):
                full_data.append(row / 3)
        elif i == 4:
            for _ in range(5):
                full_data.append(row / 5)
        else:
            full_data.append(row)

    full_data.append(np.zeros(data[0].size))
    return np.array(full_data).T


if __name__ == "__main__":
    print("Starting...")

    MAIN_PATH = Path("examples", "Spain", "data")

    excel_path = Path(MAIN_PATH, "gases.xlsx")

    gei = list(deaggregate(load_excel(excel_path, "tabla-0", 9, 4, 71, 7)))
    co2 = list(deaggregate(load_excel(excel_path, "tabla-0", 76, 4, 138, 7)))
    ch4 = list(deaggregate(load_excel(excel_path, "tabla-0", 143, 4, 205, 7)))
    n2o = list(deaggregate(load_excel(excel_path, "tabla-0", 210, 4, 272, 7)))
    # pfc = deaggregate(load_excel(excel_path, "tabla-0", 277, 4, 339, 7))
    # hfc = deaggregate(load_excel(excel_path, "tabla-0", 344, 4, 406, 7))
    # sf6 = deaggregate(load_excel(excel_path, "tabla-0", 411, 4, 473, 7))
    # gac = deaggregate(load_excel(excel_path, "tabla-0", 478, 4, 540, 7))
    # sox = deaggregate(load_excel(excel_path, "tabla-0", 545, 4, 607, 7))
    # nox = deaggregate(load_excel(excel_path, "tabla-0", 612, 4, 674, 7))
    # nh3 = deaggregate(load_excel(excel_path, "tabla-0", 679, 4, 741, 7))
    # pro3 = deaggregate(load_excel(excel_path, "tabla-0", 746, 4, 808, 7))
    # covnm = deaggregate(load_excel(excel_path, "tabla-0", 813, 4, 875, 7))

    pollutants_sector = []

    for year in range(3, -1, -1):
        pollutants_year = []
        pollutants_year.append(gei[year])
        pollutants_year.append(co2[year])
        pollutants_year.append(ch4[year])
        pollutants_year.append(n2o[year])
        pollutants_sector.append(csr_matrix(pollutants_year))

    # Interpolate years to have more data
    for i in range(len(pollutants_sector) - 1):
        idx = 2 * i + 1
        pollutants_sector.insert(idx, (pollutants_sector[i + 1] + pollutants_sector[i]) / 2)

    target_pollutants = []
    for pollutant_year in pollutants_sector:
        # ! INCREASED TARGET POLLUTANT
        target_pollutants.append(1.5 * np.ravel(np.sum(pollutant_year, axis=1)))

    ecology = Ecology(pollutants_sector=pollutants_sector)

    target_ecology = TargetEcology(pollutants=target_pollutants)

    with Path(MAIN_PATH, "ecology.pkl").open("wb") as f:
        pickle.dump(ecology.model_dump(), f)

    with Path(MAIN_PATH, "target_ecology.pkl").open("wb") as f:
        pickle.dump(target_ecology.model_dump(), f)

    print("Finished.")
