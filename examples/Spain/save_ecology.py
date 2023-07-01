"""Save the supply-use tables and other data needed for planning."""

from os.path import join
import pickle

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, read_excel
from scipy.sparse import csr_matrix


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


idx_2 = [7, 12, 17, 19, 21, 23, 27, 30, 33, 35, 37, 43, 48, 49]
idx_3 = [0, 5, 10, 16, 52, 57]


def deaggregate(data: NDArray) -> NDArray:
    data = list(data)
    full_data = []
    for i, row in enumerate(data):
        if i in idx_2:
            for _ in range(2):
                full_data.append(row / 2)
        elif i in idx_3:
            for _ in range(3):
                full_data.append(row / 3)
        elif i == 13 or i == 20 or i == 26:
            for _ in range(4):
                full_data.append(row / 4)
        elif i == 3:
            for _ in range(5):
                full_data.append(row / 5)
        elif i == 4:
            for _ in range(8):
                full_data.append(row / 8)
        else:
            full_data.append(row)

    full_data.append(np.zeros(data[0].size))
    return np.array(full_data).T


if __name__ == "__main__":
    print("Starting...")

    MAIN_PATH = join("examples", "Spain", "data")

    excel_path = join(MAIN_PATH, "gases.xlsx")

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

    pollutants = []

    for year in range(3, -1, -1):
        pollutants_year = []
        pollutants_year.append(gei[year])
        pollutants_year.append(co2[year])
        pollutants_year.append(ch4[year])
        pollutants_year.append(n2o[year])
        pollutants.append(np.array(pollutants_year))

    # Interpolate years to have more data
    for i in range(len(pollutants) - 1):
        idx = 2 * i + 1
        pollutants.insert(idx, (pollutants[i + 1] + pollutants[i]) / 2)

    print("shape", pollutants[0].shape)
    print("len", len(pollutants))

    print("Finished.")
