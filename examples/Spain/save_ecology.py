"""Save the supply-use tables and other data needed for planning."""

from pathlib import Path
import pickle

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, read_excel
from scipy.sparse import csr_matrix

from planning import Ecology, TargetEcology


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


def disaggregate(data: NDArray) -> NDArray:
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

    pollutant_names = []

    pollutant_names.append(load_excel(excel_path, "tabla-0", 8, 1, 8, 1)[0][0])
    gei = list(disaggregate(load_excel(excel_path, "tabla-0", 9, 4, 71, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 75, 1, 75, 1)[0][0])
    co2 = list(disaggregate(load_excel(excel_path, "tabla-0", 76, 4, 138, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 142, 1, 142, 1)[0][0])
    ch4 = list(disaggregate(load_excel(excel_path, "tabla-0", 143, 4, 205, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 209, 1, 209, 1)[0][0])
    n2o = list(disaggregate(load_excel(excel_path, "tabla-0", 210, 4, 272, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 276, 1, 276, 1)[0][0])
    pfc = list(disaggregate(load_excel(excel_path, "tabla-0", 277, 4, 339, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 343, 1, 343, 1)[0][0])
    hfc = list(disaggregate(load_excel(excel_path, "tabla-0", 344, 4, 406, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 410, 1, 410, 1)[0][0])
    sf6 = list(disaggregate(load_excel(excel_path, "tabla-0", 411, 4, 473, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 477, 1, 477, 1)[0][0])
    gac = list(disaggregate(load_excel(excel_path, "tabla-0", 478, 4, 540, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 544, 1, 544, 1)[0][0])
    sox = list(disaggregate(load_excel(excel_path, "tabla-0", 545, 4, 607, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 611, 1, 611, 1)[0][0])
    nox = list(disaggregate(load_excel(excel_path, "tabla-0", 612, 4, 674, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 678, 1, 678, 1)[0][0])
    nh3 = list(disaggregate(load_excel(excel_path, "tabla-0", 679, 4, 741, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 745, 1, 745, 1)[0][0])
    pro3 = list(disaggregate(load_excel(excel_path, "tabla-0", 746, 4, 808, 7)))
    pollutant_names.append(load_excel(excel_path, "tabla-0", 812, 1, 812, 1)[0][0])
    covnm = list(disaggregate(load_excel(excel_path, "tabla-0", 813, 4, 875, 7)))

    pollutant_sector = []

    for year in range(3, -1, -1):
        pollutants_year = []
        pollutants_year.append(gei[year])
        pollutants_year.append(co2[year])
        pollutants_year.append(ch4[year])
        pollutants_year.append(n2o[year])
        pollutants_year.append(pfc[year])
        pollutants_year.append(hfc[year])
        pollutants_year.append(sf6[year])
        pollutants_year.append(gac[year])
        pollutants_year.append(sox[year])
        pollutants_year.append(nox[year])
        pollutants_year.append(nh3[year])
        pollutants_year.append(pro3[year])
        pollutants_year.append(covnm[year])
        pollutant_sector.append(csr_matrix(pollutants_year))

    # Interpolate years to have more data
    for i in range(len(pollutant_sector) - 1):
        idx = 2 * i + 1
        pollutant_sector.insert(idx, (pollutant_sector[i + 1] + pollutant_sector[i]) / 2)

    target_pollutants = []
    for pollutant_year in pollutant_sector:
        # ! INCREASED TARGET POLLUTANT
        target_pollutants.append(1.5 * np.ravel(np.sum(pollutant_year, axis=1)))

    ecology = Ecology(pollutant_sector=pollutant_sector, pollutant_names=pollutant_names)

    target_ecology = TargetEcology(pollutants=target_pollutants)

    with Path(MAIN_PATH, "ecology.pkl").open("wb") as f:
        pickle.dump(ecology.model_dump(), f)

    with Path(MAIN_PATH, "target_ecology.pkl").open("wb") as f:
        pickle.dump(target_ecology.model_dump(), f)

    print("Finished.")
