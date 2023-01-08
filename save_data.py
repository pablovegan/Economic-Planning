"""Save the supply-use tables and other data needed for planning."""

from os.path import join
from pickle import dump

import numpy as np
import pandas as pd



excel_path = join('data', 'nrio_sut_181108.xlsx')
sheet_name = '2016'

use_domestic = load_excel(excel_path, sheet_name, 4, 3, 63, 61)
use_imported = load_excel(excel_path, sheet_name, 94, 3, 153, 61)
supply = load_excel(excel_path, sheet_name, 159, 3, 218, 61)

export_prices = load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten()
import_prices = load_excel(excel_path, sheet_name, 4, 80, 63, 80).flatten()

full_target_output = load_excel(excel_path, sheet_name, 4, 79, 63, 79).flatten() * (1/12)
export_vector = load_excel(excel_path, sheet_name, 4, 78, 63, 78).flatten() * (1/12) * 3  # * en el programa de loke se multiplica por el número de time steps
# ! DUDA: Por qué en el target se quita el export???
target_output = full_target_output - export_vector

deprec = array(eye(supply.shape[0]))

worked_hours = load_excel(excel_path, sheet_name, 69, 3, 69, 61).flatten()

with open('data/economy.pkl', 'wb') as file:
    dump((use_domestic,
          use_imported,
          supply,
          export_prices,
          import_prices,
          full_target_output,
          export_vector,
          target_output,
          deprec,
          worked_hours), file)




def load_excel(sheet_path: str,
               sheet_name: str,
               min_row: int,
               min_col: int,
               max_row: int,
               max_col: int
               ) -> pd.DataFrame:
    """Load an excel sheet given the rows and columns."""
    df = pd.read_excel(sheet_path,
                       sheet_name=sheet_name,
                       usecols=range(min_col - 1, max_col),
                       skiprows=range(min_row - 1),
                       nrows=max_row - min_row + 1)
    return df.to_numpy()


if __name__ == '__main__':

    time_steps = 3

    excel_path = join('data', 'nrio_sut_181108.xlsx')
    sheet_name = '2016'

    economy = {}

    supply = load_excel(excel_path, sheet_name, 159, 3, 218, 61)
    use_domestic = load_excel(excel_path, sheet_name, 4, 3, 63, 61)
    economy['use_imported'] = load_excel(excel_path, sheet_name, 94, 3, 153, 61)
    economy['supply_use'] = supply - use_domestic - economy['use_imported']
    economy['depreciation'] = np.array(np.eye(supply.shape[0]))

    full_target_output = load_excel(excel_path, sheet_name, 4, 79, 63, 79) * (1/12)
    economy['export_vector'] = load_excel(excel_path, sheet_name, 4, 78, 63, 78) * time_steps * (1/12)  # noqa: E501
    economy['target_output'] = full_target_output - economy['export_vector']

    economy['export_prices'] = load_excel(excel_path, sheet_name, 4, 80, 63, 80)
    economy['import_prices'] = load_excel(excel_path, sheet_name, 4, 80, 63, 80)

    worked_hours = load_excel(excel_path, sheet_name, 69, 3, 69, 61).reshape([-1, 1])
    economy['cost_coefs'] = worked_hours

    with open(join('data', 'economy.pkl'), 'wb') as file:
        dump(economy, file)
