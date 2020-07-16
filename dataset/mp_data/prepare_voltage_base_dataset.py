import argparse
from os import path, makedirs, getcwd
from datetime import datetime
from joblib import Parallel, delayed

import h5py
import pandas as pd
from tqdm import tqdm
from pymatgen import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter


def get_battery_data_from_battery_id(battery_id, mp_dr, key_for_data):
    # https://discuss.materialsproject.org/t/accessing-battery-database-through-pymatgen/295/2
    batt_data = mp_dr._make_request('/battery/%s' % battery_id)
    all_data = batt_data[0]['adj_pairs']
    if len(all_data) == 1:
        data = all_data[0]
        val = [battery_id] + [data.get(key) for key in key_for_data]
    else:
        val = []
        working_ion = all_data[0]['working_ion']
        ion_element = Element(working_ion)
        tmp_charge_data = all_data[0]
        not_ion_element = Composition(tmp_charge_data['formula_charge']).elements[-1]
        max_delta_volume = 0
        prev_ion_change_rate = 0
        all_rated_volatage = 0

        for data in all_data:
            charge_data = [tmp_charge_data[key] for key in ['formula_charge', 'id_charge', 'stability_charge']]
            discharge_data = [data[key] for key in ['formula_discharge', 'id_discharge', 'stability_discharge']]
            # calculate the voltage
            change_comp = Composition(tmp_charge_data['formula_charge'])
            discharge_comp = Composition(data['formula_discharge'])
            all_ion_change_rate = (discharge_comp.get_atomic_fraction(ion_element) \
                / discharge_comp.get_atomic_fraction(not_ion_element)) - \
                (change_comp.get_atomic_fraction(ion_element) \
                / change_comp.get_atomic_fraction(not_ion_element))
            all_rated_volatage += (all_ion_change_rate - prev_ion_change_rate) * data['average_voltage']
            average_voltage = all_rated_volatage / all_ion_change_rate
            prev_ion_change_rate = all_ion_change_rate
            # calculate the max volume change
            max_delta_volume = max([max_delta_volume, data['max_delta_volume']])
            # append
            val.append([battery_id] + charge_data + discharge_data + \
                [average_voltage, max_delta_volume, working_ion])
    return val


def get_discharge_structure_data(id_discharge, mp_dr, properties):
    data = [id_discharge]
    data.append(mp_dr.get_task_data(id_discharge)[0]['material_id'])
    for target in properties:
        value = mp_dr.get_task_data(id_discharge, target)
        data.append(str(CifWriter(value[0][target])))
    return data


def parse_arguments():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Prepare voltage base dataset')
    parser.add_argument('--out', '-o', type=str, required=True,
                        help='directory path to save the data')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse the arguments.
    args = parse_arguments()
    # create save path
    target_dir = args.out
    out_dir_path = path.normpath(path.join(getcwd(), target_dir))
    makedirs(out_dir_path, exist_ok=True)
    time = datetime.now().strftime("%Y_%m_%d")
    voltage_base_csv_path = path.join(out_dir_path, "voltage_base_" + time + ".csv")
    property_csv_path = path.join(out_dir_path, "property_" + time + ".csv")
    final_structure_path = path.join(out_dir_path, "final_structure_" + time + ".h5")
    initial_structure_path = path.join(out_dir_path, "initial_structure_" + time + ".h5")

    # get all battery ids
    MP_KEY = 'Gyp8UAHg6erw9ImW'
    mp_dr = MPRester(MP_KEY)
    # https://discuss.materialsproject.org/t/accessing-all-batteries-using-materials-project-api-or-all-possible-battids/399/2
    battery_ids = mp_dr._make_request('/battery/all_ids')

    # get all battery data
    key_for_data = [
        'formula_charge', 'id_charge', 'stability_charge',
        'formula_discharge', 'id_discharge', 'stability_discharge',
        'average_voltage', 'max_delta_volume', 'working_ion'
    ]
    all_data = Parallel(n_jobs=16, verbose=3)([
        delayed(get_battery_data_from_battery_id)(battery_id, mp_dr, key_for_data)
        for battery_id in battery_ids
    ])

    # flatten
    all_row = []
    for val in all_data:
        if isinstance(val[0], list) is False:
            all_row.append(val)
        else:
            for row in val:
                all_row.append(row)

    # convert list to dataframe
    columns = ['battery_id'] + key_for_data
    df = pd.DataFrame(all_row, columns=columns)
    print("data size (init): {}".format(len(df)))

    # check stability
    df = df[(df['stability_charge'] < 0.3) & (df['stability_discharge'] < 0.3)]
    print("data size (stablity): {}".format(len(df)))

    # check voltage range (0.0 < average_voltage < 6.0)
    df = df[(df['average_voltage'] > 0.0) & (df['average_voltage'] < 6.0)]
    print("data size (voltage): {}".format(len(df)))

    # remove multi ion reaction
    df = df.drop_duplicates(subset='id_discharge', keep=False)
    print("data size (multi ion): {}".format(len(df)))

    # save
    df = df.reset_index(drop=True)
    df.to_csv(voltage_base_csv_path)

    # get structure data
    properties = [
        'initial_structure',
        'final_structure',
    ]
    ids = list(set(list(df['id_discharge'].values) + list(df['id_charge'].values)))
    discharge_data = Parallel(n_jobs=16, verbose=3)([
        delayed(get_discharge_structure_data)(id, mp_dr, properties)
        for id in ids
    ])
    new_columns = ['material_id', 'new_material_id'] + properties
    df = pd.DataFrame(discharge_data, columns=new_columns)

    # save final structure data
    with h5py.File(initial_structure_path, 'w') as f:
        for mp_id, init_structure in zip(df['material_id'], df['initial_structure']):
            f.create_dataset(mp_id, data=init_structure)

    # save initial structure data
    with h5py.File(final_structure_path, 'w') as f:
        for mp_id, fin_structure in zip(df['material_id'], df['final_structure']):
            f.create_dataset(mp_id, data=fin_structure)

    # add dummy value
    for col in ['energy', 'energy_per_atom', 'formation_energy_per_atom', 'band_gap', 'efermi']:
        df[col] = [-1.5 if col == 'energy' else 1.0 for _ in range(len(df))]
    df = df.drop(['initial_structure', 'final_structure'], axis='columns')
    df.to_csv(property_csv_path, index=False)
