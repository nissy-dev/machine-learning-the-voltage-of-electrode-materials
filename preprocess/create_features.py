import h5py
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from os import path, makedirs, getcwd
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from utils import NUM_OF_FEATURE, ElementDataSingleton, create_feature_from_formula, \
    create_feature_from_wokring_ion, create_feature_from_crystal_system


def parse_arguments():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Create features')

    parser.add_argument('--element-data', '-ed', type=str, default='../dataset/element_data',
                        help='path of element data directory')
    parser.add_argument('--battery-data', '-bd', type=str, default='../dataset/mp_data/data_2019_12_03.csv',
                        help='path of battery csv data')
    parser.add_argument('--cif-data', type=str, default='../dataset/mp_data/data_2019_12_03.h5',
                        help='path to cif data (relative path)')
    parser.add_argument('--out', '-o', type=str, default='../dataset/feature',
                        help='directory path to save feature data')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse the arguments.
    args = parse_arguments()

    # create save path
    target_dir = args.out
    out_dir_path = path.normpath(path.join(getcwd(), target_dir))
    makedirs(out_dir_path, exist_ok=True)
    time = datetime.now().strftime("%Y_%m_%d")
    out_csv_path = path.join(out_dir_path, "repro_features_" + time + ".csv")
    # save the parameter
    with open(path.join(out_dir_path, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # load data
    cif_path = path.normpath(path.join(getcwd(), args.cif_data))
    cif_data = h5py.File(cif_path, "r")
    battery_table = pd.read_csv(args.battery_data, index_col=False)
    element_data = ElementDataSingleton.get_instance(element_data_path=args.element_data)

    # create features
    features = np.zeros((len(battery_table), NUM_OF_FEATURE))
    for i, (id_charge, id_discharge, ion) in tqdm(enumerate(zip(battery_table['id_charge'],
                                                                battery_table['id_discharge'],
                                                                battery_table['working_ion']))):

        # load crystal object from cif
        charge_crystal = Structure.from_str(cif_data[id_charge].value, 'cif')
        discharge_crystal = Structure.from_str(cif_data[id_discharge].value, 'cif')

        # specific features
        finder = SpacegroupAnalyzer(charge_crystal)
        space_group_feat = np.array([finder.get_space_group_number()])
        crystal_system_feat = create_feature_from_crystal_system(finder.get_crystal_system())
        ion_feat = create_feature_from_wokring_ion(ion)
        dischr_comp = Composition(discharge_crystal.formula)
        ion_concentrate = np.array([dischr_comp.get_atomic_fraction(Element(ion))])

        # calculate features from element property
        charge_formula = charge_crystal.formula
        discharge_formula = discharge_crystal.formula
        feature_from_chr_formula = create_feature_from_formula(charge_formula, element_data)
        feature_from_dischr_formula = create_feature_from_formula(discharge_formula, element_data)
        feature_from_ion = create_feature_from_formula(ion, element_data)

        # if charge_formula is a single element
        if feature_from_chr_formula.shape[0] == 44:
            feature_from_chr_formula = np.concatenate([feature_from_chr_formula, np.zeros(44)])

        # concate all features
        features[i] = np.concatenate([space_group_feat, crystal_system_feat, ion_feat, ion_concentrate,
                                      feature_from_chr_formula, feature_from_dischr_formula, feature_from_ion])

    df = pd.DataFrame(data=features,
                      columns=['feat_{}'.format(i+1) for i in range(239)])
    df.to_csv(out_csv_path)
