import numpy as np
from os import path, linesep
from pymatgen.core import Composition, Element


NUM_OF_FEATURE = 239

COLUMNS = [
    'AtimicNumber',
    'AtomicVolume',
    'AtomicWeight',
    'BCCefflatcnt',
    'BCCenergydiff',
    'BCCfermi',
    'BCCmagmom',
    'BCCvolume_pa',
    'BCCvolume_padiff',
    'BoilingT',
    'ColumnNumber',
    'CovalentRadius',
    'Density',
    'FirstIonizationEnergy',
    'GSbandgap',
    'GSefflatcnt',
    'GSestBCClatcnt',
    'GSestFCClatcnt',
    'GSmagmom',
    'GSvolume_pa',
    'HeatCapacityMass',
    'HeatCapacityMolar',
    'ICSDVolume',
    'IsAlkali',
    'IsDBlock',
    'IsFBlock',
    'IsMetal',
    'IsMetalloid',
    'IsNonmetal',
    'MendeleevNumber',
    'NUnfilled',
    'NValance',
    'NdUnfilled',
    'NdValence',
    'NfUnfilled',
    'NfValence',
    'NpUnfilled',
    'NpValence',
    'NsUnfilled',
    'NsValence',
    'OxidationStates',
    'Polarizability',
    'RowNumber',
    'SpaceGroupNumber',
]


class ElementDataSingleton:

    _element_data_instance = None

    @classmethod
    def get_instance(self, element_data_path, columns=COLUMNS):
        if not self._element_data_instance:
            self._element_data_instance = {}
            # store element_data
            for col in columns:
                element_data = None
                with open(path.join(element_data_path, col + '.table'), 'r') as f:
                    if col != 'OxidationStates':
                        element_data = [0 if line.rstrip(linesep) == 'Missing' or line.rstrip(linesep) == ''
                                        else float(line) for line in f.readlines()]

                self._element_data_instance[col] = element_data

        return self._element_data_instance


def create_feature_from_formula(formula, element_data, columns=COLUMNS):
    comp = Composition(formula)
    num_of_base_feat = 44

    if len(comp) == 1:
        features = np.zeros(num_of_base_feat)
        ele = comp.elements[0]
        for i, col in enumerate(columns):
            if col != 'OxidationStates':
                features[i] = element_data[col][ele.number - 1]
            else:
                # FIXME: I don't under stand how to treat `OxidationStates` data 
                features[i] = 0
    else:
        weighted_mean_feat = np.zeros(num_of_base_feat)
        mean_deviation_feat = np.zeros(num_of_base_feat)
        element_dict = comp.to_reduced_dict

        tot_ratio = sum(element_dict.values())
        for i, col in enumerate(columns):
            if col != 'OxidationStates':
                tmp_mean_feat = 0.0
                for element_str, comp_ratio in element_dict.items():
                    ele = Element(element_str)
                    tmp_mean_feat += element_data[col][ele.number - 1] * comp_ratio

                mean_feat = tmp_mean_feat / tot_ratio
                weighted_mean_feat[i] = mean_feat

                tmp_deviation_feat = 0.0
                for element_str, comp_ratio in element_dict.items():
                    ele = Element(element_str)
                    tmp_deviation_feat += abs(element_data[col][ele.number - 1] - mean_feat) * comp_ratio

                deviation_feat = tmp_deviation_feat / tot_ratio
                mean_deviation_feat[i] = deviation_feat
            else:
                # FIXME: I don't under stand how to treat `OxidationStates` data 
                mean_feat = 0.0
                deviation_feat = 0.0
                weighted_mean_feat[i] = mean_feat
                mean_deviation_feat[i] = deviation_feat

        features = np.concatenate([weighted_mean_feat, mean_deviation_feat])

    return features


def create_feature_from_wokring_ion(ion):
    feature = np.zeros(10)
    if ion == 'Li':
        feature[0] = 1.0
    elif ion == 'Ca':
        feature[1] = 1.0
    elif ion == 'Cs':
        feature[2] = 1.0
    elif ion == 'Rb':
        feature[3] = 1.0
    elif ion == 'K':
        feature[4] = 1.0
    elif ion == 'Y':
        feature[5] = 1.0
    elif ion == 'Na':
        feature[6] = 1.0
    elif ion == 'Al':
        feature[7] = 1.0
    elif ion == 'Zn':
        feature[8] = 1.0
    elif ion == 'Mg':
        feature[9] = 1.0

    return feature


def create_feature_from_crystal_system(crystal_system):
    feature = np.zeros(7)
    if crystal_system == 'triclinic':
        feature[0] = 1.0
    elif crystal_system == 'monoclinic':
        feature[1] = 1.0
    elif crystal_system == 'orthorhombic':
        feature[2] = 1.0
    elif crystal_system == 'tetragonal':
        feature[3] = 1.0
    elif crystal_system == 'trigonal':
        feature[4] = 1.0
    elif crystal_system == 'hexagonal':
        feature[5] = 1.0
    elif crystal_system == 'cubic':
        feature[6] = 1.0

    return feature
