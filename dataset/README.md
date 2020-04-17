# Dataset

## element_data

Based on Supporting Information, these elemental properties are collected by [Wolfram](https://reference.wolfram.com/language/note/ElementDataSourceInformation.html).

I downloaded files from [this page](https://bitbucket.org/wolverton/magpie/src/master/lookup-data/).

## mp_data

I collected the battery data from Materials Project on 2019/12/03.  
These data are collected by Battery API (You could confirm the API usage [here](https://discuss.matsci.org/t/accessing-battery-database-through-pymatgen/295/2).)

### data_2019_12_03.csv

This csv contains the following properties. Total size is 5134.

- battery_id
- formula_charge
- id_charge
- stability_charge
- formula_discharge
- id_discharge
- stability_discharge
- average_voltage
- capacity_grav
- capacity_vol
- energy_grav
- energy_vol
- max_delta_volume
- working_ion

### data_2019_12_03.h5

This HDF file contains structure data (cif format).

key : id_charge or id_discharge (material_id)  
value : structure data (string, cif format)

You could confirm the usage from [these lines](https://github.com/nd-02110114/machine-learning-the-voltage-of-electrode-materials/blob/9d2c3ebd2010af674237a12bc877f35b6e45d75a/preprocess/create_features.py#L59-L60)
