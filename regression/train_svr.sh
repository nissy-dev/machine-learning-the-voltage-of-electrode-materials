#!/bin/bash
#PBS -l select=1:ncpus=16

cd $PBS_O_WORKDIR
# activate venv
source ../venv/bin/activate

# work around
export HDF5_USE_FILE_LOCKING=FALSE

data_date=2020_07_15
feat_date=2020_07_16
date=$(date "+%Y_%m_%d")
seeds=(1234 42 169 32 1988)

for seed in ${seeds[@]}
do
    python train.py --feat-path ../dataset/feature/repro_new_features_${feat_date}.csv \
                    --battery-path ../dataset/mp_data/voltage_base_${data_date}.csv \
                    --method svr \
                    --seed ${seed} \
                    -o result/svr_${date}/${seed}
done

# deactivate venv
deactivate
