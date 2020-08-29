#!/bin/bash
#PBS -l select=1:ncpus=16

cd $PBS_O_WORKDIR
# activate venv
source ../venv/bin/activate

# work around
export HDF5_USE_FILE_LOCKING=FALSE

data_date=2020_08_05
feat_date=2020_08_29
date=$(date "+%Y_%m_%d")
seeds=(1234 42 169 32 1988)

for seed in ${seeds[@]}
do
    python train.py --feat-path ../dataset/feature/repro_features_${feat_date}.csv \
                    --battery-path ../dataset/mp_data/voltage_${data_date}.csv \
                    --method krr \
                    --seed ${seed} \
                    -o result/krr_${date}/${seed}
done


# deactivate venv
deactivate
