#!/bin/bash
#PBS -l select=1:ncpus=8

cd $PBS_O_WORKDIR
# activate venv
source ../venv/bin/activate

# work around
export HDF5_USE_FILE_LOCKING=FALSE

data_date=2020_07_15
feat_date=2020_07_15
date=$(date "+%Y_%m_%d")

python train.py --feat-path ../dataset/feature/repro_features_${feat_date}.csv --battery-path ../dataset/mp_data/voltage_base_${data_date}.csv --method krr -o result/krr_${date} && \
python train.py --feat-path ../dataset/feature/repro_features_${feat_date}.csv --battery-path ../dataset/mp_data/voltage_base_${data_date}.csv --method svr -o result/svr_${date} && \
python train_dnn.py --feat-path ../dataset/feature/repro_features_${feat_date}.csv --battery-path ../dataset/mp_data/voltage_base_${data_date}.csv -o result/dnn_${date}


# deactivate venv
deactivate
