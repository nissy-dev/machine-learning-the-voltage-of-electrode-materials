# Scripts for training models

## Train DNN

```
$ python train_dnn.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv \
                      --battery-path ../dataset/mp_data/voltage_final_structure_YYYY_MM_DD.h5 \
                      --test-ion Na \
                      --out-dir result/dnn_YYYY_MM_DD
```
