# Scripts for training models

## Train SVR and KRR

```
$ python train.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv \
                  --battery-path ../dataset/mp_data/data_2019_12_03.csv \
                  --method krr(svr) \
                  --target-ion Li_Ca_Y_Al_Zn_Mg \
                  --test-ion Na \
                  --out-dir result/krr
```

## Train DNN

```
$ python train_dnn.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv \
                      --battery-path ../dataset/mp_data/data_2019_12_03.csv \
                      --target-ion Li_Ca_Y_Al_Zn_Mg \
                      --test-ion Na \
                      --out-dir result/dnn
```
