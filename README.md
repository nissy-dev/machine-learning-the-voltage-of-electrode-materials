# Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries

This repository is the reproduction for "Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries".

## Repository

```
├── dataset
├── preprocess              # feature engineering scripts
├── regression              # train scripts
├── .flake8
├── .gitignore
├── README.md
└── requirements.txt
```

## How to reproduce

Python version : 3.6.5

### 1. Clone this repository and setup the environment

```
$ git clone https://github.com/nd-02110114/machine-learning-the-voltage-of-electrode-materials.git
$ cd machine-learning-the-voltage-of-electrode-materials
$ python -v venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### 2. Go to the preprocess folder and create features

```
$ cd preprocess
$ python create_features.py
```

### 3. Go to the regression folder and train models

`YYYY_MM_DD` is the date you created features. Please set your date.

```
$ cd regression
$ python train.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv --method svr --out-dir result/svr
$ python train.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv --method krr --out-dir result/krr
$ python train_dnn.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv --out-dir result/dnn
```

## Result

|                        |    SVR    |    KRR    |    DNN    |   \*SVR   |   \*KRR   |   \*DNN   |
| :--------------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| fold 1                 |   0.40    |   0.42    |   0.47    |   0.51    |   0.54    |   0.42    |
| fold 2                 |   0.42    |   0.42    |   0.46    |   0.25    |   0.28    |   0.48    |
| fold 3                 |   0.41    |   0.43    |   0.47    |   0.26    |   0.27    |   0.42    |
| fold 4                 |   0.40    |   0.42    |   0.43    |   0.35    |   0.47    |   0.44    |
| fold 5                 |   0.43    |   0.45    |   0.50    |   0.38    |   0.43    |   0.44    |
| fold 6                 |   0.43    |   0.44    |   0.47    |   0.62    |   0.71    |   0.42    |
| fold 7                 |   0.46    |   0.45    |   0.53    |   0.43    |   0.42    |   0.43    |
| fold 8                 |   0.39    |   0.41    |   0.47    |   0.59    |   0.62    |   0.42    |
| fold 9                 |   0.44    |   0.44    |   0.47    |   0.53    |   0.57    |   0.45    |
| fold 10                |   0.43    |   0.43    |   0.44    |   0.28    |   0.30    |   0.48    |
| 10 fold MAE (mean±std) | 0.42±0.02 | 0.43±0.01 | 0.47±0.03 | 0.42±0.13 | 0.46±0.14 | 0.43±0.03 |
| H-test                 |   0.41    |   0.43    |   0.49    |   0.40    |   0.39    |   0.43    |
| Na-test                |   (0.56)    |   (0.56)    |   (0.58)    |   1.00    |   0.93    |   1.25    |

\*SVR, \*KRR, \*DNN are reported in the paper.

## References

[1] Rajendra P. JoshiJesse, et al. Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries. _ACS Appl. Mater. Interfaces_, 18494-18503, 2019.  
[2] Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, et al. Commentary: The materials project: A materials genome approach to accelerating materials innovation. _Apl Materials_, 1(1):011002, 2013
