# Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries

This repository is the reproduction for "Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries".

## Repository

```
├── dataset
├── preprocess              # feature engineering scripts
├── regression              # train scripts
└── README.md
```

## How to reproduce

## 1. Clone this repository and setup the environment

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

`YYYY_MM_DD` is the date you created features on. Please set your date.

```
$ cd regression
$ python train.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv --method krr
$ python train_dnn.py --feat-path ../dataset/feature/repro_features_YYYY_MM_DD.csv
```

## Result

|                        |    SVR    |    KRR    | DNN |   \*SVR   |   \*KRR    |   \*DNN   |
| :--------------------- | :-------: | :-------: | :-: | :-------: | :--------: | :-------: |
| fold 1                 |   0.44    |   0.46    |     |   0.51    |    0.54    |   0.42    |
| fold 2                 |   0.42    |   0.44    |     |   0.25    |    0.28    |   0.48    |
| fold 3                 |   0.42    |   0.43    |     |   0.26    |    0.27    |   0.42    |
| fold 4                 |   0.43    |   0.44    |     |   0.35    |    0.47    |   0.44    |
| fold 5                 |   0.40    |   0.42    |     |   0.38    |    0.43    |   0.44    |
| fold 6                 |   0.44    |   0.43    |     |   0.62    |    0.71    |   0.42    |
| fold 7                 |   0.49    |   0.51    |     |   0.43    |    0.42    |   0.43    |
| fold 8                 |   0.39    |   0.41    |     |   0.59    |    0.62    |   0.42    |
| fold 9                 |   0.40    |   0.40    |     |   0.53    |    0.57    |   0.45    |
| fold 10                |   0.40    |   0.40    |     |   0.28    |    0.30    |   0.48    |
| 10 fold MAE (mean±std) | 0.42±0.03 | 0.43±0.03 |     | 0.42±0.13 | 00.46±0.14 | 0.43±0.03 |
| H-test                 |   0.40    |   0.41    |     |   0.40    |    0.39    |   0.43    |
| Na-test                |   0.56    |   0.55    |     |   1.00    |    0.93    |   1.25    |

\*SVR, \*KRR, \*DNN are reported in the paper.

## References

[1] Rajendra P. JoshiJesse, et al. Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries. _ACS Appl. Mater. Interfaces_, 18494-18503, 2019.
