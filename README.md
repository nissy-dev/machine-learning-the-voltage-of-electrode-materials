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

## result

## References

[1] Rajendra P. JoshiJesse, et al. Machine Learning the Voltage of Electrode Materials in Metal-Ion Batteries. _ACS Appl. Mater. Interfaces_, 18494-18503, 2019.
