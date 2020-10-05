import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from os import path, makedirs, getcwd, environ
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


def seed_every_thing(seed=1234):
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce DNN regression')
    # for dataset path
    parser.add_argument('--feat-path', default='../dataset/feature/repro_features_YYYY_MM_DD.csv',
                        type=str, help='path to feature (relative path)')
    parser.add_argument('--battery-path', default='../dataset/mp_data/data_2019_12_03.csv',
                        type=str, help='path to csv data (relative path)')
    parser.add_argument('--out-dir', '-o', default='result',
                        type=str, help='path for output directory')
    # for model
    parser.add_argument('--train-ratio', default=0.9, type=float,
                        help='percentage of train data to be loaded (default 0.9)')
    parser.add_argument('--test-ratio', default=0.1, type=float,
                        help='percentage of test data to be loaded (default 0.1)')
    parser.add_argument('--fold', type=int, default=10,
                        help='fold value for cross validation, (default: 10)')
    # target ion : Li, Ca, Cs, Rb, K, Y, Na, Al, Zn, Mg
    parser.add_argument('--target-ion', type=str, default='Li_Ca_Y_Al_Zn_Mg',
                        help='drop a specific ion data, (default: Li_Ca_Y_Al_Zn_Mg)')
    parser.add_argument('--test-ion', type=str, default='Na',
                        help='test data includes one ion, (default: Na)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed value for random value, (default: 1234)')
    return parser.parse_args()


def make_dnn_model():
    model = Sequential([
        Dense(60, input_dim=80, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.25),
        Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.10),
        Dense(1),
    ])
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model


def main():
    # get args
    args = parse_arguments()

    # set seed
    seed_every_thing(args.seed)

    # make output directory
    out_dir = args.out_dir
    out_dir_path = path.normpath(path.join(getcwd(), out_dir))
    makedirs(out_dir_path, exist_ok=True)
    # save arguments
    with open(path.join(out_dir_path, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # load raw data and feature
    feat_path = path.normpath(path.join(getcwd(), args.feat_path))
    battery_path = path.normpath(path.join(getcwd(), args.battery_path))
    feat_data = pd.read_csv(feat_path, index_col=0)
    battery_data = pd.read_csv(battery_path, index_col=0)
    past_index = battery_data.index
    battery_data = battery_data.reset_index()
    table_data = battery_data.join(feat_data)
    table_data.index = past_index

    # drop duplicated rows
    feat_columns = ['feat_{}'.format(i+1) for i in range(239)]
    table_data = table_data[~table_data.duplicated(subset=feat_columns)]

    # collect target ion data
    if args.target_ion is not None:
        ion_list = args.target_ion.split('_')
        train_data = table_data[table_data['working_ion'].isin(ion_list)]

    # target data
    target = train_data['average_voltage'].values
    # feature data
    feat_columns = ['feat_{}'.format(i+1) for i in range(239)]
    features = train_data[feat_columns]
    # index
    index = train_data.index

    # dimension reduction by PCA
    pca_scaler = StandardScaler()
    pca = PCA(n_components=80)
    features = pca_scaler.fit_transform(features)
    features = pca.fit_transform(features)

    # split and scaling
    X_train, X_test, y_train, y_test, train_idx, test_idx = \
        train_test_split(features, target, index, test_size=args.test_ratio)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # create KFold cross validation instance
    kf = KFold(n_splits=args.fold)
    idx_list = kf.split(X_train)

    # cross validation
    cv_score = pd.DataFrame(
        index=['cv_{}'.format(i+1) for i in range(args.fold)],
        columns=['R2_train', 'MAE_train', 'R2_valid', 'MAE_valid']
    )
    for i, (train_index, valid_index) in enumerate(idx_list):
        X_kf_train, X_kf_valid = X_train_scaled[train_index], X_train_scaled[valid_index]
        y_kf_train, y_kf_valid = y_train[train_index], y_train[valid_index]

        # initialize model
        regressor = make_dnn_model()
        best_weights_filepath = path.join(out_dir_path, 'best_weights_fold_{}.hdf5'.format(i+1))
        callbacks = [
            EarlyStopping(monitor='val_mae', min_delta=0, patience=15, verbose=0, mode='auto'),
            ModelCheckpoint(best_weights_filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='auto')
        ]

        # fit and predict
        # FIXME: The paper didn't mention about epoch and batch size
        regressor.fit(X_kf_train, y_kf_train, epochs=150, batch_size=32,
                      validation_data=(X_kf_valid, y_kf_valid), callbacks=callbacks)
        regressor.load_weights(best_weights_filepath)
        pred_y_kf_train = regressor.predict(X_kf_train)
        pred_y_kf_valid = regressor.predict(X_kf_valid)

        # save score
        cv_name = 'cv_{}'.format(i+1)
        cv_score.loc[cv_name, 'R2_train'] = r2_score(y_kf_train, pred_y_kf_train)
        cv_score.loc[cv_name, 'MAE_train'] = mean_absolute_error(y_kf_train, pred_y_kf_train)
        cv_score.loc[cv_name, 'R2_valid'] = r2_score(y_kf_valid, pred_y_kf_valid)
        cv_score.loc[cv_name, 'MAE_valid'] = mean_absolute_error(y_kf_valid, pred_y_kf_valid)

        # verbose
        print('CV: {0}/{1} R2_train: {2} MAE_train: {3}\t'
              'R2_valid: {4} MAE_valid: {5}'.format(
                  i+1, args.fold, cv_score.loc[cv_name, 'R2_train'], cv_score.loc[cv_name, 'MAE_train'],
                  cv_score.loc[cv_name, 'R2_valid'], cv_score.loc[cv_name, 'MAE_valid']))

    # dump csv
    cv_score.to_csv(path.join(out_dir_path, 'cv_score.csv'))

    # H-test
    # retrain all data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_scaled, y_train, test_size=args.test_ratio)

    # initialize model
    regressor = make_dnn_model()
    best_weights_filepath = path.join(out_dir_path, 'best_weights_h_test.hdf5'.format(i+1))
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
        ModelCheckpoint(best_weights_filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='auto')
    ]

    # fit and predict
    # FIXME: The paper didn't mention about epoch and batch size
    regressor.fit(X_train, y_train, epochs=150, batch_size=32,
                  validation_data=(X_valid, y_valid), callbacks=callbacks)
    regressor.load_weights(best_weights_filepath)
    pred_y_valid = regressor.predict(X_valid)
    pred_y_test = regressor.predict(X_test_scaled)

    # save score
    test_pred_val = pd.DataFrame({'test_ground_truth': y_test, 'test_pred': pred_y_test.reshape(-1),
                                  'raw_index': test_idx})
    test_score = pd.DataFrame(index=['test'], columns=['R2_valid', 'MAE_valid', 'R2_test', 'MAE_test'])
    test_score.loc['test', 'R2_valid'] = r2_score(y_valid, pred_y_valid)
    test_score.loc['test', 'MAE_valid'] = mean_absolute_error(y_valid, pred_y_valid)
    test_score.loc['test', 'R2_test'] = r2_score(y_test, pred_y_test)
    test_score.loc['test', 'MAE_test'] = mean_absolute_error(y_test, pred_y_test)
    # dump csv
    test_pred_val.to_csv(path.join(out_dir_path, 'test_pred_value.csv'))
    test_score.to_csv(path.join(out_dir_path, 'test_score.csv'))

    # for Na-test (optional)
    if args.test_ion is not None:
        # preprocess
        test_data = table_data[table_data['working_ion'].isin([args.test_ion])]
        y = test_data['average_voltage'].values
        test_features = test_data[feat_columns]
        test_features = pca_scaler.transform(test_features)
        test_features = pca.transform(test_features)
        X_test_scaled = x_scaler.transform(test_features)

        # predict
        pred_y = regressor.predict(X_test_scaled)

        # save score
        test_pred_val = pd.DataFrame({'test_ground_truth': y, 'test_pred': pred_y.reshape(-1),
                                      'raw_index': test_data.index.values})
        test_score = pd.DataFrame(index=['test'], columns=['R2_test', 'MAE_test'])
        test_score.loc['test', 'R2_test'] = r2_score(y, pred_y)
        test_score.loc['test', 'MAE_test'] = mean_absolute_error(y, pred_y)
        # dump csv
        test_pred_val.to_csv(path.join(out_dir_path, 'test_{}_pred_value.csv'.format(args.test_ion)))
        test_score.to_csv(path.join(out_dir_path, 'test_{}_score.csv'.format(args.test_ion)))


if __name__ == '__main__':
    main()
