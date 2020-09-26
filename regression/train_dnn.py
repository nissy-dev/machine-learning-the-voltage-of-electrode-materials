import json
import random
import argparse
from os import path, makedirs, getcwd, environ


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--feat-path', required=True,
                        type=str, help='path to feature (relative path)')
    parser.add_argument('--battery-path', required=True,
                        type=str, help='path to csv data (relative path)')
    parser.add_argument('--out-dir', '-o', default='result',
                        type=str, help='path for output directory')
    # for model
    parser.add_argument('--train-ratio', default=0.6, type=float,
                        help='percentage of train data to be loaded (default 0.6)')
    parser.add_argument('--valid-ratio', default=0.2, type=float,
                        help='percentage of valid data to be loaded (default 0.2)')
    parser.add_argument('--test-ratio', default=0.2, type=float,
                        help='percentage of test data to be loaded (default 0.2)')
    parser.add_argument('--fold', type=int, default=10,
                        help='fold value for cross validation, (default: 10)')
    # target ion : Li, Ca, Cs, Rb, K, Y, Na, Al, Zn, Mg
    parser.add_argument('--target-ion', type=str, default='Li_Ca_K_Al_Zn_Mg_Y_Rb_Cs',
                        help='drop a specific ion data, (default: Li_Ca_K_Al_Zn_Mg_Y_Rb_Cs)')
    parser.add_argument('--test-ion', type=str, default='Na',
                        help='test data includes one ion, (default: Na)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed value for random value, (default: 1234)')
    return parser.parse_args()


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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


def train_valid_test_split(table, train_ratio, val_ratio, test_ratio, stratify):
    # validation
    assert train_ratio + val_ratio + test_ratio == 1
    x_remaining, x_test = train_test_split(table, test_size=test_ratio, stratify=table[stratify])

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - test_ratio
    ratio_val_adjusted = val_ratio / ratio_remaining

    # Produces train and val splits.
    x_train, x_val = train_test_split(x_remaining, test_size=ratio_val_adjusted, stratify=x_remaining[stratify])
    return x_train, x_val, x_test



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
    feat_data.index = battery_data.index
    table_data = battery_data.join(feat_data)

    # collect target ion data
    if args.target_ion is not None:
        ion_list = args.target_ion.split('_')
        train_data = table_data[table_data['working_ion'].isin(ion_list)]

    # split
    train_table, valid_table, test_table = \
        train_valid_test_split(train_data, train_ratio=args.train_ratio, val_ratio=args.valid_ratio,
                               test_ratio=args.test_ratio, stratify='working_ion')
    test_idx = test_table.index

    # PCA and MinMaxScaler for train data
    feat_columns = ['feat_{}'.format(i+1) for i in range(239)]
    X_train = train_table[feat_columns]
    pca = PCA(n_components=80)
    X_train = pca.fit_transform(X_train)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(X_train)
    y_train = train_table['average_voltage'].values

    # transoform valid and test data
    X_valid = x_scaler.transform(pca.transform(valid_table[feat_columns]))
    y_valid = valid_table['average_voltage'].values
    X_test = x_scaler.transform(pca.transform(test_table[feat_columns]))
    y_test = test_table['average_voltage'].values

    # initialize model
    regressor = make_dnn_model()
    best_model_path = path.join(out_dir_path, 'best_model.hdf5')
    callbacks = [
        EarlyStopping(monitor='val_mae', min_delta=0, patience=15, verbose=0, mode='auto'),
        ModelCheckpoint(best_model_path, monitor='val_mae', verbose=1, save_best_only=True, mode='auto')
    ]

    # fit and predict
    regressor.fit(X_train, y_train, epochs=150, batch_size=32,
                  validation_data=(X_valid, y_valid), callbacks=callbacks)
    regressor.load_weights(best_model_path)
    pred_y_train = regressor.predict(X_train)
    pred_y_valid = regressor.predict(X_valid)

    # calc score
    score = pd.DataFrame(index=['hold_out'], columns=['MAE'])
    score.loc['hold_out', 'R2_valid'] = r2_score(y_valid, pred_y_valid)
    score.loc['hold_out', 'MAE_valid'] = mean_absolute_error(y_valid, pred_y_valid)

    # save cv_score to csv
    score.to_csv(path.join(out_dir_path, 'score.csv'))

    # H-test
    regressor.load_weights(best_model_path)
    pred_y_test = regressor.predict(X_test)

    # save score
    test_pred_val = pd.DataFrame({'test_ground_truth': y_test,
                                  'test_pred': pred_y_test.reshape(-1),
                                  'raw_index': test_idx})
    test_score = pd.DataFrame(index=['test'], columns=['R2_test', 'MAE_test', 'RMSE_test'])
    test_score.loc['test', 'R2_test'] = r2_score(y_test, pred_y_test)
    test_score.loc['test', 'MAE_test'] = mean_absolute_error(y_test, pred_y_test)
    test_score.loc['test', 'RMSE_test'] = root_mean_squared_error(y_test, pred_y_test)
    # dump csv
    test_pred_val.to_csv(path.join(out_dir_path, 'test_pred_value.csv'))
    test_score.to_csv(path.join(out_dir_path, 'test_score.csv'))

    # for Na-test (optional)
    if args.test_ion is not None:
        # preprocess
        test_data = table_data[table_data['working_ion'].isin([args.test_ion])]
        y = test_data['average_voltage'].values
        X = x_scaler.transform(pca.transform(test_data[feat_columns]))

        # predict
        pred_y = regressor.predict(X)

        # save score
        test_pred_val = pd.DataFrame({'test_ground_truth': y, 'test_pred': pred_y.reshape(-1),
                                      'raw_index': test_data.index.values})
        test_score = pd.DataFrame(index=['test'], columns=['R2_test', 'MAE_test', 'RMSE_test'])
        test_score.loc['test', 'R2_test'] = r2_score(y, pred_y)
        test_score.loc['test', 'MAE_test'] = mean_absolute_error(y, pred_y)
        test_score.loc['test', 'RMSE_test'] = root_mean_squared_error(y, pred_y)
        # dump csv
        test_pred_val.to_csv(path.join(out_dir_path, 'test_{}_pred_value.csv'.format(args.test_ion)))
        test_score.to_csv(path.join(out_dir_path, 'test_{}_score.csv'.format(args.test_ion)))


if __name__ == '__main__':
    main()
