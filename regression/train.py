import json
import argparse
import numpy as np
import pandas as pd
from os import path, makedirs, getcwd
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce SVR/KRR regression')
    # for dataset path
    parser.add_argument('--feat-path', default='../dataset/feature/repro_features_YYYY_MM_DD.csv',
                        type=str, help='path to feature (relative path)')
    parser.add_argument('--battery-path', default='../dataset/mp_data/data_2019_12_03.csv',
                        type=str, help='path to csv data (relative path)')
    parser.add_argument('--out-dir', '-o', default='result',
                        type=str, help='path for output directory')
    # for model
    parser.add_argument('--method', choices=['svr', 'krr'], default='krr',
                        help='choose a regression method, SVR, KRR (default: krr)')
    parser.add_argument('--train-ratio', default=0.9, type=float,
                        help='percentage of train data to be loaded (default 0.9)')
    parser.add_argument('--test-ratio', default=0.1, type=float,
                        help='percentage of test data to be loaded (default 0.1)')
    parser.add_argument('--fold', type=int, default=10,
                        help='fold value for cross validation, (default: 10)')
    # target ion : Li, Ca, Cs, Rb, K, Y, Na, Al, Zn, Mg
    parser.add_argument('--target-ion', type=str, default='Li_Ca_K_Y_Al_Zn_Mg',
                        help='drop a specific ion data, (default: Li_Ca_K_Y_Al_Zn_Mg)')
    parser.add_argument('--test-ion', type=str, default='Na',
                        help='test data includes one ion, (default: Na)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed value for random value, (default: 1234)')
    # sampling
    parser.add_argument('--sampling', action='store_true',
                        help='sampling 3977 data for comparing with the previous study, (default: false)')
    return parser.parse_args()


def main():
    # get args
    args = parse_arguments()

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

    # collect target ion data
    if args.target_ion is not None:
        ion_list = args.target_ion.split('_')
        train_data = table_data[table_data['working_ion'].isin(ion_list)]

    # sampling
    if args.sampling:
        train_data = train_data.sample(n=3977, random_state=args.seed)

    # target data
    target = train_data['average_voltage'].values
    # feature data
    feat_columns = ['feat_{}'.format(i+1) for i in range(239)]
    features = train_data[feat_columns]
    # index
    index = train_data.index

    # dimension reduction by PCA
    pca = PCA(n_components=80)
    features = pca.fit_transform(features)

    # split and scaling
    X_train, X_test, y_train, y_test, train_idx, test_idx = \
        train_test_split(features, target, index,
                         test_size=args.test_ratio, random_state=args.seed)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # create search param and model
    if args.method == 'svr':
        params = {
            "C": np.logspace(-5, 5, 11, base=10),
            "gamma": np.logspace(-5, 5, 11, base=10)
        }
        model = SVR()
    else:
        params = {
            "alpha": np.logspace(-5, 5, 11, base=10),
            "gamma": np.logspace(-5, 5, 11, base=10),
        }
        model = KernelRidge(kernel='rbf')

    # grid search
    kf = KFold(n_splits=args.fold, random_state=args.seed)
    my_mae = make_scorer(mean_absolute_error, greater_is_better=False)
    clf = GridSearchCV(model, params, scoring=my_mae,
                       cv=kf, n_jobs=-1, verbose=True)
    clf.fit(X_train_scaled, y_train)
    # save score and dump
    cv_score = pd.DataFrame.from_dict(clf.cv_results_)
    cv_score.to_csv(path.join(out_dir_path, 'cv_score.csv'))

    # H-test
    regressor = clf.best_estimator_
    # refit using all data
    regressor.fit(X_train_scaled, y_train)
    pred_y_train = regressor.predict(X_train_scaled)
    pred_y_test = regressor.predict(X_test_scaled)

    # save score
    test_pred_val = pd.DataFrame({'test_ground_truth': y_test,
                                  'train_pred': pred_y_test.reshape(-1),
                                  'raw_index': test_idx})
    test_score = pd.DataFrame(index=['test'], columns=['R2_train', 'MAE_train', 'R2_test', 'MAE_test'])
    test_score.loc['test', 'R2_train'] = r2_score(y_train, pred_y_train)
    test_score.loc['test', 'MAE_train'] = mean_absolute_error(
        y_train, pred_y_train)
    test_score.loc['test', 'R2_test'] = r2_score(y_test, pred_y_test)
    test_score.loc['test', 'MAE_test'] = mean_absolute_error(
        y_test, pred_y_test)
    # dump csv
    test_pred_val.to_csv(
        path.join(out_dir_path, 'test_model_test_pred_value.csv'))
    test_score.to_csv(path.join(out_dir_path, 'test_score.csv'))

    # for Na-test (optional)
    if args.test_ion is not None:
        # preprocess
        test_data = table_data[table_data['working_ion'].isin([args.test_ion])]
        y = test_data['average_voltage'].values
        feat_columns = ['feat_{}'.format(i+1) for i in range(239)]
        test_features = test_data[feat_columns]
        test_features = pca.transform(test_features)
        X_test_scaled = x_scaler.transform(test_features)
        # predict
        pred_y_test = regressor.predict(X_test_scaled)
        # save score
        test_pred_val = pd.DataFrame({'test_ground_truth': y, 'test_pred': pred_y_test.reshape(-1),
                                      'raw_index': test_data.index.values})
        test_score = pd.DataFrame(index=['test'], columns=[
                                  'R2_test', 'MAE_test'])
        test_score.loc['test', 'R2_test'] = r2_score(y, pred_y_test)
        test_score.loc['test', 'MAE_test'] = mean_absolute_error(y, pred_y_test)
        # dump csv
        test_pred_val.to_csv(path.join(out_dir_path, 'test_{}_pred_value.csv'.format(args.test_ion)))
        test_score.to_csv(path.join(out_dir_path, 'test_{}_score.csv'.format(args.test_ion)))


if __name__ == '__main__':
    main()
