
import glob
from os import path
import numpy as np
import pandas as pd


if __name__ == '__main__':
    files = glob.glob('result/**/')
    print(files)
    seeds = [32, 42, 169, 1234, 1988]
    index = []
    base_index = [f'H_test_{val}' for val in ['R2', 'MAE', 'RMSE']] + \
        [f'Na_test_{val}' for val in ['R2', 'MAE', 'RMSE']]

    for filepath in files:
        index.extend([filepath] + base_index)

    df = pd.DataFrame(index=index)
    for seed in seeds:
        value = []
        for i, filepath in enumerate(files):
            value += [0]
            dir_name = filepath + str(seed)
            H_test = pd.read_csv(path.join(dir_name, 'test_score.csv'), index_col=0)
            H_test = H_test.T
            value += H_test['test'].to_list()
            Na_test = pd.read_csv(path.join(dir_name, 'test_Na_score.csv'), index_col=0)
            Na_test = Na_test.T
            value += Na_test['test'].to_list()
        df[str(seed)] = value

    df['mean'] = np.mean(df.values, axis=1)
    df['std'] = np.std(df.values, axis=1)
    df.to_csv('aggregated.csv')
