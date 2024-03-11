import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np


class TabularDataLoader:
    """
    Class that holds the DataFrame where the tabular data is located.
    """

    def __init__(self, file_path, pred_vars, trte_ratio):
        # Load file and convert into float32 since model parameters initialized w/ Pytorch are in float32
        dataframe = pd.read_csv(file_path, sep=',', index_col=0)
        self.dataframe = dataframe.astype('float32')

        X = self.dataframe.drop(columns=pred_vars)
        y = self.prepare_labels()
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_val, self.Y_val = self.train_test_val_split(X, y,
                                                                                                                 1 - trte_ratio)
        #self._normalize_data()
        self._create_batches(64)

    def describe_dataframe(self):
        return self.dataframe.describe()

    def fetch_columns(self):
        return self.dataframe.columns

    def input_dim_train(self):
        return len(self.X_train.iloc[0])

    def input_dim_test(self):
        return len(self.Y_train.iloc[0])

    def train_test_val_split(self, X, y, tr_ratio):
        te_val_ratio = (1 - tr_ratio) / 2
        size = len(X)
        tr_size = int(size * tr_ratio)
        te_val_size = int(size * te_val_ratio)
        return X[:tr_size], y[:tr_size], \
               X[tr_size:tr_size + te_val_size], y[tr_size:tr_size + te_val_size], \
               X[tr_size + te_val_size:], y[tr_size + te_val_size:]

    def prepare_labels(self):
        '''
        Labels are supposed to be in the form of (censorship, time of event)
        '''
        pfs = self.dataframe['PFS']
        cnsr = self.dataframe['CENSOR']
        result = []
        for p, c in zip(pfs, cnsr):
            b = False
            if c == 0:
                b = True
            result += [(b, p)]
        return result

    def _normalize_data(self):
        '''
        Normalizes train and test data
        '''
        scaler = MinMaxScaler(feature_range=(0,1))

        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns,
                                    index=self.X_train.index)

        self.X_test = pd.DataFrame(scaler.fit_transform(self.X_test), columns=self.X_test.columns,
                                   index=self.X_test.index)

        self.X_val = pd.DataFrame(scaler.fit_transform(self.X_val), columns=self.X_val.columns,
                                   index=self.X_val.index)

    def _create_batches(self, size):
        '''
        Creates batches for training out of the original data
        '''
        self.X_train_batch, self.Y_train_batch, self.idxs_train_batch = create_batches(self.X_train.values,
                                                                                       self.Y_train, size)
        self.X_test_batch, self.Y_test_batch, self.idxs_test_batch = create_batches(self.X_test.values, self.Y_test,
                                                                                    size)
        self.X_val_batch, self.Y_val_batch, self.idxs_val_batch = create_batches(self.X_val.values, self.Y_val,
                                                                                    size)


def create_batches(data, labels, size):
    data_batches_X = []
    data_batches_Y = []
    num_batches = len(data) // size if (len(data) % size == 0) else (len(data) // size) + 1
    len_sequence = len(data[0])
    idxs = list(range(len(data)))
    idxs_chosen = []

    for num_batch in range(num_batches):
        batchX = []
        batchY = []
        batch_idxs = []
        for _ in range(size):
            if len(idxs) > 0:
                random_elem = random.choice(idxs)
                batchX += [data[random_elem]]
                batchY += [labels[random_elem]]
                idxs.remove(random_elem)
                batch_idxs.append(random_elem)
        data_batches_X += [batchX]
        data_batches_Y += [batchY]
        idxs_chosen += [batch_idxs]

    return data_batches_X, data_batches_Y, idxs_chosen
