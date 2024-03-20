import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np


class TabularDataLoader:
    """
    Class that holds the DataFrame where the tabular data is located.
    """

    def __init__(self, file_path, pred_vars, test_ratio, val_ratio, batch_size):
        # Load file and convert into float32 since model parameters initialized w/ Pytorch are in float32
        dataframe = pd.read_csv(file_path, sep=',', index_col=0).astype('float32')
        self.dataframe = dataframe

        dataframe = normalize_data(dataframe)

        train_set, test_set, val_set = self.train_test_val_split(dataframe, test_ratio, val_ratio)

        self.X_train = train_set.drop(columns = pred_vars)
        self.X_test = test_set.drop(columns = pred_vars)
        self.X_val = val_set.drop(columns=pred_vars)

        self.Y_train = self.prepare_labels(train_set)
        self.Y_test = self.prepare_labels(test_set)
        self.Y_val = self.prepare_labels(val_set)

        # We keep this in order to use it for demographic data and stuff
        self.Y_dataframe = test_set

        #self._normalize_data()
        self._create_batches(batch_size)

    def describe_dataframe(self):
        return self.dataframe.describe()

    def fetch_columns(self):
        return self.dataframe.columns

    def input_dim_train(self):
        return len(self.X_train.iloc[0])

    def input_dim_test(self):
        return len(self.Y_train.iloc[0])

    def train_test_val_split(self, tabular_data, test_ratio, val_ratio):
        '''
        Method that takes the general DF and separates it into train/test/val DFs while keeping the CENSOR variable
        in similar proportions between dataframes
        :param tabular_data: DF
        :param test_ratio: float value representing test ratio
        :param val_ratio: float value representing val ratio
        :return: train, test and val sets (DFs)
        '''
        A_indices = tabular_data[tabular_data['CENSOR'] == 0].index
        B_indices = tabular_data[tabular_data['CENSOR'] == 1].index

        # Splitting A_indices into training, testing, and validation sets
        A_train, A_temp = train_test_split(A_indices, test_size=test_ratio + val_ratio, random_state=42)
        A_test, A_val = train_test_split(A_temp, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

        # Splitting B_indices into training, testing, and validation sets
        B_train, B_temp = train_test_split(B_indices, test_size=test_ratio + val_ratio, random_state=42)
        B_test, B_val = train_test_split(B_temp, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

        # Combining the sets
        train_indices = list(A_train) + list(B_train)
        test_indices = list(A_test) + list(B_test)
        val_indices = list(A_val) + list(B_val)

        # Creating the sets
        train_set = tabular_data.loc[train_indices]
        test_set = tabular_data.loc[test_indices]
        val_set = tabular_data.loc[val_indices]

        return train_set, test_set, val_set

    def prepare_labels(self, dataframe):
        '''
        Labels are supposed to be in the form of (censorship, time of event)
        '''
        pfs = dataframe['PFS']
        cnsr = dataframe['CENSOR']
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

def normalize_data(dataframe):
    '''
    Normalizes a dataframe after removing PFS and CENSOR columns. Once the normalization is done, we add the cols back in
    :param dataframe: DF to normalize
    :return: normalized DF based in the genetic expressions
    '''
    DF = dataframe.drop(['PFS', 'CENSOR'], axis=1)
    maxVal = max([x for L in DF.values for x in L])
    X_normalized = DF / maxVal

    X_normalized['PFS'] = dataframe['PFS']
    X_normalized['CENSOR'] = dataframe['CENSOR']

    return X_normalized
