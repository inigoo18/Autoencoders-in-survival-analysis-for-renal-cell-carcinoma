import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from torch.utils.data import DataLoader

from Logic.CustomDataset import CustomDataset
from Logic.IterationObject import IterationObject


class TabularDataLoader:
    """
    Class that holds the DataFrame where the tabular data is located.
    """

    def __init__(self, file_path, pred_vars, cli_vars, test_ratio, val_ratio, batch_size, folds):
        # Load file and convert into float32 since model parameters initialized w/ Pytorch are in float32
        dataframe = pd.read_csv(file_path, sep=',', index_col=0).astype('float32')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cli_vars = cli_vars
        self.pred_vars = pred_vars

        self.input_dim = len(dataframe.columns) - len(cli_vars) - len(pred_vars)

        dataframe = normalize_data(dataframe, cli_vars)

        allDatasets = []

        for fold in range(folds):

            dataframe = shift_data(dataframe, folds)

            train_set, test_set, val_set = self.train_test_val_split(dataframe, test_ratio, val_ratio)

            train_loader = self.custom_loader(train_set)
            test_loader = self.custom_loader(test_set)
            val_loader = self.custom_loader(val_set)

            train_loader = list(create_batches(train_loader, batch_size))
            test_loader = list(create_batches(test_loader, batch_size))
            val_loader = list(create_batches(val_loader, batch_size))

            it = IterationObject(train_loader, test_loader, val_loader)
            allDatasets += [it]

        self.allDatasets = allDatasets

    def custom_loader(self, DF):
        DF_gen = DF.drop(self.pred_vars + self.cli_vars, axis=1).values
        DF_cli = DF[self.cli_vars].values
        pred_vals = self.prepare_labels(DF[self.pred_vars])

        cd = CustomDataset(torch.tensor(DF_gen).to(self.device), torch.tensor(DF_cli).to(self.device), torch.tensor(pred_vals).to(self.device))
        return cd

    def train_test_val_split(self, tabular_data, test_ratio, val_ratio):
        '''
        Method that takes the general DF and separates it into train/test/val DFs while keeping the CENSOR variable
        in similar proportions between dataframes
        :param tabular_data: DF
        :param test_ratio: float value representing test ratio
        :param val_ratio: float value representing val ratio
        :return: train, test and val sets (DFs)
        '''
        A_indices = tabular_data[tabular_data['PFS_P_CNSR'] == 0].index
        B_indices = tabular_data[tabular_data['PFS_P_CNSR'] == 1].index

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
        pfs = dataframe['PFS_P']
        cnsr = dataframe['PFS_P_CNSR']
        result = []
        for p, c in zip(pfs, cnsr):
            b = False
            if c == 0:
                b = True
            result += [(b, p)]
        return result

    def unroll_batch(self, data, dim):
        '''
        Data in any loader is usually ordered by batches. This method helps us unroll said batch and keep only the genetic data
        :param data:
        :return:
        '''
        res = torch.tensor([]).to(self.device)
        for x in data:
            res = torch.cat((res, x[dim]), dim = 0)
        return res




def create_batches(loader, batch_size):
    return DataLoader(loader, batch_size = batch_size, shuffle = False)


def normalize_data(dataframe, cliVars, mode = "Max"):
    '''
    Normalizes a dataframe after removing PFS and CENSOR columns. Once the normalization is done, we add the cols back in
    :param dataframe: DF to normalize
    :return: normalized DF based in the genetic expressions
    '''
    print("Using", mode, "normalization");
    DF = dataframe.drop(['PFS_P', 'PFS_P_CNSR'] + cliVars, axis=1)
    DF_cli = dataframe[cliVars]
    maxVal = max([x for L in DF.values for x in L])
    minVal = min([x for L in DF.values for x in L])
    if mode == "Max":
        X_normalized = DF / maxVal
    else:
        X_normalized = (DF - minVal) / (maxVal - minVal)

    X_normalized['PFS_P'] = dataframe['PFS_P']
    X_normalized['PFS_P_CNSR'] = dataframe['PFS_P_CNSR']

    if mode == "Max":
        DF_cli = DF_cli / DF_cli.max()
    else:
        DF_cli = (DF_cli - DF_cli.min()) / (DF_cli.max() - DF_cli.min())

    X_normalized = pd.concat([X_normalized, DF_cli], axis = 1)

    return X_normalized

def shift_data(df, K):
    df_copy = df.copy(deep=True)

    lastRows = df_copy[(len(df) // K) * (K - 1):]
    df_copy = df_copy.drop(lastRows.index)
    df_copy = pd.concat([lastRows, df_copy])
    return df_copy