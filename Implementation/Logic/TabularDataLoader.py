import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from torch.utils.data import DataLoader
import math

from Logic.CustomDataset import CustomDataset
from Logic.IterationObject import IterationObject


class TabularDataLoader:
    """
    Class that holds the DataFrame where the tabular data is located.
    """

    def __init__(self, file_path, pred_vars, cli_vars, test_ratio, val_ratio, batch_size, folds, cohort):
        # Load file and convert into float32 since model parameters initialized w/ Pytorch are in float32
        dataframe = pd.read_csv(file_path, sep=',', index_col=0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cli_vars = cli_vars
        self.pred_vars = pred_vars

        # we shuffle the dataset so that censored and uncensored patients alternate. This way, when we do cross-val,
        # we have a sufficient number of censored and uncensored patients within each fold
        dataframe = reorder_dataframe(dataframe)

        # we filter by a specific treatment arm
        dataframe = filter_cohort(dataframe, cohort)

        self.input_dim = len(dataframe.columns) - len(cli_vars) - len(pred_vars)

        dataframe = dataframe.astype('float32')

        # normalize the dataset with the maximal transcriptomic value
        dataframe = normalize_data(dataframe, cli_vars)

        allDatasets = []
        allColumnNames = []

        # for each fold...
        for fold in range(folds):

            # we shift the data one fold
            dataframe = shift_data(dataframe, folds)

            # we create train, test, val sets using cross validation.
            train_set, test_set, val_set = self.train_test_val_split(dataframe, test_ratio, val_ratio)

            train_loader = self.custom_loader(train_set)
            test_loader = self.custom_loader(test_set)
            val_loader = self.custom_loader(val_set)

            train_loader = list(create_batches(train_loader, batch_size))
            test_loader = list(create_batches(test_loader, batch_size))
            val_loader = list(create_batches(val_loader, batch_size))

            # this IterationObject allows us to keep track of the train, test and val sets for a given fold
            it = IterationObject(train_loader, test_loader, val_loader, test_set.columns)
            allDatasets += [it]

        self.allDatasets = allDatasets

    def custom_loader(self, DF):
        '''
        We initialize a CustomDataset class that takes in the genetic, clinical and features to be predicted.
        :param DF: the dataframe we're currently using
        :return: CustomDataset object with genetic, clinical and predictor variables.
        '''
        DF_gen = DF.drop(self.pred_vars + self.cli_vars, axis=1).values
        DF_cli = DF[self.cli_vars].values
        pred_vals = self.prepare_labels(DF[self.pred_vars])

        cd = CustomDataset(torch.tensor(DF_gen).to(self.device), torch.tensor(DF_cli), torch.tensor(pred_vals))
        return cd

    def train_test_val_split(self, tabular_data, test_ratio, val_ratio):
        '''
        Method that takes the general DF and separates it into train/test using cross-val, then train set is further
        separated into train/val.
        :param tabular_data: DF
        :param test_ratio: float value representing test ratio (if 10 folds, then 1/10)
        :param val_ratio: float value representing val ratio
        :return: train, test and val sets (DFs)
        '''

        test_separation = int(len(tabular_data) * test_ratio)

        train_data = tabular_data[test_separation:]
        test_data = tabular_data[:test_separation]

        train_set, val_set = train_test_split(train_data, test_size = val_ratio, random_state = 32)
        test_set = test_data

        train_set, test_set = validate_test_set(train_set, test_set)

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
            result += [(b, round(p/3, 2))]
        return result

    def unroll_batch(self, data, dim):
        '''
        Data in any loader is usually ordered by batches. This method helps us unroll said batch and keep only the genetic data
        :param data: data ordered by batches
        :return: all samples in a tensor
        '''
        res = torch.tensor([]).to(self.device)
        for x in data:
            res = torch.cat((res, x[dim].to(self.device)), dim = 0)
        return res

    def get_transcriptomic_data(self, data):
        '''
        Helper function that calls unroll_batch, and detaches the tensor, since scikit-surv cannot take in tensors.
        :param data: data ordered by batches
        :return: all samples in a numpy array
        '''
        return self.unroll_batch(data, 0).cpu().detach().numpy()




def create_batches(loader, batch_size):
    '''
    Creates a DataLoader instance, which keeps track of the different batches.
    :param loader: whether train, test or val loader
    :param batch_size: size of the batch
    :return:
    '''
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

def filter_cohort(df, cohort):
    if cohort == 'ALL':
        return df.drop('TRT01P', axis = 1)
    df = df[df['TRT01P'] == cohort] #TRT01P is the feature for cohort
    df = df.drop('TRT01P', axis = 1) # we remove the feature as we no longer need it.
    return df

def swap_patients(train_df, test_df, train_pat, test_pat):
    '''
    Method used in validate_test_set to make sure the invariant holds (check validate_test_set). Performs a simple swap
    '''
    train_df_t = train_df.drop(train_pat.name)
    test_df_t = test_df.drop(test_pat.name)
    train_df_t.loc[test_pat.name] = test_pat
    test_df_t.loc[train_pat.name] = train_pat
    return train_df_t, test_df_t

def validate_test_set(train_df, test_df):
    '''
    Train data should have the largest PFS, both in censored and uncensored. During the KFold process, if this were not
    the case, we need to swap patients around for this condition to hold.
    :param train_df: train df
    :param test_df: test df
    :return: train and test correctly validated.
    '''
    train_UncensoredPFS = train_df[train_df['PFS_P_CNSR'] == 0]['PFS_P']
    train_CensoredPFS = train_df[train_df['PFS_P_CNSR'] == 1]['PFS_P']

    test_UncensoredPFS = test_df[test_df['PFS_P_CNSR'] == 0]['PFS_P']
    test_CensoredPFS = test_df[test_df['PFS_P_CNSR'] == 1]['PFS_P']

    if (max(test_CensoredPFS) > max(train_CensoredPFS)):
        # swap max test with max train
        test_pat = test_df.loc[test_CensoredPFS.idxmax()]
        train_pat = train_df.loc[train_CensoredPFS.idxmax()]
        train_df, test_df = swap_patients(train_df, test_df, train_pat, test_pat)


    if (max(test_UncensoredPFS) > max(train_UncensoredPFS)):
        # swap max test with max train
        test_pat = test_df.loc[test_UncensoredPFS.idxmax()]
        train_pat = train_df.loc[train_UncensoredPFS.idxmax()]
        train_df, test_df = swap_patients(train_df, test_df, train_pat, test_pat)

    return train_df, test_df


def merge_lists(A, B, step):
    '''
    Merges two lists A and B using a specific 'step'. This is used to reorder the dataset so that censored and
    uncensored patients are spread uniformly.
    :param A: a list
    :param B: another list
    :param step: step to integrate B within A
    :return:
    '''
    new_list = []
    len_A = len(A)
    len_B = len(B)

    i, j = 0, 0

    while i < len_A or j < len_B:
        # Append step elements from A
        for _ in range(step):
            if i < len_A:
                new_list.append(A[i])
                i += 1

        # Append one element from B
        if j < len_B:
            new_list.append(B[j])
            j += 1

    return new_list

def reorder_dataframe(gene_data):
    '''
    Method used to make sure that the censored and uncensored patients are spread uniformly in the dataset.
    This allows us to make folds where, in each one of them, are censored/uncensored patients.
    '''
    A_indices = gene_data[gene_data['PFS_P_CNSR'] == 0].index
    B_indices = gene_data[gene_data['PFS_P_CNSR'] == 1].index

    step = math.floor(max(len(A_indices), len(B_indices)) / min(len(A_indices), len(B_indices)))
    maxIndices, minIndices = None, None

    if len(A_indices) > len(B_indices):
        maxIndices = A_indices
        minIndices = B_indices
    else:
        maxIndices = B_indices
        minIndices = A_indices

    DF_indices = merge_lists(maxIndices, minIndices, step)
    gene_data = gene_data.loc[DF_indices]
    return gene_data