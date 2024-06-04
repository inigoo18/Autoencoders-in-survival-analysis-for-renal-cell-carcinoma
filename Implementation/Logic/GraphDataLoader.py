import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
import math
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Logic.CustomDataset import CustomDataset
from Logic.IterationObject import IterationObject


class GraphDataLoader:
    """
    Class that holds the data where the graphs are located. Very similar to TabularDataLoader, check comments there
    for explainability.
    """

    def __init__(self, file_path, pred_vars, cli_vars, test_ratio, val_ratio, batch_size, folds, cohort):
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cli_vars = cli_vars
        self.pred_vars = pred_vars

        self.input_dim = len(graphs[0].nodes)

        graphs = reorder_dataset(graphs)

        graphs = filter_cohort(graphs, cohort)

        graphs = normalize_data(graphs, cli_vars)

        allDatasets = []

        for fold in range(folds):

            graphs = shift_data(graphs, folds)

            train_set, test_set, val_set = self.train_test_val_split(graphs, test_ratio, val_ratio)

            train_loader = self.custom_loader(train_set)
            test_loader = self.custom_loader(test_set)
            val_loader = self.custom_loader(val_set)

            train_loader = list(create_batches(train_loader, batch_size))
            test_loader = list(create_batches(test_loader, batch_size))
            val_loader = list(create_batches(val_loader, batch_size))

            it = IterationObject(train_loader, test_loader, val_loader, list(test_set[0].nodes()))
            allDatasets += [it]

        self.allDatasets = allDatasets

    def custom_loader(self, graphs):
        # we use the method collect_all_graph_data to convert from networkx object to Data object for use in the network
        gen_data = collect_all_graph_data(graphs, self.device)

        # we fetch all information within the graph regarding the clinical variables.
        cli_data = []
        for g in graphs:
            tmp = []
            for cli in self.cli_vars:
                tmp += [g.graph[cli]]
            cli_data += [tmp]

        # we fetch all predicted variables and then we create a DataFrame out of it to use the helper function
        # prepare_labels that's already implemented along TabularDataLoader
        pred_data = []
        for g in graphs:
            tmp = []
            for pred in self.pred_vars:
                tmp += [g.graph[pred]]
            pred_data += [tmp]
        pred_data = self.prepare_labels(pd.DataFrame(pred_data, columns = self.pred_vars))

        cd = CustomDataset(gen_data, torch.tensor(cli_data), torch.tensor(pred_data))
        return cd

    def train_test_val_split(self, graphs, test_ratio, val_ratio):
        '''
        Method that takes the general DF and separates it into train/test/val DFs while keeping the CENSOR variable
        in similar proportions between dataframes
        :param graphs: list of graphs (nx)
        :param test_ratio: float value representing test ratio
        :param val_ratio: float value representing val ratio
        :return: train, test and val sets (DFs)
        '''
        test_separation = int(len(graphs) - (len(graphs) * test_ratio))

        train_data = graphs[:test_separation]
        test_data = graphs[test_separation:]

        train_set, val_set = train_test_split(train_data, test_size=val_ratio, random_state=42)
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
        Data in any loader is usually ordered by batches. This method helps us unroll said batch and keep only the important data
        :param data: a loader
        :param dim: dimension to unroll by
        :return: if dim = 0, returns genetic data, if dim = 1, clinical, otherwise if dim = 2, returns targets
        '''
        if dim == 0:
            res = []
            for x in data:
                res += [x[dim]]
            return list(DataLoader(res, batch_size = len(res)))
        else:
            res = torch.tensor([]).to(self.device)
            for x in data:
                res = torch.cat((res, x[dim].to(self.device)), dim=0)
            return res

    def get_transcriptomic_data(self, data):
        return self.unroll_batch(data, 0)[0].x.reshape(-1, self.input_dim)


def create_batches(loader, batch_size):
    return DataLoader(loader, batch_size = batch_size, shuffle = False)


def collect_all_graph_data(graphs, device):
    D = []
    # edges are the same for all graphs so we only need to compute this once.
    G = graphs[0]
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    edge_index = torch.tensor([(node_to_index[edge[0]], node_to_index[edge[1]]) for edge in G.edges()] +
                              [(node_to_index[edge[1]], node_to_index[edge[0]]) for edge in G.edges()]).t().contiguous()

    for g in graphs:
        features = []
        for node, attr in g.nodes(data=True):
            features += [[float(attr['node_attr'])]]
        features = torch.tensor(features)
        d = Data(x=features, edge_index=edge_index)
        d.validate(raise_on_error=True)
        D += [d]
    return D


def normalize_data(graphs, cliVars):
    # Normalize expression level
    maxVal = -1
    minVal = -1
    for graph in graphs:
        for node, data in graph.nodes(data = True):
            for attribute, value in data.items():
                if (value > maxVal) or (maxVal == -1):
                    maxVal = value
                if (value < minVal) or (minVal == -1):
                    minVal = value

    for graph in graphs:
        for node, data in graph.nodes(data = True):
            for attribute, value in data.items():
                data[attribute] = value / maxVal

    # Normalize clinical data
    max_cli_vars_dict = {}

    for c in cliVars:
        max_cli_vars_dict[c] = -1

    for g in graphs:
        for c in cliVars:
            val = g.graph[c]
            if (val > max_cli_vars_dict[c] or max_cli_vars_dict[c] == -1):
                max_cli_vars_dict[c] = val

    for g in graphs:
        for c in cliVars:
            g.graph[c] = g.graph[c] / max_cli_vars_dict[c]

    return graphs

def shift_data(graphs, K):
    size = len(graphs)
    idx = (size // K) * (K - 1)
    lastrows = graphs[idx:]
    graphs_copy = graphs[:idx]
    return lastrows + graphs_copy

def filter_cohort(graphs, cohort):
    res = []
    for g in graphs:
        if (g.graph['TRT01P'] == cohort):
            res += [g]
        del g.graph['TRT01P']
    if cohort == 'ALL':
        return graphs
    return res

def validate_test_set(train_set, test_set):
    '''
        (Graph version)
        Train data should have the largest PFS, both in censored and uncensored. During the KFold process, if this were not
        the case, we need to swap patients around for this condition to hold.
        :param train_df: train df
        :param test_df: test df
        :return: train and test correctly validated.
        '''
    uncensoredTrainIdx, uncensoredTrainVal = (-1,-1)
    uncensoredTestIdx, uncensoredTestVal = (-1,-1)
    censoredTrainIdx, censoredTrainVal = (-1,-1)
    censoredTestIdx, censoredTestVal = (-1,-1)

    for g in range(len(train_set)):
        if (uncensoredTrainIdx == -1) or (train_set[g].graph['PFS_P_CNSR'] == 0 and train_set[g].graph['PFS_P'] > uncensoredTrainVal):
            uncensoredTrainIdx = g
            uncensoredTrainVal = train_set[g].graph['PFS_P']
        if (censoredTrainIdx == -1) or (train_set[g].graph['PFS_P_CNSR'] == 1 and train_set[g].graph['PFS_P'] > censoredTrainVal):
            censoredTrainIdx = g
            censoredTrainVal = train_set[g].graph['PFS_P']

    for g in range(len(test_set)):
        if (uncensoredTestIdx == -1) or (test_set[g].graph['PFS_P_CNSR'] == 0 and test_set[g].graph['PFS_P'] > uncensoredTestVal):
            uncensoredTestIdx = g
            uncensoredTestVal = test_set[g].graph['PFS_P']
        if (censoredTestIdx == -1) or (test_set[g].graph['PFS_P_CNSR'] == 1 and test_set[g].graph['PFS_P'] > censoredTestVal):
            censoredTestIdx = g
            censoredTestVal = test_set[g].graph['PFS_P']

    if train_set[uncensoredTrainIdx].graph['PFS_P'] < test_set[uncensoredTestIdx].graph['PFS_P']:
        tmp = train_set[uncensoredTrainIdx]
        train_set[uncensoredTrainIdx] = test_set[uncensoredTestIdx]
        test_set[uncensoredTestIdx] = tmp

    if train_set[censoredTrainIdx].graph['PFS_P'] < test_set[censoredTestIdx].graph['PFS_P']:
        tmp = train_set[censoredTrainIdx]
        train_set[censoredTrainIdx] = test_set[censoredTestIdx]
        test_set[censoredTestIdx] = tmp

    return train_set, test_set


def merge_lists(A, B, step):
    new_list = []
    len_A = len(A)
    len_B = len(B)

    i, j = 0, 0

    while i < len_A or j < len_B:
        # Append `step` elements from A
        for _ in range(step):
            if i < len_A:
                new_list.append(A[i])
                i += 1

        # Append one element from B
        if j < len_B:
            new_list.append(B[j])
            j += 1

    return new_list


def reorder_dataset(loaded_object):
    A_graphs = []
    B_graphs = []
    for g in loaded_object:
        if g.graph['PFS_P_CNSR'] == 0:
            A_graphs += [g]
        else:
            B_graphs += [g]

    step = math.floor(max(len(A_graphs), len(B_graphs)) / min(len(A_graphs), len(B_graphs)))
    maxGraphs, minGraphs = None, None

    if len(A_graphs) > len(B_graphs):
        maxGraphs = A_graphs
        minGraphs = B_graphs
    else:
        maxGraphs = B_graphs
        minGraphs = A_graphs

    reshaped_graphs = merge_lists(maxGraphs, minGraphs, step)

    return reshaped_graphs