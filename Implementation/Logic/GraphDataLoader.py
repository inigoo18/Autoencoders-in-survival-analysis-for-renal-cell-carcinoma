import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Logic.CustomDataset import CustomDataset


class GraphDataLoader:
    """
    Class that holds the data where the graphs are located.
    """

    def __init__(self, file_path, pred_vars, cli_vars, test_ratio, val_ratio, batch_size):
        # Load file and convert into float32 since model parameters initialized w/ Pytorch are in float32
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cli_vars = cli_vars
        self.pred_vars = pred_vars

        graphs = normalize_data(graphs, cli_vars)

        train_set, test_set, val_set = self.train_test_val_split(graphs, test_ratio, val_ratio)

        train_loader = self.custom_loader(train_set)
        test_loader = self.custom_loader(test_set)
        val_loader = self.custom_loader(val_set)

        self.train_loader = list(create_batches(train_loader, batch_size))
        self.test_loader = list(create_batches(test_loader, batch_size))
        self.val_loader = list(create_batches(val_loader, batch_size))

    def describe_dataframe(self):
        return self.dataframe.describe()

    def fetch_columns(self):
        return self.dataframe.columns

    def input_dim(self):

        # K batches
        # 3 features (CustomDataset) where 0: genData, 1: cliData, 2: labels
        # N patients
        # M features
        # TODO : some analysis
        #print(len(self.train_loader)) # 15
        #print(len(self.train_loader[0])) # 3
        #print(len(self.train_loader[0][0])) # 32
        #print(len(self.train_loader[0][0][0])) # 2
        return self.train_loader[0][0][0].x.shape[1]

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
        A_indices = []
        B_indices = []
        # first we load graphs that are censored and others that aren't
        for g in graphs:
            if g.graph['PFS_P_CNSR'] == 0:
                A_indices += [g]
            else:
                B_indices += [g]

        # Splitting A_indices into training, testing, and validation sets
        A_train, A_temp = train_test_split(A_indices, test_size=test_ratio + val_ratio, random_state=42)
        A_test, A_val = train_test_split(A_temp, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

        # Splitting B_indices into training, testing, and validation sets
        B_train, B_temp = train_test_split(B_indices, test_size=test_ratio + val_ratio, random_state=42)
        B_test, B_val = train_test_split(B_temp, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

        # Combining the sets
        train_set = list(A_train) + list(B_train)
        test_set = list(A_test) + list(B_test)
        val_set = list(A_val) + list(B_val)

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



def create_batches(loader, batch_size):
    return DataLoader(loader, batch_size = batch_size, shuffle = True)


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
        D += [d.to(device)]
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

