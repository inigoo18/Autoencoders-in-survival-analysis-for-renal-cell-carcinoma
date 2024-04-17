import numpy as np
import os
import torch

from Logic.Autoencoders.GNNExample import GNNExample
from Logic.Autoencoders.MinWorkingExample import MWE_AE
from Logic.Autoencoders.VariationalExample import VariationalExample
from Logic.FoldObject import FoldObject
from Logic.GraphDataLoader import GraphDataLoader
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType
from Logic.TabularDataLoader import TabularDataLoader
from Logic.Trainer import Trainer
from Logic.TrainingModel import TrainingModel

def tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1, BATCH_SIZE, FOLDS) # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    combinations = [[]]
    foldObjects = []

    for comb in combinations:
        print(comb)
        foldObject = FoldObject(comb, FOLDS, d.allDatasets)
        for fold in range(FOLDS):
            title = "TAB{{L_"+str(L) + "_F_" + str(fold) + "_" + '+'.join(str(loss.name) for loss in comb)
            loss_fn = LossHandler(comb, loss_args, None)
            aeModel = MWE_AE(d.input_dim, L)
            vaeModel = VariationalExample(d.input_dim, L)
            if LossType.VARIATIONAL in comb:
                aeModel = vaeModel
            optim = torch.optim.Adam(aeModel.parameters(), lr = 0.0001)
            instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                        aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L, False)#, 'best_model_loss_1478.pth')

            trainer = Trainer(instanceModel)
            trainer.train()
            meanRes, mseError = trainer.evaluate()
            foldObject.MSE += [mseError]
            foldObject.ROC += [meanRes]
        foldObjects += [foldObject]




def graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_graph_R3.pkl'))

    d = GraphDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1,
                          BATCH_SIZE, FOLDS)  # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    combinations = [[]]

    foldObjects = []

    for comb in combinations:
        print(comb)
        foldObject = FoldObject(comb, FOLDS, d.allDatasets)
        for fold in range(FOLDS):
            title = "GPH{{L_" + str(L) + "_F_" + str(fold) + "_" + '+'.join(str(loss.name) for loss in comb)
            loss_fn = LossHandler(comb, loss_args, None)
            aeModel = GNNExample(1, d.input_dim, L, BATCH_SIZE)
            vaeModel = VariationalExample(d.input_dim, L)
            if LossType.VARIATIONAL in comb:
                print("VARIATIONAL NOT YET IMPLEMENTED FOR GNNs!!!")
                #aeModel = vaeModel
            optim = torch.optim.Adam(aeModel.parameters(), lr=0.0001)
            instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                          aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L,
                                          True)  # , 'best_model_loss_1478.pth')

            trainer = Trainer(instanceModel)
            trainer.train()
            meanRes, mseError = trainer.evaluate()
            foldObject.MSE += [mseError]
            foldObject.ROC += [meanRes]
        foldObjects += [foldObject]


if __name__ == "__main__":
    option = "Graph"

    torch.manual_seed(42)
    np.random.seed(42)

    L = 256
    loss_args = {'noise_factor': 0.05, 'reg_param': 0.35, 'rho': 0.001}
    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']
    EPOCHS = 7
    FOLDS = 3

    if option == "Tabular":
        BATCH_SIZE = 32
        tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS)
    else:
        BATCH_SIZE = 32
        graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS)
