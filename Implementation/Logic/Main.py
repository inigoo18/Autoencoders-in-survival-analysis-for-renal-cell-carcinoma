import numpy as np
import os
import torch

from Logic.Autoencoders.GNNExample import GNNExample
from Logic.Autoencoders.MinWorkingExample import MWE_AE
from Logic.Autoencoders.VariationalExample import VariationalExample
from Logic.GraphDataLoader import GraphDataLoader
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType
from Logic.TabularDataLoader import TabularDataLoader
from Logic.Trainer import Trainer
from Logic.TrainingModel import TrainingModel

def tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1, BATCH_SIZE) # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    instanceModels = []

    #combinations = [[]]

    for comb in combinations:
        print(comb)
        title = "TABULAR_L_"+str(L)+ "_" + '+'.join(str(loss.name) for loss in comb)
        loss_fn = LossHandler(comb, loss_args)
        aeModel = MWE_AE(d.input_dim(), L)
        vaeModel = VariationalExample(d.input_dim(), L)
        if LossType.VARIATIONAL in comb:
            aeModel = vaeModel
        optim = torch.optim.Adam(aeModel.parameters(), lr = 0.0001)
        instanceModel = TrainingModel(title, d, clinicalVars,
                                    aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L, False)#, 'best_model_loss_1478.pth')
        instanceModels += [instanceModel]

    trainer = Trainer(instanceModels)
    trainer.trainAll()
    trainer.evaluateAll()



def graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_graph_R3.pkl'))

    d = GraphDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1,
                          BATCH_SIZE)  # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    instanceModels = []

    #combinations = [[]]

    print("Input dim:", d.input_dim())
    for comb in combinations:
        print(comb)
        title = "GRAPH_L_" + str(L) + "_" + '+'.join(str(loss.name) for loss in comb)
        loss_fn = LossHandler(comb, loss_args)
        aeModel = GNNExample(1, d.input_dim(), L, BATCH_SIZE)
        vaeModel = VariationalExample(d.input_dim(), L)
        if LossType.VARIATIONAL in comb:
            aeModel = vaeModel
        optim = torch.optim.Adam(aeModel.parameters(), lr=0.005)
        instanceModel = TrainingModel(title, d, clinicalVars,
                                      aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE,
                                      L, True)#, 'model_lossGRAPH_L_256__182.pth')
        instanceModels += [instanceModel]

    trainer = Trainer(instanceModels)
    trainer.trainAll()
    trainer.evaluateAll()



if __name__ == "__main__":
    option = "Graph"

    torch.manual_seed(42)
    np.random.seed(42)

    L = 256
    loss_args = {'noise_factor': 0.05, 'reg_param': 0.35, 'rho': 0.001}
    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']
    EPOCHS = 100

    if option == "Tabular":
        BATCH_SIZE = 32
        tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS)
    else:
        BATCH_SIZE = 32
        graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS)
