import numpy as np
import os
import torch

from Logic.Autoencoders.MinWorkingExample import MWE_AE
from Logic.Autoencoders.VariationalExample import VariationalExample
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType
from Logic.TabularDataLoader import TabularDataLoader
from Logic.Trainer import Trainer
from Logic.TrainingModel import TrainingModel

def tabular_network():
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    BATCH_SIZE = 32

    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']

    d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1, BATCH_SIZE) # 70% train, 20% test, 10% val

    torch.manual_seed(42)
    np.random.seed(42)
    L = 32
    #aeModel = AE(d.input_dim_train(), L)

    losses = [LossType.DENOISING, LossType.SPARSE_KL, LossType.VARIATIONAL]

    combinations = []
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    combinations += [losses]

    loss_args = {'noise_factor': 0.1, 'reg_param': 0.15, 'rho': 0.01}

    instanceModels = []

    #combinations = [[]]

    for comb in combinations:
        print(comb)
        title = "RESULT_L_"+str(L)+ "_" + '+'.join(str(loss.name) for loss in comb)
        loss_fn = LossHandler(comb, loss_args)
        aeModel = MWE_AE(d.input_dim(), L)
        vaeModel = VariationalExample(d.input_dim(), L)
        if LossType.VARIATIONAL in comb:
            aeModel = vaeModel
        optim = torch.optim.Adam(aeModel.parameters(), lr = 0.1)
        instanceModel = TrainingModel(title, d.train_loader, d.test_loader, d.val_loader, clinicalVars,
                                    aeModel, loss_fn, optim, 5, BATCH_SIZE, d.fetch_columns(), L)#, 'best_model_loss_1478.pth')
        instanceModels += [instanceModel]



    trainer = Trainer(instanceModels)
    trainer.trainAll()
    trainer.evaluateAll()



def graph_network():
    pass



if __name__ == "__main__":
    option = "Tabular"
    if option == "Tabular":
        tabular_network()
    else:
        graph_network()
