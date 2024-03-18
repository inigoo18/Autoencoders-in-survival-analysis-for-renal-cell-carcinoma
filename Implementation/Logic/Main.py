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

if __name__ == "__main__":
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS', 'CENSOR'], 0.2, 0.1) # 70% train, 20% test, 10% val

    torch.manual_seed(42)
    np.random.seed(42)
    L = 300
    #aeModel = AE(d.input_dim_train(), L)

    losses = [LossType.VARIATIONAL]
    loss_args = {'noise_factor': 0.5}#{'reg_param': 0.2, 'rho': 0.2}
    loss_fn = LossHandler(losses, loss_args)
    aeModel = MWE_AE(d.input_dim_train(), L)
    vaeModel = VariationalExample(d.input_dim_train(), L)
    aeModel = vaeModel
    optim = torch.optim.Adam(aeModel.parameters(), lr = 0.01)
    instanceModel = TrainingModel("MWP_VARIATIONAL", d.X_train_batch, d.Y_train_batch, d.X_test_batch, d.Y_test_batch,
                                  d.X_val_batch, d.Y_val_batch, d.Y_dataframe, aeModel, loss_fn, optim, 20, d.fetch_columns(), L)#, 'best_model_loss_1478.pth')

    trainer = Trainer([instanceModel])
    trainer.trainAll()
    trainer.evaluateAll()

