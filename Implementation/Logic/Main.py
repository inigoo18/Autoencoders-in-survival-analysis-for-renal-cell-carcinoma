import numpy as np
import os
import torch

from Logic.Autoencoders.SimpleAE import AE
from Logic.Autoencoders.SparseAE import SparseAE
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType
from Logic.TabularDataLoader import TabularDataLoader
from Logic.Trainer import Trainer
from Logic.TrainingModel import TrainingModel

if __name__ == "__main__":
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS', 'CENSOR'], 0.2)

    torch.manual_seed(42)
    np.random.seed(42)
    L = 300
    #aeModel = AE(d.input_dim_train(), L)

    loss_args = {'reg_param': 0.1}
    loss_fn = LossHandler(LossType.SPARSE, loss_args)
    aeModel = SparseAE(d.input_dim_train(), L)
    optim = torch.optim.Adam(aeModel.parameters(), lr = 0.01)
    instanceModel = TrainingModel("TestModel", d.X_train_batch, d.Y_train_batch, d.X_test_batch, d.Y_test_batch,
                                  d.X_val_batch, d.Y_val_batch, aeModel, loss_fn, optim, 20, d.fetch_columns(), L)

    trainer = Trainer([instanceModel])
    trainer.trainAll()
    trainer.evaluateAll()
    print("Finish")

