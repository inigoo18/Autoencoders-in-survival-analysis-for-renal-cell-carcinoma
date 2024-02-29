import pandas as pd
import numpy as np
import os
import torch

from SimpleTest.SimpleAE import AE
from SimpleTest.TabularDataLoader import TabularDataLoader
from SimpleTest.Trainer import Trainer
from SimpleTest.TrainingModel import TrainingModel

if __name__ == "__main__":
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS', 'CENSOR'], 0.2)

    aeModel = AE(d.input_dim_train())
    optim = torch.optim.Adam(aeModel.parameters(), lr = 0.001)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    instanceModel = TrainingModel("TestModel", d.X_train_batch, d.Y_train_batch, d.X_test_batch, d.Y_test_batch, aeModel, optim, 3, loss_fn)

    trainer = Trainer([instanceModel])
    trainer.trainAll()
    trainer.evaluateAll()
    print("Finish")

