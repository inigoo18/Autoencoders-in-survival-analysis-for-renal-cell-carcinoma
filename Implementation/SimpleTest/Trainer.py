import torch
import math

from typing import List

from SimpleTest.TrainingModel import TrainingModel


class Trainer:
    """
    Takes list of models and trains them one by one with the parameters that the model itself holds.
    """


    def __init__(self, models: List[TrainingModel]):
        self.models = models # list of models


    def train(self, idx):
        model = self.models[idx]
        loss = None

        for t in range(model.epochs):
            x_batch = model.X[t]
            y_batch = model.y[t]
            for i in range(len(model.X)):
                x = torch.tensor(x_batch[i])
                y = y_batch[i]

                x_pred = model.model.forward(x)

                loss = model.loss_function(x, x_pred)

                # zero all of the gradients for the variables it will update
                model.optim.zero_grad()

                # backward pass: compute gradient of the loss w/ respect to model parameters
                loss.backward()

                # update parameters
                model.optim.step()

            print("Epoch", t, " completed with loss: ", loss)

        print("End of training report")
        print("Final loss obtained:", loss)

    def trainAll(self):
        for idx in range(len(self.models)):
            self.train(idx)


