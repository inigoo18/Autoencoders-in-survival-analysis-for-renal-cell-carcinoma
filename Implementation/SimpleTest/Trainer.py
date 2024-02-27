import torch
import math

from typing import List

from SimpleTest.NNModel import NNModel


class Trainer:
    """
    Takes list of models and trains them one by one with the parameters that the model itself holds.
    """


    def __init__(self, models: List[NNModel], data):
        self.models = models # list of models
        self.data = data # DataLoader object


    def train(self, idx):
        model = self.models[idx]
        for t in range(model.epochs):
            y_pred = model.forward(self)

            loss = model.loss_function(y_pred, self.data.Y)
            if t % 100 == 99:
                print(t, loss.item())

            # zero all of the gradients for the variables it will update
            model.optim.zero_grad()

            # backward pass: compute gradient of the loss w/ respect to model parameters
            loss.backward()

            # update parameters
            model.optim.step()


