import torch
import math

from typing import List

from sksurv.linear_model import CoxPHSurvivalAnalysis

from SimpleTest.TrainingModel import TrainingModel
import numpy as np


class Trainer:
    """
    Takes list of models and trains them one by one with the parameters that the model itself holds.
    """


    def __init__(self, models: List[TrainingModel]):
        self.models = models # list of models


    def train(self, idx):
        tr_model = self.models[idx]
        loss = None
        tr_model.model.train()
        for t in range(tr_model.epochs):
            x_batch = tr_model.X_train[t]
            y_batch = tr_model.y_train[t]
            for i in range(len(x_batch)):
                x = torch.tensor(x_batch[i])
                y = y_batch[i]

                x_pred = tr_model.model.forward(x)

                loss = tr_model.loss_function(x, x_pred)

                # zero all of the gradients for the variables it will update
                tr_model.optim.zero_grad()

                # backward pass: compute gradient of the loss w/ respect to model parameters
                loss.backward()

                # update parameters
                tr_model.optim.step()

            print("Epoch", t, " completed with loss: ", loss)

        print("End of training report")
        print("Final loss obtained:", loss)

    def trainAll(self):
        for idx in range(len(self.models)):
            self.train(idx)

    def evaluate(self, idx):
        eval_model = self.models[idx]
        eval_model.model.eval()

        latent_space_train = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtrain())).detach().numpy()
        latent_space_test = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtest())).detach().numpy()

        alphas = 10.0 ** np.linspace(-4,4,50)
        cph = CoxPHSurvivalAnalysis()
        for alpha in alphas:
            cph.set_params(alpha=alpha)
            smth = eval_model.unroll_Ytrain()
            cph.fit(latent_space_train, eval_model.unroll_Ytrain())
            # y :: (1, 10) -> event occurred at time 10
            # y :: (0, 05) -> event was censored at time 5
            survival_function = cph.predict(latent_space_test)
            print(survival_function)


    def evaluateAll(self):
        for idx in range(len(self.models)):
            self.evaluate(idx)


