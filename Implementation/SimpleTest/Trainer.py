import torch
import math

from typing import List

from sklearn.model_selection import KFold, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis

from SimpleTest.TrainingModel import TrainingModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Trainer:
    """
    Takes list of models and trains them one by one with the parameters that the model itself holds.
    """


    def __init__(self, models: List[TrainingModel]):
        self.models = models # list of models


    def train(self, idx):
        tr_model = self.models[idx]
        tr_model.model.train()
        for t in range(tr_model.epochs):
            num_batches = len(tr_model.X_train)
            epoch_loss = 0.0
            for b in range(num_batches):
                x_batch = torch.tensor(tr_model.X_train[b])
                y_batch = torch.tensor(tr_model.y_train[b])

                # Perform forward pass
                x_pred_batch = tr_model.model.forward(x_batch)

                # Compute loss for the entire batch
                loss = tr_model.loss_function(x_pred_batch, x_batch)

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

                # Zero gradients, backward pass, and update parameters
                tr_model.optim.zero_grad()
                loss.backward()
                tr_model.optim.step()

            avg_epoch_loss = epoch_loss / num_batches
            print("Epoch", t, " completed with average loss: ", avg_epoch_loss)

        print("End of training report")

    def trainAll(self):
        for idx in range(len(self.models)):
            self.train(idx)

    def evaluate(self, idx):
        eval_model = self.models[idx]
        eval_model.model.eval()

        latent_cols = ["Latent " + str(x) for x in list(range(eval_model.L))]

        latent_space_train = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtrain())).detach().numpy()
        latent_space_test = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtest())).detach().numpy()

        coxnet_model = CoxnetSurvivalAnalysis(l1_ratio = 0.9, alpha_min_ratio = 0.001)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FitFailedWarning)
        coxnet_model.fit(latent_space_train, eval_model.unroll_Ytrain())

        estimated_alphas = coxnet_model.alphas_
        cv = KFold(n_splits=5, shuffle = True, random_state = 46)
        gcv = GridSearchCV(
            CoxnetSurvivalAnalysis(l1_ratio = 0.9),
            param_grid = {"alphas": [[v] for v in estimated_alphas]},
            cv = cv,
            error_score = 0.5,
            n_jobs = 1,
        ).fit(latent_space_train, eval_model.unroll_Ytrain())

        cv_results = pd.DataFrame(gcv.cv_results_)

        alphas = cv_results.param_alphas.map(lambda x: x[0])
        mean = cv_results.mean_test_score
        std = cv_results.std_test_score

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(alphas, mean)
        ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylabel("concordance index")
        ax.set_xlabel("alpha")
        ax.axvline(gcv.best_params_["alphas"][0], c="C1")
        ax.axhline(0.5, color="grey", linestyle="--")
        ax.grid(True)
        plt.show()


        best_model = gcv.best_estimator_
        best_coefs = pd.DataFrame(best_model.coef_, index=latent_cols, columns=["coefficient"])

        non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
        print(f"Number of non-zero coefficients: {non_zero}")

        if non_zero == 0:
            print("All coefficients are 0...")
            return

        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        _, ax = plt.subplots(figsize=(6, 8))
        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
        ax.set_xlabel("coefficient")
        ax.grid(True)

        plt.show()

        print("Finished")


        # TODO: before proceeding, we need to give a name to each latent feature.
        # make sure that all works well and see if we can relate the latent features to the actual genes in some way
        # make sure not to use the StandardScaler in the make_pipeline thing.
        # we use estimated_alphas in order to gauge more or less which are the alphas that we're going to actually want
        # (how does that work? find out)
        # the plot with coefficients for each feature is going to be too much for 150 features. Try to split it,
        # or maybe only select the most relevant ones. (I think splitting is a good choice)

        # link (see which features are most important in cox ph model):
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html

        # link (how to evaluate survival models):
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html


    def evaluateAll(self):
        for idx in range(len(self.models)):
            self.evaluate(idx)




def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")