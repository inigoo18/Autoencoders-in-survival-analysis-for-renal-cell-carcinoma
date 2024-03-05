import torch
import math

from typing import List

from sklearn.model_selection import KFold, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis

from Logic.TrainingModel import TrainingModel
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
        tr_model.loss_fn.clear()
        best_validation_loss = float('inf')
        best_model_state = None
        best_epoch = -1
        tr_losses = []
        val_losses = []
        for t in range(tr_model.epochs + 1):
            tr_model.model.train()
            num_train_batches = len(tr_model.X_train)
            train_loss = 0.0
            for b in range(num_train_batches):
                x_batch = torch.tensor(tr_model.X_train[b])
                y_batch = torch.tensor(tr_model.y_train[b])

                # Perform forward pass
                x_pred_batch = tr_model.model.forward(x_batch)

                # Compute loss for the entire batch
                loss = tr_model.compute_model_loss(x_pred_batch, x_batch)

                # Accumulate loss for the epoch
                train_loss += loss.item()

                # Zero gradients, backward pass, and update parameters
                tr_model.optim.zero_grad()
                loss.backward()
                tr_model.optim.step()


            valid_loss = 0.0
            tr_model.model.eval()
            num_val_batches = len(tr_model.X_val)
            for b in range(num_val_batches):
                x_batch = torch.tensor(tr_model.X_train[b])
                y_batch = torch.tensor(tr_model.y_train[b])
                x_pred_batch = tr_model.model.forward(x_batch)
                loss = tr_model.compute_model_loss(x_pred_batch, x_batch)
                valid_loss += loss.item()

            avg_train_loss = train_loss / num_train_batches
            avg_valid_loss = valid_loss / num_val_batches
            tr_losses += [avg_train_loss]
            val_losses += [avg_valid_loss]

            # Print epoch-wise loss
            print("Epoch", t, "completed with average training loss:", round(avg_train_loss,2))
            print("Epoch", t, "completed with average validation loss:", round(avg_valid_loss,2))

            loss_dict_tr, loss_dict_val = tr_model.loss_fn.process_batch(num_train_batches, num_val_batches)

            # Check if validation loss improved
            if avg_valid_loss < best_validation_loss:
                best_validation_loss = avg_valid_loss
                best_model_state = tr_model.model.state_dict()
                best_epoch = t

        torch.save(best_model_state, 'best_model_loss_'+str(round(best_validation_loss, 2)))
        tr_model.model.load_state_dict(best_model_state)

        print("Loading model with best loss", best_validation_loss, "found in Epoch", best_epoch)

        plot_losses(np.arange(tr_model.epochs+1), loss_dict_tr, loss_dict_val, tr_model.name+"/train_val_loss.png")

    def trainAll(self):
        for idx in range(len(self.models)):
            self.train(idx)

    def evaluate(self, idx):
        eval_model = self.models[idx]
        eval_model.model.eval()

        latent_cols = ["Latent " + str(x) for x in list(range(eval_model.L))]

        latent_space_train = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtrain())).detach().numpy()
        latent_space_test = eval_model.model.get_latent_space(torch.tensor(eval_model.unroll_Xtest())).detach().numpy()

        print(latent_space_train)

        start = 0.00001
        stop = 0.1
        step = 0.00002
        estimated_alphas = np.arange(start, stop + step, step)

        # we remove warnings when coefficients in Cox PH model are 0
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FitFailedWarning)

        cv = KFold(n_splits=5, shuffle = True, random_state = 46)
        gcv = GridSearchCV(
            make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.95)),
            param_grid = {"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
            cv = cv,
            error_score = 0,
            n_jobs = 4,
        ).fit(latent_space_train, eval_model.unroll_Ytrain())

        cv_results = pd.DataFrame(gcv.cv_results_)

        alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
        mean = cv_results.mean_test_score
        std = cv_results.std_test_score

        print("Alphas: ")
        print(estimated_alphas)
        print("Mean:")
        print(mean)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(alphas, mean)
        ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylabel("concordance index")
        ax.set_xlabel("alpha")
        ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
        ax.axhline(0.5, color="grey", linestyle="--")
        ax.grid(True)
        plt.show()


        best_model = gcv.best_estimator_.named_steps['coxnetsurvivalanalysis']
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
        # link (see which features are most important in cox ph model):
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html

        # link (how to evaluate survival models):
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html


    def evaluateAll(self):
        for idx in range(len(self.models)):
            self.evaluate(idx)




def plot_losses(epochs, data_tr, data_val, dir):
    combined_tr = [sum(values) for values in zip(*[data_tr[key] for key in data_tr.keys()])]
    combined_val = [sum(values) for values in zip(*[data_val[key] for key in data_val.keys()])]

    bestVal = min(combined_val)

    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.plot(epochs, combined_tr, label="Train", marker='o', linestyle='-', color='#1f77b4', linewidth=2,
             alpha=0.8)  # Customize train curve with softer blue color
    plt.plot(epochs, combined_val, label="Val", marker='s', linestyle='-', color='#ff7f0e', linewidth=2, alpha=0.8)
    plt.text(epochs[-1] + 0.15, combined_tr[-1], 'ALL', verticalalignment='center', fontsize=7)
    plt.text(epochs[-1] + 0.15, combined_val[-1], 'ALL', verticalalignment='center', fontsize=7)

    for idx, key in enumerate(list(data_tr.keys())):
        plt.plot(epochs, data_tr[key], linestyle='--', color='#1f77b4', linewidth=2, alpha=0.5)
        plt.text(epochs[-1] + 0.15, data_tr[key][-1], key, verticalalignment='center', fontsize=7, color='#1f77b4')

    for idx, key in enumerate(list(data_val.keys())):
        plt.plot(epochs, data_val[key], linestyle='--', color='#ff7f0e', linewidth=2, alpha=0.5)
        plt.text(epochs[-1] + 0.15, data_val[key][-1], key, verticalalignment='center', fontsize=7, color='#ff7f0e')

    plt.axhline(bestVal, linestyle='-', color='#FF6961', linewidth=2, alpha=1)

    plt.title("Training and Validation Loss Over Time", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(dir)

    plt.show()
