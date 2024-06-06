import torch

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, as_concordance_index_ipcw_scorer
from torch.optim.lr_scheduler import StepLR

from Logic.CustomKFoldScikit import CustomKFold
from Logic.TrainingModel import TrainingModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec

import seaborn as sns

from sklearn.manifold import TSNE
import xlsxwriter

import copy


class Trainer:
    """
    Takes a model and trains it using the parameters that TrainingModel holds. After the training, the model
    is evaluated in this class as well.
    """


    def __init__(self, model : TrainingModel, WITH_HISTOLOGY : bool):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.WITH_HISTOLOGY = WITH_HISTOLOGY

    def train(self):
        '''
        This method trains a model.
        :return: Returns the best validation loss
        '''
        tr_model = self.model
        # we move the model to the GPU, if available
        tr_model.model.to(self.device)
        best_validation_loss = float('inf')
        best_model_state = None
        best_epoch = -1
        # LR regularizer
        scheduler = StepLR(tr_model.optim, step_size=tr_model.epochs // 4, gamma=0.5)

        for t in range(tr_model.epochs + 1):
            tr_model.model.train()
            train_loss = 0.0
            for b in tr_model.train_loader:
                x_batch = b[0]  # we keep the first attribute from CustomDataset (data)

                # In case we need to introduce noise to the training data
                x_batch = tr_model.loss_fn.initialize_loss(x_batch, tr_model.GNN)

                x_pred_batch = None
                mu = None
                log_var = None
                # Perform forward pass, if variational then we use a different set-up (mu + log var)
                if (tr_model.variational):
                    x_pred_batch, mu, log_var = tr_model.model.forward(x_batch)
                else:
                    x_pred_batch = tr_model.model.forward(x_batch)

                # Compute loss for the entire batch
                loss = tr_model.compute_model_loss(x_batch, x_pred_batch, mu, log_var)

                # Accumulate loss for the epoch
                train_loss += loss.item()

                # Zero gradients, backward pass, and update parameters
                tr_model.optim.zero_grad()
                loss.backward()
                tr_model.optim.step()

            valid_loss = 0.0
            tr_model.model.eval()
            # We repeat the process with the validation set
            with torch.no_grad():
                for b in tr_model.val_loader:
                    x_batch = b[0]
                    x_pred_batch = None
                    mu = None
                    log_var = None
                    # Perform forward pass, if variational then different set-up
                    if (tr_model.variational):
                        x_pred_batch, mu, log_var = tr_model.model.forward(x_batch)
                    else:
                        x_pred_batch = tr_model.model.forward(x_batch)

                    # Compute loss for the entire batch
                    loss = tr_model.compute_model_loss(x_batch, x_pred_batch, mu, log_var)
                    valid_loss += loss.item()

            # call scheduler (+optimizer) after training and validation epoch
            scheduler.step()

            # Print epoch-wise loss
            loss_dict_tr, loss_dict_val = tr_model.loss_fn.process_batch(len(tr_model.train_loader),
                                                                         len(tr_model.val_loader))

            avg_valid_loss = round(loss_dict_val['MSE'][-1], 2)
            print("Epoch", t, "completed with average validation loss:", avg_valid_loss)

            # Check if validation loss improved
            if avg_valid_loss < best_validation_loss:
                best_validation_loss = avg_valid_loss
                best_model_state = copy.deepcopy(tr_model.model.state_dict())
                best_epoch = t
                print("New checkpoint!")

        torch.save(best_model_state,
                   'Checkpoints/model_loss' + str(tr_model.name) + "_" + str(round(best_validation_loss)) + '.pth')
        tr_model.model.load_state_dict(best_model_state)

        print("Loading model with best loss", best_validation_loss, "found in Epoch", best_epoch)

        plot_losses(np.arange(tr_model.epochs + 1), loss_dict_tr, loss_dict_val, tr_model.name + "/train_val_loss.png")

        return best_validation_loss



    def evaluate(self):
        '''
        This method evaluates the model by obtaining the model's latent representation
        :return:
        '''
        eval_model = self.model
        eval_model.model.eval()

        latent_cols = ["Latent " + str(x) for x in list(range(eval_model.L))]
        latent_cols += eval_model.cli_vars
        latent_idxs = np.arange(eval_model.L + len(eval_model.cli_vars))

        # Unroll batch lets us obtain the whole loader's data
        latent_space_train = eval_model.model.get_latent_space(eval_model.data_loader.unroll_batch(eval_model.train_loader, dim = 0)).cpu().detach().numpy()
        latent_space_test = eval_model.model.get_latent_space(eval_model.data_loader.unroll_batch(eval_model.test_loader, dim=0)).cpu().detach().numpy()
        latent_space_val = eval_model.model.get_latent_space(eval_model.data_loader.unroll_batch(eval_model.val_loader, dim=0)).cpu().detach().numpy()

        # We add clinical variables
        latent_space_train = np.concatenate((latent_space_train, eval_model.data_loader.unroll_batch(eval_model.train_loader, dim=1).cpu().detach().numpy()), axis = 1)
        latent_space_test = np.concatenate((latent_space_test, eval_model.data_loader.unroll_batch(eval_model.test_loader, dim=1).cpu().detach().numpy()), axis = 1)
        latent_space_val = np.concatenate((latent_space_val, eval_model.data_loader.unroll_batch(eval_model.val_loader, dim=1).cpu().detach().numpy()), axis=1)

        # We concatenate train + val since the statistical models do not use validation sets
        latent_space_train = np.concatenate((latent_space_train, latent_space_val), axis=0)

        # We do the same thing with the labels
        yTrain = eval_model.data_loader.unroll_batch(eval_model.train_loader, dim=2).cpu().detach().numpy()
        yTrain = np.array([(bool(event), float(time)) for event, time in yTrain], dtype=[('event', bool), ('time', float)])
        yTest = eval_model.data_loader.unroll_batch(eval_model.test_loader, dim=2).cpu().detach().numpy()
        yTest = np.array([(bool(event), float(time)) for event, time in yTest], dtype=[('event', bool), ('time', float)])
        yVal = eval_model.data_loader.unroll_batch(eval_model.val_loader, dim=2).cpu().detach().numpy()
        yVal = np.array([(bool(event), float(time)) for event, time in yVal], dtype=[('event', bool), ('time', float)])

        # We concatenate labels in training set with validation set
        yTrain = np.concatenate((yTrain, yVal), axis=0)

        # We "draw" the latent spaces in excel sheets for debugging purposes
        draw_latent_space('LatentTrain', eval_model.name, latent_space_train)
        draw_latent_space('LatentTest', eval_model.name, latent_space_test)

        demographic_DF = pd.DataFrame()
        demographic_DF['PFS_P'] = yTest['time']

        # Parameters we set to run COX.
        # The number of TRIES is how many times we are going to perform the grid search. It can fail because the
        # search is using parameters that are too big, in that case we retry and multiply the start and stop parameters
        # by offset. We search using a step defined in the 'step' variable.
        TRIES = 3
        non_zero = 0
        offset = 0.9

        start = 0.0001
        stop = 0.01
        step = 0.0002

        OK = False
        while not OK:
            OK = True
            estimated_alphas = np.arange(start, stop + step, step)

            # we remove warnings when coefficients in Cox PH model are 0
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FitFailedWarning)
            warnings.simplefilter("ignore", ArithmeticError)

            # we scale for better performance.
            scaler = StandardScaler()
            scaler.fit(latent_space_train)
            scaled_latent_space_train = scaler.transform(latent_space_train)
            scaled_latent_space_test = scaler.transform(latent_space_test)

            # We perform grid search to find the best alpha to test with
            cv = CustomKFold(n_splits=7, shuffle=True, random_state=40)
            gcv = GridSearchCV(
                as_concordance_index_ipcw_scorer(
                    CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True, max_iter=80000, normalize=False)),
                param_grid={"estimator__alphas": [[v] for v in estimated_alphas]},
                cv=cv,
                error_score=0,
                n_jobs=5,
            ).fit(scaled_latent_space_train, yTrain)

            cv_results = pd.DataFrame(gcv.cv_results_)

            alphas = cv_results.param_estimator__alphas.map(lambda x: x[0])
            mean = cv_results.mean_test_score
            std = cv_results.std_test_score

            best_model = gcv.best_estimator_.estimator
            best_coefs = pd.DataFrame(best_model.coef_, index=latent_cols, columns=["coefficient"])
            best_alpha = gcv.best_params_["estimator__alphas"][0]

            plot_cindex(alphas, mean, std, best_alpha, eval_model.name + "/c-index")

            non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
            print(f"Number of non-zero coefficients: {non_zero}")

            survival_functions_tmp = best_model.predict_survival_function(scaled_latent_space_test, best_alpha)

            # if the coefs are all zero OR if for some reason the survival functions couldn't be estimated, try again
            if non_zero == 0 or np.isnan(survival_functions_tmp[0].y).any():
                OK = False
                TRIES -= 1
                start *= offset
                step *= offset
                stop *= offset
                print("All coefficients are 0 or survival functions are undefined... Tries left: " + str(
                    TRIES) + " with start: " + str(start))
                if TRIES == 0:
                    return np.nan, np.nan

        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        # we plot a figure showing how much weight each coef has
        plot_coefs(non_zero_coefs, coef_order, eval_model.name + "/relevant_features")

        latent_data = zip(latent_cols, latent_idxs)
        idxs_interest = []
        cols_interest = list(coef_order)

        for col, idx in latent_data:
            if col in list(coef_order):
                idxs_interest += [idx]

        data_points = latent_space_train[:, idxs_interest]

        # if we have more than 2 coefs, we perform  tsne. However this isn't uesful in our work.
        if non_zero >= 2:
            plot_tsne_coefs(data_points, cols_interest, eval_model.name + "/tsne")

        # We keep the best 5 coefficients for the correlation plots (check method for more info)
        latent_data = zip(latent_cols, latent_idxs)
        best_coefs = coef_order[-5:]
        best_indices = [idx for col, idx in latent_data if col in best_coefs]

        print("Best index is", best_indices)
        data_points_best_coef = np.array(latent_space_train)[:, best_indices]

        # we concatenate training and validation together
        original_data = np.concatenate((eval_model.data_loader.get_transcriptomic_data(eval_model.train_loader),
                                        eval_model.data_loader.get_transcriptomic_data(eval_model.val_loader)), axis = 0)

        # we plot correlation coefs using mutual information
        plot_correlation_coefs(original_data,
                         data_points_best_coef,
                         eval_model.name + "/correlation_best_features", best_indices, eval_model.test_genes)

        # Predict using the best model and the test latent space
        cph_risk_scores = best_model.predict(scaled_latent_space_test, alpha = best_alpha)

        times = yTest['time']

        va_times = np.arange(min(times), max(times), 0.5)
        cph_auc, _ = cumulative_dynamic_auc(yTrain, yTest, cph_risk_scores, va_times)

        # we plot the Area under ROC
        meanRes = plot_auc(va_times, cph_auc, eval_model.name + "/ROC")

        # Using survival functions, obtain median OR mean and assign it to each patient.
        survival_functions = best_model.predict_survival_function(scaled_latent_space_test, best_alpha)
        predicted_times = []


        # we can either use mean or median to predict PFS with.
        mode = "Mean"
        print("Using prediction mode: ", mode)

        if mode == "Mean":
            for g in range(len(survival_functions)):
                mean_value = np.trapz(survival_functions[g].y, survival_functions[g].x) # area under survival function
                predicted_times += [mean_value]
        elif mode == "Median":
            for g in range(len(survival_functions)):
                median_value = np.interp(0.5, survival_functions[g].y[::-1], survival_functions[g].x[::-1])
                predicted_times += [median_value]

        demographic_DF['predicted_PFS'] = predicted_times

        # we get the overall plot that shows how the model performed with the predictions
        mseError = evaluate_demographic_data(eval_model, survival_functions, demographic_DF)
        # we obtain the percentage of overestimation in our PFS predictions
        percentageOverEstimation = (demographic_DF['predicted_PFS'] > demographic_DF['PFS_P']).mean() * 100

        return meanRes, mseError, percentageOverEstimation




def plot_tsne_coefs(data, names, dir):
    '''
    In this method we first apply KMeans to the output of the autoencoder, with two clusters, then we apply dimensionality
    reduction with t-SNE to reduce the dimensions down to 2. While this is not used in the report, it might add in
    some visualization of the latent embeddings of the autoencoder.
    '''
    print("Applying tSNE on data with following variables:")
    for i in names:
        print(i)

    kmeans = KMeans(n_clusters = 2, random_state = 42)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_

    x_embedded = TSNE(n_components=2, perplexity=2).fit_transform(np.array(data))

    centroids_manual = []
    for label in np.unique(cluster_labels):
        cluster_points = x_embedded[cluster_labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids_manual.append(centroid)
    centroids_manual = np.array(centroids_manual)

    plt.figure(figsize=(8, 6))

    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=cluster_labels, cmap = 'viridis', alpha=0.7, label='Data Points')
    plt.scatter(centroids_manual[:, 0], centroids_manual[:, 1], c='red', marker='*', s=150,
                label='Centroids')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization with KMeans clustering', fontsize=14)
    # plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Results/"+dir)
    plt.clf()
    plt.close()



def plot_losses(epochs, data_tr, data_val, dir):
    '''
    We plot the loss evolution through the epochs of the training and validation data.
    We add in other losses such as SPARSE or VARIATIONAL in order to see their effect on the total loss.
    '''
    sns.set_style("whitegrid")

    OFFSET = 5

    combined_tr = [sum(values) for values in zip(*[data_tr[key] for key in data_tr.keys()])]
    combined_val = data_val['MSE']

    epochs = epochs[OFFSET:]
    combined_tr = combined_tr[OFFSET:]
    combined_val = combined_val[OFFSET:]

    bestVal = min(combined_val)

    offsetY = max(max(combined_tr), max(combined_val)) * 0.05
    maxY = max(max(combined_tr), max(combined_val)) + offsetY
    minY = 0

    plt.figure(figsize=(10, 10))  # Set the figure size

    # Train losses with different types of losses
    plt.subplot(2, 1, 1)

    plt.plot(epochs, combined_tr, label="Train", marker='o', linestyle='-', color='#1f77b4', linewidth=2,
             alpha=1)

    shades_blue = ['#6abf9e', '#1f77b4', '#2ca02c', '#002c5a']
    for idx, key in enumerate(list(data_tr.keys())):
        color = shades_blue[idx]
        plt.plot(epochs, data_tr[key][OFFSET:], linestyle='--', color=color, label=key, linewidth=2, alpha=0.8)

    plt.ylim(minY, maxY)
    plt.title("Training (+ components) Loss Over Time", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Validation losses with only MSE (the reconstruction loss)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, combined_tr, label="Train", marker='o', linestyle='-', color='#1f77b4', linewidth=2,
             alpha=.6)  # Customize train curve with softer blue color
    plt.plot(epochs, combined_val, label="Val", marker='o', linestyle='-', color='#ff7f0e', linewidth=2, alpha=1)
    plt.axhline(bestVal, linestyle='-',color='#FF6961', linewidth=2, alpha=.5)
    plt.ylim(minY, maxY)

    plt.title("Training and Validation Loss Over Time", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("Results/"+dir)
    plt.clf()
    plt.close()

def plot_cindex(alphas, mean, std, best_alpha, dir):
    '''
    We plot the C-index throughout different alpha values, and show the best with a vertical line.
    The horizontal line at 0.5 represents what a score befitting a random model would be like
    '''
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index IPCW")
    ax.set_xlabel("alpha")
    ax.axvline(best_alpha, c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.title("C-Index of the COX PH model using Train data")
    plt.savefig("Results/"+dir)
    plt.clf()
    plt.close()

def plot_coefs(non_zero_coefs, coef_order, dir):
    '''
    We plot the coefficients that we obtained from the COX PH model
    '''
    _, ax = plt.subplots(figsize=(6, 8))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.savefig("Results/"+dir)
    plt.clf()
    plt.close()
    # plt.show()

def plot_auc(va_times, cph_auc, dir):
    '''
    We plot the Area under ROC throughout the different trimesters. We also show the average of the different scores
    obtained throughout the trimesters
    '''
    meanRes = np.mean(cph_auc[~np.isnan(cph_auc)])
    plt.plot(va_times, cph_auc, marker="o")
    plt.axhline(meanRes, linestyle="--")

    plt.xlabel("trimesters from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.title("Area under Curve using best COX PH model with test data")
    plt.grid(True)
    plt.savefig("Results/"+dir)
    plt.clf()
    plt.close()
    return meanRes


def plot_correlation_coefs(oriX, predX, dir, latentIdxs ,geneNames):
    '''
    We plot the correlation using Mutual Information between the original (transcriptomic) data and the latent
    representations. This tells us which genes are most represented by which latent features.
    '''

    mutual_informations = []
    for idx in range(len(latentIdxs)):
        mutual_informations += [mutual_info_regression(oriX, predX[:, idx])]

    resList = []

    for idx, lat_idx in enumerate(latentIdxs):
        sorted_indices = np.argsort(mutual_informations[idx])
        # Get the indices of the top 5 values
        top_indices = sorted_indices[-5:]
        # Get the top 5 values
        top_values = mutual_informations[idx][top_indices]
        genes = [geneNames[i] for i in top_indices]
        resList += [(genes, top_values)]


    fig = plt.figure(figsize=(12, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.5, hspace=0.5)

    # AX 0
    ax = fig.add_subplot(outer[0])
    im = ax.imshow(mutual_informations, cmap='coolwarm', aspect=100 * predX.shape[1])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(latentIdxs)
    fig.colorbar(im, ax=ax)

    # AX 1
    inner_right = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[1], hspace=1)

    for i in range(5):
        ax = plt.Subplot(fig, inner_right[i])
        fig.add_subplot(ax)
        ypos = np.arange(len(resList[i][0]))
        ax.barh(ypos, resList[i][1], align='center')
        ax.set_yticks(ypos)
        ax.set_yticklabels(resList[i][0])
        ax.set_xlabel('Mutual information')
        ax.set_title('Latent ' + str(latentIdxs[i]), fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("Results/" + dir)
    plt.clf()
    plt.close()




def evaluate_demographic_data(eval_model, survival_functions, demographic_df):
    '''
    This method shows (i) the survival functions (ii) boxplots with the distribution for the original and predicted
    PFS values (iii) the residuals in the predictions (iv) the actual vs the predicted PFS
    '''

    # Calculate MSE
    mse = mean_squared_error(demographic_df['PFS_P'], demographic_df['predicted_PFS'])

    # Set Seaborn style
    sns.set_style("whitegrid")

    plt.figure(figsize=(16, 12))

    # AX 0, 0 :: Survival function
    for idx, g in enumerate(survival_functions):
        color = plt.cm.prism(np.random.rand())
        plt.subplot(2, 2, 1)
        plt.plot(g.x, g.y, color=color, alpha=0.3)

        median_value = np.interp(0.5, g.y[::-1], g.x[::-1])
        plt.plot(demographic_df.iloc[idx]['predicted_PFS'], 0.5, 'x', color=color, alpha=0.5, markersize=10)
        plt.title('Survival Function')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')

    # AX 0, 1 :: Box plots
    plt.subplot(2, 2, 2)
    plt.boxplot([demographic_df['PFS_P'], demographic_df['predicted_PFS']], labels=['y', r'$\hat{y}$'])
    plt.title('Box Plot')
    plt.ylabel('Time')

    # AX 1, 0 :: Residuals
    residuals = demographic_df['PFS_P'] - demographic_df['predicted_PFS']
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, bins=20, color='skyblue', alpha=0.7, kde=True)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # AX 1, 1 :: Actual vs predicted
    indices = np.arange(len(demographic_df['PFS_P']))
    bar_width = 0.4
    plt.subplot(2, 2, 4)
    plt.bar(indices, demographic_df['PFS_P'], bar_width, color='red', edgecolor = 'red', label='Actual', alpha=0.7)
    plt.bar(indices + bar_width + 0.1, demographic_df['predicted_PFS'], bar_width, color='blue', edgecolor = 'blue', label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted')
    plt.xlabel('Patient')
    plt.ylabel('Time')
    plt.legend()

    plt.suptitle("MSE = " + str(round(mse, 2)), fontsize=16, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig("Results/"+eval_model.name + "/prediction")
    plt.clf()
    plt.close()

    return mse


def draw_latent_space(filename, folder, latent_space):
    '''
    We draw the latent space in an excel sheet for debugging purposes.
    '''
    row = 1
    col = 1

    workbook = xlsxwriter.Workbook('Results/' + folder + '/' + filename + '.xlsx')
    worksheet = workbook.add_worksheet()

    bold_format = workbook.add_format({'bold': True, 'bg_color': '#DDDDDD'})

    for i in range(latent_space.shape[1]):
        worksheet.write(0, i + 1, 'Lat. ' + str(i), bold_format)

    for i in range(latent_space.shape[0]):
        worksheet.write(i + 1, 0, 'Pat. ' + str(i), bold_format)

    for xs in latent_space:
        for x in xs:
            red_component = max(int(x * 255), 30)
            blue_component = max(int((1 - x) * 255), 30)
            purple_hex = '#{:02X}30{:02X}'.format(red_component, blue_component)
            purple_format = workbook.add_format({'bg_color': purple_hex, 'font_color': 'white'})
            worksheet.write(row, col, round(x.item(), 2), purple_format)
            col += 1
        row += 1
        col = 1
    workbook.close()