import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import math

from scipy.stats import f_oneway

from Logic.Autoencoders.GNNExample import GNNExample
from Logic.Autoencoders.GNNVariationalExample import GNNVariationalExample
from Logic.Autoencoders.MinWorkingExample import MWE_AE
from Logic.Autoencoders.VariationalExample import VariationalExample
from Logic.FoldObject import FoldObject
from Logic.GraphDataLoader import GraphDataLoader
from Logic.Losses.LossHandler import LossHandler
from Logic.Losses.LossType import LossType
from Logic.TabularDataLoader import TabularDataLoader
from Logic.Trainer import Trainer
from Logic.TrainingModel import TrainingModel

import xlsxwriter

import matplotlib.patches as mpatches

def tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY):
    '''
        Pipeline for the tabular autoencoder.
        :param BATCH_SIZE: size of the batch for the data loader
        :param L: latent dimensionality
        :param loss_args: dictionary specifying the value of each loss
        :param clinicalVars: which are the clinical variables we need to consider
        :param EPOCHS: number of epochs
        :param FOLDS: number of folds
        :param COHORTS: the different cohorts to filter by in the data loader
        :return: cohortResults, which is a dictionary where for each cohort the results are shown, and the combinations used.
    '''

    # We fetch the preprocessed data
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    # The different penalty combinations we want to evaluate for
    combinations = [[], [LossType.DENOISING], [LossType.SPARSE_KL], [LossType.VARIATIONAL], [LossType.DENOISING, LossType.SPARSE_KL]]
    cohortResults = {}

    # For each cohort (treatment arm)
    for cohort in COHORTS:
        # We load the data through the TabularDataLoader
        d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, (1/FOLDS), 0.2, BATCH_SIZE, FOLDS, cohort)
        # In this array we will put the results that we obtain for each fold
        foldObjects = []
        # For each combination (penalty)...
        for comb in combinations:
            print("Next combination: " + str(comb))
            # We store the datasets we procured in the TabularDataLoader into the FoldObject. This way we don't need
            # to reprocess the data for the next fold, but rather just fetch it from here
            foldObject = FoldObject(comb, FOLDS, d.allDatasets)
            # For each of the folds we have...
            for fold in range(FOLDS):
                # Title to name our result folder under
                title = "TAB{{L_"+str(L) + "_F_" + str(fold) + "_C_" + cohort + "_" + '+'.join(str(loss.name) for loss in comb)
                # LossHandler is a custom class that computes the loss of the autoencoder based on what penalties we have selected
                loss_fn = LossHandler(comb, loss_args, False)
                # Our autoencoder (and the variational counterpart in case we're using the Variational penalty)
                aeModel = MWE_AE(d.input_dim, L)
                vaeModel = VariationalExample(d.input_dim, L)
                if LossType.VARIATIONAL in comb:
                    aeModel = vaeModel
                # Optimizers, and the TrainingModel, which contains all information for the training and evaluation of the model
                optim = torch.optim.Adam(aeModel.parameters(), lr = 0.0005)
                instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                            aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L, False)#, 'best_model_loss_1478.pth')

                trainer = Trainer(instanceModel, WITH_HISTOLOGY)
                # we train the autoencoder
                bestValLoss = trainer.train()
                # we evaluate the model (as well as training the statistical model)
                meanRes, mseError, percentageOverEstimation = trainer.evaluate()
                # hereafter we procure our results
                foldObject.Reconstruction += [bestValLoss]
                foldObject.MSE += [mseError]
                foldObject.ROC += [meanRes]
                foldObject.OverEstimation += [percentageOverEstimation]
                # in case we used too much GPU (and to avoid overflowing errors), we empty the cache
                with torch.no_grad():
                    torch.cuda.empty_cache()
            foldObjects += [foldObject]
        cohortResults[cohort] = foldObjects
    return cohortResults, combinations




def graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY):
    '''
        Pipeline for the graph autoencoder. (please look at tabular implementation for more comments)
        :param BATCH_SIZE: size of the batch for the data loader
        :param L: latent dimensionality
        :param loss_args: dictionary specifying the value of each loss
        :param clinicalVars: which are the clinical variables we need to consider
        :param EPOCHS: number of epochs
        :param FOLDS: number of folds
        :param COHORTS: the different cohorts to filter by in the data loader
        :return: cohortResults, which is a dictionary where for each cohort the results are shown, and the combinations used.
    '''
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_graph_R3.pkl'))

    combinations = [[], [LossType.DENOISING], [LossType.SPARSE_KL], [LossType.VARIATIONAL], [LossType.DENOISING, LossType.SPARSE_KL]]
    cohortResults = {}

    for cohort in COHORTS:
        d = GraphDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, (1/FOLDS), 0.2,
                            BATCH_SIZE, FOLDS, cohort)
        foldObjects = []
        for comb in combinations:
            print("Next combination: " + str(comb))
            foldObject = FoldObject(comb, FOLDS, d.allDatasets)
            for fold in range(FOLDS):
                title = "GPH{{L_"+str(L) + "_F_" + str(fold) + "_C_" + cohort + "_" + '+'.join(str(loss.name) for loss in comb)
                loss_fn = LossHandler(comb, loss_args, True)
                aeModel = GNNExample(1, d.input_dim, L, BATCH_SIZE)
                vaeModel = GNNVariationalExample(1, d.input_dim, L, BATCH_SIZE)
                if LossType.VARIATIONAL in comb:
                    aeModel = vaeModel
                optim = torch.optim.Adam(aeModel.parameters(), lr=0.0005)
                instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                              aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L,
                                              True)  # , 'best_model_loss_1478.pth')

                trainer = Trainer(instanceModel, WITH_HISTOLOGY)
                bestValLoss = trainer.train()
                meanRes, mseError, percentageOverEstimation = trainer.evaluate()
                foldObject.Reconstruction += [bestValLoss]
                foldObject.MSE += [mseError]
                foldObject.ROC += [meanRes]
                foldObject.OverEstimation += [percentageOverEstimation]
                with torch.no_grad():
                    torch.cuda.empty_cache()
            foldObjects += [foldObject]
        cohortResults[cohort] = foldObjects
    return cohortResults, combinations



def visualize_results(names, ys, typename, L, FOLDS, COHORTS):
    '''
    This method creates graphs for the different result metrics, whether its reconstruction, AUC or PFS loss
    :param names: names of the penalties
    :param ys: the results we've obtained
    :param typename: specifies the metric we're measuring
    :param L: latent dimensionality for title
    :param FOLDS: how many folds for title
    :param COHORTS: the cohorts we have used
    '''
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']
    plt.figure(figsize=(10, 8))

    ys = np.array(ys)

    pvalues_cohort = []
    for i in range(len(ys[0])):
        xs = [sublist[i] for sublist in ys]
        pvalues_cohort += [f_oneway(xs[0], xs[1])[1]]

    pvalues_model = []
    for idx_c, cohort in enumerate(ys):
        for i in range(len(ys[0]) - 1):
            pvalues_model += [f_oneway(cohort[0], cohort[i + 1])[1]]

    convert_to_excel(names, ys, typename, L, FOLDS, COHORTS, pvalues_cohort, pvalues_model)

    for c_idx, cohort in enumerate(ys):
        off = 0.1
        if c_idx == 0:
            off = -off
        for i, x in enumerate(cohort):
            plt.boxplot(x[~np.isnan(x)], positions=[i + off], patch_artist=True, boxprops=dict(facecolor=colors[c_idx]))

    legend_patches = [mpatches.Patch(color=colors[i], label=COHORTS[i]) for i in range(len(COHORTS))]
    plt.legend(handles=legend_patches, loc='upper right')  # Adjust legend location

    plt.xticks(np.arange(len(names)), names, fontsize=10, rotation=90)

    # Set y-axis range from 0 to 1
    if typename == 'ROC':
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
        plt.title('Scores for AUC ROC for each treatment arm')
    elif typename == 'MSE':
        plt.ylim(0,12)
        plt.yticks(np.arange(0, 13, 1), fontsize=10)
        plt.title('Predictions (MSE) for each treatment arm')
    elif typename == 'Reconstruction':
        plt.ylim(0,100)
        plt.yticks(np.arange(0, 110, 10), fontsize=10)
        plt.title('Autoencoder reconstruction for each treatment arm')
    else:
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 110, 10), fontsize=10)
        plt.title('Percentage of overestimation of prediction in PFS for each treatment arm')

    # Add labels and title
    plt.xlabel('Component')
    plt.ylabel('Score')
    plt.tight_layout()

    if typename == 'Reconstruction':
        plt.ylabel('Val. loss')

    plt.savefig("Results/" + "Summary_L"+str(L)+ "_F" + str(FOLDS) + "_" +typename + ".png")
    plt.clf()
    plt.close()


def convert_to_excel(names, ys, typename, L, FOLDS, COHORTS, pvalues_cohort, pvalues_model):
    '''
    Method that converts the results to an excel sheet, with additional information such as p-values and the
    results for each fold
    :param names: name of the penalties
    :param ys: the results we've obtained
    :param typename: specifies the metric we're measuring
    :param L: latent dimensionality for title
    :param FOLDS: number of folds for title
    :param COHORTS: the cohorts we have used
    :param pvalues_cohort: the pvalues comparing the difference between cohorts
    :param pvalues_model: the pvalues comparing the difference of the penalties with the non penalty autoencoder.
    :return:
    '''

    workbook = xlsxwriter.Workbook("Results/" + "Excel_L" + str(L) + "_F" + str(FOLDS) + "_" + typename + ".xlsx")
    worksheet = workbook.add_worksheet()

    bold_format = workbook.add_format({'bold': True, 'bg_color': '#DDDDDD'})
    green_format = workbook.add_format({'bold': True, 'bg_color': '#AAFF00'})
    red_format = workbook.add_format({'bold': True, 'bg_color': '#EE4B2B'})

    row = 0
    col = 0

    worksheet.write(row, col, 'Cohort', bold_format)
    col += 1
    worksheet.write(row, col, 'Type', bold_format)
    col += 1
    for c_f in range(FOLDS):
        worksheet.write(row, col, 'Fold ' + str(c_f), bold_format)
        col += 1
    worksheet.write(row, col, 'P-value', bold_format)
    row += 1
    col = 0

    for c_idx, cohort in enumerate(ys):
        worksheet.write(row, col, COHORTS[c_idx], bold_format)
        col += 1
        for i, foldX in enumerate(cohort):
            worksheet.write(row, col, str(names[i]))
            for x in foldX:
                col += 1
                strToInsert = x
                if np.isnan(x):
                    strToInsert = "-"
                worksheet.write(row, col, strToInsert)
            col += 1
            if str(names[i]) == '[]':
                worksheet.write(row, col, '-')
            else:
                pvalue = pvalues_model.pop(0)
                frmt = red_format
                if pvalue <= 0.05:
                    frmt = green_format
                if not math.isnan(pvalue):
                    worksheet.write(row, col, pvalue, frmt)
                else:
                    worksheet.write(row, col, 'N/A')

            col = 1
            row += 1
        col = 0

    col = 2 + FOLDS + 2
    row = 0

    worksheet.write(row, col, 'Type', bold_format)
    col += 1
    worksheet.write(row, col, 'P-value (avelumab + axitinib vs sunitinib)', bold_format)
    row += 1
    col -= 1
    for c_idx, pvalue in enumerate(pvalues_cohort):
        col = 2 + FOLDS + 2
        worksheet.write(row, col, str(names[c_idx]))
        col += 1
        frmt = red_format
        if pvalue <= 0.05:
            frmt = green_format
        if not math.isnan(pvalue):
            worksheet.write(row, col, pvalue, frmt)
        else:
            worksheet.write(row, col, 'N/A')
        row += 1

    workbook.close()



if __name__ == "__main__":
    # Option specifies either 'Tabular' or 'Graph'.
    # WITH_HISTOLOGY can be set to true if we want to consider all of the histology features.
    option = "Tabular"
    WITH_HISTOLOGY = False

    # We set the randomness to 42 for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Latent dimensionality, penalty hyperparameters, and the clinical (+ histology) features to consider
    L = 64
    loss_args = {'noise_factor': 0.001, 'reg_param': 0.10, 'rho': 0.001}
    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']
    # Number of epochs, folds, and which arms (here named COHORTS) we want to measure. Cohorts can be set to ['ALL'] in case
    # we do not want to differentiate between arms
    EPOCHS = 100
    FOLDS = 10
    COHORTS = ['Avelumab+Axitinib','Sunitinib']

    if WITH_HISTOLOGY is False:
        clinicalVars = ['HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA']

    # Batch size and initialization of our work
    if option == "Tabular":
        BATCH_SIZE = 16
        foldObjects, combinations = tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY)
    else:
        BATCH_SIZE = 16
        foldObjects, combinations = graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY)

    # We grab the specific metrics out of the 'FoldObject' object, which holds the different metrics
    # obtained for each fold.
    namedCombs = [[str(x) for x in y] for y in combinations]
    foldMSE = [[fold.MSE for fold in foldObjects[cohort]] for cohort in COHORTS]
    foldROC = [[fold.ROC for fold in foldObjects[cohort]] for cohort in COHORTS]
    foldRec = [[fold.Reconstruction for fold in foldObjects[cohort]] for cohort in COHORTS]
    foldOverEstimation = [[fold.OverEstimation for fold in foldObjects[cohort]] for cohort in COHORTS]

    # We use the visualize_results method to obtain figures + excel sheets of the result.
    visualize_results(namedCombs, foldMSE, 'MSE', L, FOLDS, COHORTS)
    visualize_results(namedCombs, foldROC, 'ROC', L, FOLDS, COHORTS)
    visualize_results(namedCombs, foldRec, 'Reconstruction', L, FOLDS, COHORTS)
    visualize_results(namedCombs, foldOverEstimation, 'OverEstimation', L, FOLDS, COHORTS)
