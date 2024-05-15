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
        Pipeline for the graph autoencoder. (look at tabular implementation for more comments)
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
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))
    losses = [LossType.DENOISING, LossType.SPARSE_KL]#, LossType.VARIATIONAL]

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    combinations = [[], [LossType.DENOISING], [LossType.SPARSE_KL], [LossType.VARIATIONAL], [LossType.DENOISING, LossType.SPARSE_KL]]
    #combinations = [[], [LossType.DENOISING], [LossType.SPARSE_KL], [LossType.DENOISING, LossType.SPARSE_KL]]
    cohortResults = {}


    for cohort in COHORTS:
        d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1, BATCH_SIZE, FOLDS, cohort)  # 60% train, 25% test, 15% val
        foldObjects = []
        for comb in combinations:
            print(comb)
            foldObject = FoldObject(comb, FOLDS, d.allDatasets)
            for fold in range(FOLDS):
                title = "TAB{{L_"+str(L) + "_F_" + str(fold) + "_C_" + cohort + "_" + '+'.join(str(loss.name) for loss in comb)
                loss_fn = LossHandler(comb, loss_args)
                aeModel = MWE_AE(d.input_dim, L)
                vaeModel = VariationalExample(d.input_dim, L)
                if LossType.VARIATIONAL in comb:
                    aeModel = vaeModel
                optim = torch.optim.Adam(aeModel.parameters(), lr = 0.0001)
                instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                            aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L, False)#, 'best_model_loss_1478.pth')

                trainer = Trainer(instanceModel, WITH_HISTOLOGY)
                bestValLoss = trainer.train()
                meanRes, mseError = trainer.evaluate()
                foldObject.Reconstruction += [bestValLoss]
                foldObject.MSE += [mseError]
                foldObject.ROC += [meanRes]
                with torch.no_grad():
                    torch.cuda.empty_cache()
            foldObjects += [foldObject]
        cohortResults[cohort] = foldObjects
    return cohortResults, combinations




def graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY):
    '''
        Pipeline for the graph autoencoder. (look at tabular implementation for more comments)
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

    losses = [LossType.DENOISING, LossType.SPARSE_KL, LossType.VARIATIONAL]

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    combinations = [[], [LossType.DENOISING], [LossType.SPARSE_KL], [LossType.VARIATIONAL], [LossType.DENOISING, LossType.SPARSE_KL]]#[[LossType.VARIATIONAL]]
    cohortResults = {}

    for cohort in COHORTS:
        d = GraphDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1,
                            BATCH_SIZE, FOLDS, cohort)  # 70% train, 20% test, 10% val
        foldObjects = []
        for comb in combinations:
            print(comb)
            foldObject = FoldObject(comb, FOLDS, d.allDatasets)
            for fold in range(FOLDS):
                title = "GPH{{L_"+str(L) + "_F_" + str(fold) + "_C_" + cohort + "_" + '+'.join(str(loss.name) for loss in comb)
                loss_fn = LossHandler(comb, loss_args, d.adjacency_matrix)
                aeModel = GNNExample(1, d.input_dim, L, BATCH_SIZE)
                vaeModel = GNNVariationalExample(1, d.input_dim, L, BATCH_SIZE)
                if LossType.VARIATIONAL in comb:
                    aeModel = vaeModel
                optim = torch.optim.Adam(aeModel.parameters(), lr=0.0001)
                instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                              aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L,
                                              True)  # , 'best_model_loss_1478.pth')

                trainer = Trainer(instanceModel, WITH_HISTOLOGY)
                bestValLoss = trainer.train()
                meanRes, mseError = trainer.evaluate()
                foldObject.Reconstruction += [bestValLoss]
                foldObject.MSE += [mseError]
                foldObject.ROC += [meanRes]
                with torch.no_grad():
                    torch.cuda.empty_cache()
            foldObjects += [foldObject]
        cohortResults[cohort] = foldObjects
    return cohortResults, combinations



def visualize_results(names, ys, typename, L, FOLDS, COHORTS):
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
        plt.ylim(0,15)
        plt.yticks(np.arange(0, 16, 1), fontsize=10)
        plt.title('Predictions (MSE) for each treatment arm')
    else:
        plt.ylim(0,200)
        plt.yticks(np.arange(0, 220, 20), fontsize=10)
        plt.title('Autoencoder reconstruction for each treatment arm')

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

    print("EXCEL")
    print(ys)
    print(names)
    print(pvalues_cohort)
    print(pvalues_model)

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
    option = "Tabular"
    WITH_HISTOLOGY = False

    torch.manual_seed(42)
    np.random.seed(42)

    L = 64
    loss_args = {'noise_factor': 0.001, 'reg_param': 0.15, 'rho': 0.001}
    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']
    EPOCHS = 15
    FOLDS = 10
    COHORTS = ['Avelumab+Axitinib','Sunitinib']
    #COHORTS = ['ALL']

    if WITH_HISTOLOGY is False:
        #clinicalVars = ['HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA']
        clinicalVars = []

    if option == "Tabular":
        BATCH_SIZE = 16
        foldObjects, combinations = tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY)
    else:
        BATCH_SIZE = 16
        foldObjects, combinations = graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS, COHORTS, WITH_HISTOLOGY)

    namedCombs = [[str(x) for x in y] for y in combinations]
    foldMSE = [[fold.MSE for fold in foldObjects[cohort]] for cohort in COHORTS]
    foldROC = [[fold.ROC for fold in foldObjects[cohort]] for cohort in COHORTS]
    foldRec = [[fold.Reconstruction for fold in foldObjects[cohort]] for cohort in COHORTS]

    visualize_results(namedCombs, foldMSE, 'MSE', L, FOLDS, COHORTS)
    visualize_results(namedCombs, foldROC, 'ROC', L, FOLDS, COHORTS)
    visualize_results(namedCombs, foldRec, 'Reconstruction', L, FOLDS, COHORTS)

    print("FINISHED!!")
