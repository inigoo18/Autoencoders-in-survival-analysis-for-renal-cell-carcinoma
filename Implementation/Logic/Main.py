import numpy as np
import os
import torch
import matplotlib.pyplot as plt

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

def tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R3.csv'))

    d = TabularDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1, BATCH_SIZE, FOLDS) # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL, LossType.VARIATIONAL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    #combinations = [[LossType.DENOISING], [LossType.SPARSE_KL]]
    foldObjects = []

    for comb in combinations:
        print(comb)
        foldObject = FoldObject(comb, FOLDS, d.allDatasets)
        for fold in range(FOLDS):
            title = "TAB{{L_"+str(L) + "_F_" + str(fold) + "_" + '+'.join(str(loss.name) for loss in comb)
            loss_fn = LossHandler(comb, loss_args, None)
            aeModel = MWE_AE(d.input_dim, L)
            vaeModel = VariationalExample(d.input_dim, L)
            if LossType.VARIATIONAL in comb:
                aeModel = vaeModel
            optim = torch.optim.Adam(aeModel.parameters(), lr = 0.0001)
            instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                        aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L, False)#, 'best_model_loss_1478.pth')

            trainer = Trainer(instanceModel)
            bestValLoss = trainer.train()
            meanRes, mseError = trainer.evaluate()
            foldObject.Reconstruction += [bestValLoss]
            foldObject.MSE += [mseError]
            foldObject.ROC += [meanRes]
        foldObjects += [foldObject]
    return foldObjects




def graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS):
    current_directory = os.getcwd()
    somepath = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_graph_R3.pkl'))

    d = GraphDataLoader(somepath, ['PFS_P', 'PFS_P_CNSR'], clinicalVars, 0.2, 0.1,
                          BATCH_SIZE, FOLDS)  # 70% train, 20% test, 10% val

    losses = [LossType.DENOISING, LossType.SPARSE_KL] #LossType.VARIATIONAL

    combinations = [[]]
    combinations.extend([[loss] for loss in losses])

    for i in range(len(losses)):
        for j in range(i + 1, len(losses)):
            combinations.append([losses[i], losses[j]])

    if losses not in combinations:
        combinations += [losses]

    #combinations = [[LossType.VARIATIONAL]]

    foldObjects = []

    for comb in combinations:
        print(comb)
        foldObject = FoldObject(comb, FOLDS, d.allDatasets)
        for fold in range(FOLDS):
            title = "GPH{{L_" + str(L) + "_F_" + str(fold) + "_" + '+'.join(str(loss.name) for loss in comb)
            loss_fn = LossHandler(comb, loss_args, None)
            aeModel = GNNExample(1, d.input_dim, L, BATCH_SIZE)
            vaeModel = GNNVariationalExample(1, d.input_dim, L, BATCH_SIZE)
            if LossType.VARIATIONAL in comb:
                aeModel = vaeModel
            optim = torch.optim.Adam(aeModel.parameters(), lr=0.001)
            instanceModel = TrainingModel(title, d, foldObject.iterations[fold], clinicalVars,
                                          aeModel, loss_fn, optim, EPOCHS, BATCH_SIZE, L,
                                          True)  # , 'best_model_loss_1478.pth')

            trainer = Trainer(instanceModel)
            bestValLoss = trainer.train()
            meanRes, mseError = trainer.evaluate()
            foldObject.Reconstruction += [bestValLoss]
            foldObject.MSE += [mseError]
            foldObject.ROC += [meanRes]
        foldObjects += [foldObject]

    return foldObjects



def visualize_results(names, ys, typename, L, FOLDS):
    means = [np.mean(y[~np.isnan(y)]) for y in np.array(ys)]
    stds = [np.std(y[~np.isnan(y)]) for y in np.array(ys)]

    x = np.arange(len(means))

    # Create a larger figure
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # plot:
    plt.errorbar(x, means, stds, fmt='o', linewidth=2, capsize=6)

    plt.xticks(x, names, fontsize=10, rotation=90)

    # Set y-axis range from 0 to 1
    if typename == 'ROC':
        plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel('Component')
    plt.ylabel('Score')
    plt.title('Scores in ' + typename)
    plt.tight_layout()

    if typename == 'Reconstruction':
        plt.ylabel('Val. loss')

    plt.savefig("Results/" + "Summary_L"+str(L)+ "_F" + str(FOLDS) + "_" +typename + "_" + str(round(np.mean(means),2)) + ".png")
    plt.clf()
    plt.close()



if __name__ == "__main__":
    option = "Tabular"

    torch.manual_seed(42)
    np.random.seed(42)

    L = 256
    loss_args = {'noise_factor': 0.05, 'reg_param': 0.1, 'rho': 0.005}
    clinicalVars = ['MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                    'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA']
    EPOCHS = 80
    FOLDS = 3

    if option == "Tabular":
        BATCH_SIZE = 32
        foldObjects = tabular_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS)
    else:
        BATCH_SIZE = 32
        foldObjects = graph_network(BATCH_SIZE, L, loss_args, clinicalVars, EPOCHS, FOLDS)

    foldNames = [fold.name for fold in foldObjects]
    foldMSE = [fold.MSE for fold in foldObjects]
    foldROC = [fold.ROC for fold in foldObjects]
    foldRec = [fold.Reconstruction for fold in foldObjects]

    visualize_results(foldNames, foldMSE, 'MSE', L, FOLDS)
    visualize_results(foldNames, foldROC, 'ROC', L, FOLDS)
    visualize_results(foldNames, foldRec, 'Reconstruction', L, FOLDS)

    print("FINISHED!!")
