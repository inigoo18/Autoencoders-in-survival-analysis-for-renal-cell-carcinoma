import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Class that holds the DataFrame where the data is located.
    WIP, but it should load the csv, and have some auxiliary functions to have everything ready, such as train/test split,
    normalized and original version, etc. Just some class to keep everything tidy.
    """

    def __init__(self, file_path, pred_var, trte_ratio):
        self.dataframe = pd.read_csv(file_path)
        X = self.dataframe.drop(columns=[pred_var])
        y = self.dataframe[pred_var]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size = trte_ratio, random_state = 42)
        self.X = self.dataframe
        self.Y = self.dataframe[pred_var]

    def describe_dataframe(self):
        return self.dataframe.describe()