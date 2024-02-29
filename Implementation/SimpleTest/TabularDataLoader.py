import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TabularDataLoader:
    """
    Class that holds the DataFrame where the tabular data is located.
    """

    def __init__(self, file_path, pred_var, trte_ratio):
        dataframe = pd.read_csv(file_path, sep = ',', index_col = 0)
        self.dataframe = dataframe.astype('float32')
        X = self.dataframe.drop(columns=[pred_var])
        y = self.dataframe[pred_var]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size = trte_ratio, random_state = 42)
        self._normalize_data()

    def describe_dataframe(self):
        return self.dataframe.describe()

    def input_dim_train(self):
        return len(self.X_train.iloc[0])

    def input_dim_test(self):
        return len(self.Y_train.iloc[0])

    def _normalize_data(self):
        '''
        Normalizes train and test data
        '''
        scaler = MinMaxScaler()

        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns,
                                 index=self.X_train.index)

        self.X_test = pd.DataFrame(scaler.fit_transform(self.X_test), columns=self.X_test.columns,
                                    index=self.X_test.index)
