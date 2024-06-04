from sklearn.model_selection import StratifiedKFold
import numpy as np

class CustomKFold(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):

        filtered_data = y[y['event']]

        # Get the index of the sample with the maximum Survival_in_days among the filtered data
        index_of_max_survival = np.argmax(filtered_data['time'])

        # Get the index of the original array corresponding to the index in the filtered data
        original_max_index = np.where(y['event'])[0][index_of_max_survival]

        for train_index, test_index in super().split(X, y, y['event']):
            # Ensure m is always in the training set
            # if X[train_index].max() < X[test_index].max():
            #    train_index = list(train_index) + [test_index[X[test_index].argmax()]]
            if original_max_index in test_index:
                test_index = np.delete(test_index, np.where(test_index == original_max_index))
                train_index = np.append(train_index, original_max_index)
            yield train_index, test_index

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits