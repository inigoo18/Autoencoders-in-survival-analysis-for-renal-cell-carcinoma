class CustomDataset():
    """
    This class is used to have all types of data in one place. For example, the entire train set can be housed
    within this class. This way when we need to merge genData and cliData together, it can be done easily, as well
    as checking the labels for later use.
    """
    def __init__(self, genData, cliData, labels):
        self.genData = genData
        self.cliData = cliData
        self.labels = labels

    def __len__(self):
        return len(self.genData)

    def __getitem__(self, idx):
        return self.genData[idx], self.cliData[idx], self.labels[idx]