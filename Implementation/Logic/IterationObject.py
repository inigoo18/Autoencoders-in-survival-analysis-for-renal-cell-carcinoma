class IterationObject():
    def __init__(self, train, test, val, test_genes):
        '''
                This class is used in the FoldObject class to store the different datasets for each fold
                :param train: train dataset
                :param test: test dataset
                :param val: validation dataset
        '''
        self.train_data = train
        self.test_data = test
        self.val_data = val
        self.test_genes = test_genes
