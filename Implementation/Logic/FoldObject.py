class FoldObject():
    def __init__(self, name, folds, iterations):
        '''
                This class holds the different KFold datasets as well as the containers for the metrics once the
                training process is done.
                :param name: name of the different penalties
                :param folds: how many folds
                :param iterations: iteration object list
        '''
        self.folds = folds
        self.iterations = iterations
        self.MSE = []
        self.ROC = []
        self.Reconstruction = []
        self.OverEstimation = []
        self.name = [str(x) for x in name] # convert enum representations to strings