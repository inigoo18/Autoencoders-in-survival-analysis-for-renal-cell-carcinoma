class FoldObject():
    def __init__(self, name, folds, iterations):
        self.folds = folds
        self.iterations = iterations
        self.MSE = []
        self.ROC = []
        self.name = [str(x) for x in name] # convert enum representations to strings