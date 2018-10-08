class SGD:
    def __init__(self, learnRate):
        self.learnRate = learnRate

    def string(self):
        return "SGD\nlearnRate=%.5f"%(self.learnRate,)

