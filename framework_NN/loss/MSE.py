class MSELoss():

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        dif = input - target
        return (dif * dif).sum(0)