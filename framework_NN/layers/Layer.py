# Базовый слой, предоставляющий возможно хранить/возвращать гиперпараметры модели

class Layer:

    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters