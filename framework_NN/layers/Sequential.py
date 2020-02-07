# Слой для последовательно хранения других слоёв

from layers import Layer

class Sequential(Layer):

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = list()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    # Выполнение выявления признаков всех слоёв
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    # Возвращем все глобальные параметры сети
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params