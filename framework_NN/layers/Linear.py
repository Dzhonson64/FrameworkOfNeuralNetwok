# Линейный слой

import numpy as np
from Tensor import Tensor
from layers.Layer import Layer

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()

        self.use_bias = bias # байес (сдвиг)

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs)) # случайные числа по Гауссовскому распределению
        self.weight = Tensor(W, autograd=True) # создание весов в слое


        if self.use_bias:
            # создание баеса в слое
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)

        if self.use_bias:
            self.parameters.append(self.bias)

    # Выполнение выявления признаков слоем
    def forward(self, input):
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)