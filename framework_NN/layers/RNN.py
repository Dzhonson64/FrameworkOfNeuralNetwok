# Рекуррентный нейронный слой


from Tensor import Tensor
from activations_func import Sigmoid, Tanh
from layers.Layer import Layer
from layers.Linear import Linear
import numpy as np

#Обычный рекуррентный слой
class RNNCell(Layer):

    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        super().__init__()

        self.n_inputs = n_inputs # кол-во данных во входе
        self.n_hidden = n_hidden # кол-во данных в скрытом слое
        self.n_output = n_output # кол-во данных в выходном слое

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation == Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden) # вес для перобразование из входного слоя в скрытый
        self.w_hh = Linear(n_hidden, n_hidden) # вес для перобразование из скрытого слоя в скрытый
        self.w_ho = Linear(n_hidden, n_output) # вес для перобразование из скрытого слоя в выходной

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)  # преобразование скрытого слоя в скрытый для нового "нейрона"
        combined = self.w_ih.forward(input) + from_prev_hidden  # объединяем обработанный вход и получившийся новый скрытый слов
        new_hidden = self.activation.forward(combined) # создание скрытого слоя из рекуррентного "нейрона" для слебудещго "нейрона"(== память сети)
        output = self.w_ho.forward(new_hidden)  # создание выходных данных из рекуррентного "нейрона"
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


#Обычный рекуррентный слой с ячейкой(нейроном) LSTMCell
class LSTMCell(Layer):

    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()

        self.n_inputs = n_inputs  # кол-во данных во входе
        self.n_hidden = n_hidden  # кол-во данных в скрытом слое
        self.n_output = n_output  # кол-во данных в выходном слое

        self.xf = Linear(n_inputs, n_hidden)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)

        self.hf = Linear(n_hidden, n_hidden, bias=False)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)

        self.w_ho = Linear(n_hidden, n_output, bias=False)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()

        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        prev_hidden = hidden[0] # кратковременная память  сети
        prev_cell = hidden[1] # долгосрочная память  сети

        # определяем какую информацию мы можем забыть и возвращаем результат, как часть того сколько нужно забыть, балгодаря сигмойде [0, 1]
        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()

        # определеяем какую информациюнадо сохранить. Точно также приводим к [0, 1]
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()

        # определеяем какую информациюнадо можно добавить.
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()

        # Заменяем старое состояние ячейки на новоое, забывая (f) и прибаляя (i * g) то, что нам нужнло
        c = (f * prev_cell) + (i * g)

        # Решаем какую долю информации нам вернуть в виде окончательного рещзультата [0, 1]
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()

        # Выводим инофрмацию, с приведением нового сотостояния к диапазону от [-1, 1]
        h = o * c.tanh()

        output = self.w_ho.forward(h)
        return output, (h, c)

    def init_hidden(self, batch_size=1):
        init_hidden = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_cell = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_hidden.data[:, 0] += 1
        init_cell.data[:, 0] += 1
        return (init_hidden, init_cell)
