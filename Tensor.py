import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)  # данные
        self.creators = creators  # тензоры, которые образовали данный тензор
        self.creation_op = creation_op  # операции тензовров, которые образовали данный тензор
        self.autograd = autograd  # включение/отключение автоградиента при обратном распространении
        if id is None:
            # id не присвоен
            id = np.random.randint(0, 100000)
        self.id = id
        self.grad = None
        self.children = {}
        self.index_select_indices = None

        if creators is not None:
            # заполнение словаря операторами, которые образовали данный тензор
            for creator in creators:
                if self.id not in creator.children:
                    # данный тензор впервые образован каким-либо операндом
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    # Перезругзка операции сложения (other - второй операнд)
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add"
                          )
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * (-1),
                          autograd=True,
                          creators=[self],
                          creation_op="neg",
                          )
        return Tensor(self.data * (-1))

    # Перезругзка операции вычитания (other - второй операнд)
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub"
                          )
        return Tensor(self.data - other.data)

    # Перезругзка операции умножения (other - второй операнд)
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul"
                          )
        return Tensor(self.data * other.data)

    # Операция суммирования элементов матрицы (dim - размерность тензора)
    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim)
                          )
        return Tensor(self.data.sum(dim))

    # Операция матричного умножения элементов матрицы (dim - размерность тензора)
    def mm(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data),
                          autograd=True,
                          creators=[self, other],
                          creation_op="mm")

        return Tensor(self.data.dot(other.data))

    # Cтроковое представление объекта
    def __repr__(self):
        return str(self.data.__repr__())

    # Преобразование в строку
    def __str__(self):
        return str(self.data.__str__())

    # Транспонирование матрицы
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())


    def sigmoid(self):
        if (self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if (self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")

        return Tensor(np.tanh(self.data))


    # Раширение матрицы (dim - количество осей массива (его размерность), copies - кол-во копий исходной матрицы)
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))

        trans_cmd.insert(dim, len(self.data.shape))  # добавление кол-ва размерностей исходнойматрицы
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    # Проверка, что градиент был ипользован от всех его родителей
    def is_all_children_grads_accounted_for(self):
        for k, v in self.children.items():
            if v != 0:
                return False
        return True

    # Обратное распространение сети
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                # есть градиент от родителя
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and (self.is_all_children_grads_accounted_for() or grad_origin is None):
                # Есть родители и (тензор получил все градиенты от весх потомков или grad_origin - пустой)
                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                if self.creation_op == "neg":
                    self.creators[0].backward(Tensor(self.grad.__neg__()))
                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)
                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)
                if self.creation_op == "mm":
                    act = self.creators[0]  # Обычно слой активации
                    weights = self.creators[1]  # Обычно весовая матрица
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)
                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())
                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))
                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))



class Layer:
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters



class Linear(Layer):
    def __init__(self, inputs_dim, output_dim):
        super().__init__()
        W = np.random.randn(inputs_dim, output_dim) * np.sqrt(
            2.0 / (inputs_dim))  # случайные числа по Гауссовскому распределению
        self.weight = Tensor(W, autograd=True) # веса слоя
        self.bias = Tensor(np.zeros(output_dim), autograd=True) # кооректирующие веса слоя (байес)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    # Выполнение выявления признаков слоем
    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


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

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class SGD:
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters # глобальные параметры сети
        self.alpha = alpha  # регелировка скорости изменения глобальных параметров (коэффициент регуляризации)

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            # Изменяем значение глобальных парметров на градиент
            p.data -= p.grad.data * self.alpha
            if zero:
                # Забываем стрые градиенты
                p.grad.data *= 0



class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


np.random.seed(0)
data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True) # Инициализация данных
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True) # Инициализация меток
model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()]) # Инициализация модели сети
optim = SGD(parameters=model.get_parameters(), alpha=1) # Инициализация метода оптимизации
criterion = MSELoss() # Инициализация функции ошибки
for i in range(10):
    pred = model.forward(data)  # Прогноз
    loss = criterion.forward(pred, target)  # Сравнение
    loss.backward(Tensor(np.ones_like(loss.data)))  # Обучение
    optim.step() # Изменение весов
    print(loss)
