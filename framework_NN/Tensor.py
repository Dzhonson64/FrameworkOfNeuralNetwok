# Класс, отвечающий за операции при прямом распространении, хранение данные, градиента и его обратного распространения

import numpy as np


class Tensor:

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None

        if id is None:
            self.id = np.random.randint(0, 1000000000)
        else:
            self.id = id

        self.creators = creators  # потомки
        self.creation_op = creation_op  # операция, которая породила данные тензор
        self.children = {}  # словарь (id-потомка : кол-во раз, когда  потомок участвовал в создании данного тензора)

        if creators is not None:
            # Вычисление кол-ва потомков
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    # Проверка на то, чтобы все потомки получили градиент от всех родителей
    def is_all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    # Обратное распространение ошибки
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            # авто-градиент включен
            if grad is None:
                # создание градиента по умолчанию (для проведения тестов)
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                # есть градиент от родителя
                if self.children[grad_origin.id] == 0:
                    # пытаемся взять градиент, у которомы мы уже его брали
                    return
                    print(self.id)
                    print(self.creation_op)
                    print(len(self.creators))
                    for c in self.creators:
                        print(c.creation_op)
                    raise Exception("cannot backprop more than once")
                else:
                    # сообщаем родителю, что от градиент был успешно взят
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                # градиент для переменной запоминается впервые
                self.grad = grad
            else:
                self.grad += grad

            assert grad.autograd == False


            # Продолжаем обратное распространие, если есть ещё потомки в графе, к которым можно продолжить обратное распространение
            # и если все граденты от всех потомков был учтён или нет градиента, который содеражит родитель
            if self.creators is not None and (self.is_all_children_grads_accounted_for() or grad_origin is None):
                # Есть родители и (тензор получил все градиенты от весх потомков или grad_origin - пустой)


                if self.creation_op == "add":
                    # Данные тензор породила операция Сложение
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "sub":
                    # Данные тензор породила операция Вычитание
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if self.creation_op == "mul":
                    # Данные тензор породила операция Умножение
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "mm":
                    # Данные тензор породила операция Матричное умножение
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)

                if self.creation_op == "transpose":
                    # Данные тензор породила операция Транспонирование матрицы
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    # Данные тензор породила операция Суммирование элементов
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if "expand" in self.creation_op:
                    # Данные тензор породила операция Расширение матрицы
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "neg":
                    # Данные тензор породила операция Обращение знака
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sigmoid":
                    # Данные тензор породила функция Сигмойда
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == "tanh":
                    # Данные тензор породила функция Тангенсоида
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == "index_select":
                    # Данные тензор породила операция Матричное умножение
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == "cross_entropy":
                    # Данные тензор породила операция Кросс-энтропия
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    # Суммирование элементов
    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    # Раширение матрицы (dim - количество осей массива (его размерность), copies - кол-во копий исходной матрицы)
    def expand(self, dim, copies):

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    # Транспонирование матрицы
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())

    # Матричное умножение
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    # ф-я сигмоида
    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    # ф-я тангенсоида
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):

        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    # ф-я софтмакс
    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)
        return softmax_output

    # ф-я кросс-энтропия
    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
