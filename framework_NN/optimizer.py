# Оптимизоторы


# Стохастический градиентный спуск
class SGD():

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters  # глобальные параметры сети
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
