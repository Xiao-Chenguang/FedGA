from itertools import chain


class ModelOper:
    def __add__(self, model):
        self_param = chain(self.parameters(), self.buffers())
        model_param = chain(model.parameters(), model.buffers())
        for p_self, p_model in zip(self_param, model_param):
            p_self.detach().add_(p_model)

    def __sub__(self, model):
        self_param = chain(self.parameters(), self.buffers())
        model_param = chain(model.parameters(), model.buffers())
        for p_self, p_model in zip(self_param, model_param):
            p_self.detach().sub_(p_model)

    def __mul__(self, factor):
        self_param = chain(self.parameters(), self.buffers())
        for p_self in self_param:
            p_self.detach().mul_(factor)

    def __truediv__(self, factor):
        self_param = chain(self.parameters(), self.buffers())
        for p_self in self_param:
            p_self.detach().copy_(p_self.detach().div(factor).type_as(p_self))
