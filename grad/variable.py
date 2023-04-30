import numpy as np
from . import gate as g
import graphviz

is_grad_enabled = True


class Variable:
    def __init__(self, data, gate=None, requires_grad=True):
        if isinstance(data, Variable):
            self = data
            return
        elif isinstance(data, (int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
            data = int(data)
        elif isinstance(data, (float, np.float16, np.float32, np.float64)):
            data = float(data)
        else:
            raise TypeError('expected int or float, got ' +
                            str(type(data)) + " instead")
        self.data = data
        self.grad = 0
        if gate is None or not is_grad_enabled:
            self.gate = g.Identity()
        else:
            self.gate = gate
        self.requires_grad = requires_grad
        self.is_graph = False

    def __repr__(self):
        return f"{self.data}"

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __str__(self):
        return str(self.data)
    
    def __format__(self, *args, **kwargs):
        return self.data.__format__(*args, **kwargs)

    def backward(self, grad=None, make_graph=False):
        if not self.requires_grad or not is_grad_enabled:
            return
        if grad is None:
            if make_graph:
                grad = Variable(1, requires_grad=False)
            else:
                grad = 1
        self.grad += grad
        self.gate.backward(grad=grad, make_graph=make_graph)
    
    def draw_graph(self, graph=None):
        if graph is None:
            graph = graphviz.Digraph()
        if not self.is_graph:
            if self.gate is not None and self.gate.name == 'identity' and self.requires_grad:
                graph.node(str(id(self)), f'{self.data:.4g}', style='filled', fillcolor='lightblue')
            else:
                graph.node(str(id(self)), f'{self.data:.4g}')
            self.is_graph = True
        if self.gate is not None and self.gate.name != 'identity':
            graph.node(str(id(self.gate)), self.gate.name, style='filled', fillcolor='lightgreen')
            label = None
            if self.requires_grad:
                label = f'{self.grad:.4g}'
            graph.edge(str(id(self.gate)), str(id(self)), label=label)
            self.gate.draw_graph(graph)
        return graph
    
    def clear_graph(self):
        self.is_graph = False
        self.gate.clear_graph()


    def zero_grad(self):
        if not self.requires_grad or not is_grad_enabled:
            return
        self.grad = 0
        self.gate.zero_grad()

    def __eq__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data == other.data

    def __lt__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data < other.data

    def __gt__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data > other.data

    def __le__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data <= other.data

    def __ge__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data >= other.data

    def __ne__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return self.data != other.data

    def __pos__(self):
        return self

    def __neg__(self):
        gate = g.Neg()
        return Variable(gate(self), gate=gate)

    def __abs__(self):
        gate = g.Abs()
        return Variable(gate(self), gate=gate)

    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        gate = g.Add()
        return Variable(gate(self, other), gate=gate)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        gate = g.Neg()
        other = Variable(gate(other), gate=gate)
        gate = g.Add()
        return Variable(gate(self, other), gate=gate)

    def __rsub__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        gate = g.Mul()
        return Variable(gate(self, other), gate=gate)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        gate = g.Div()
        return Variable(gate(self, other), gate=gate)

    def __rtruediv__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return other / self

    def __pow__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        gate = g.Pow()
        return Variable(gate(self, other), gate=gate)

    def __rpow__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other, requires_grad=False)
        return other ** self

    def sqrt(self):
        return self ** 0.5

    def sin(self):
        gate = g.Sin()
        return Variable(gate(self), gate=gate)

    def arcsin(self):
        gate = g.Asin()
        return Variable(gate(self), gate=gate)

    def sinh(self):
        gate = g.Sinh()
        return Variable(gate(self), gate=gate)

    def arcsinh(self):
        gate = g.Asinh()
        return Variable(gate(self), gate=gate)

    def cos(self):
        gate = g.Cos()
        return Variable(gate(self), gate=gate)

    def arccos(self):
        gate = g.Acos()
        return Variable(gate(self), gate=gate)

    def cosh(self):
        gate = g.Cosh()
        return Variable(gate(self), gate=gate)

    def arccosh(self):
        gate = g.Acosh()
        return Variable(gate(self), gate=gate)

    def tan(self):
        gate = g.Tan()
        return Variable(gate(self), gate=gate)

    def arctan(self):
        gate = g.Atan()
        return Variable(gate(self), gate=gate)

    def tanh(self):
        gate = g.Tanh()
        return Variable(gate(self), gate=gate)

    def arctanh(self):
        gate = g.Atanh()
        return Variable(gate(self), gate=gate)

    def exp(self):
        gate = g.Exp()
        return Variable(gate(self), gate=gate)

    def log(self):
        gate = g.Log()
        return Variable(gate(self), gate=gate)

    def conjugate(self):
        return self


def array(data, requires_grad=True):
    if isinstance(data, Variable):
        return data
    if isinstance(data, (list, tuple)):
        return [array(x, requires_grad=requires_grad) for x in data]
    if isinstance(data, np.ndarray):
        return np.array([array(x, requires_grad=requires_grad) for x in data])
    return Variable(data, requires_grad=requires_grad)


def array_grad(data):
    if isinstance(data, Variable):
        return data.grad
    if isinstance(data, (list, tuple)):
        return [array_grad(x) for x in data]
    if isinstance(data, np.ndarray):
        return np.array([array_grad(x) for x in data])
    return data


def array_zero_grad(data):
    if isinstance(data, Variable):
        data.zero_grad()
    elif isinstance(data, (list, tuple)):
        for x in data:
            array_zero_grad(x)
    elif isinstance(data, np.ndarray):
        for x in data:
            array_zero_grad(x)
