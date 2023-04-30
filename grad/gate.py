import numpy as np
import graphviz


class Gate:
    def __init__(self, name):
        self.name = name
        self.vars = []

    def __repr__(self):
        return f"Gate({self.name})"

    def forward(self):
        raise NotImplementedError

    def backward(self, grad, make_graph=False):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def draw_graph(self, graph):
        for var in self.vars:
            if not var.is_graph:
                if var.gate is not None and var.gate.name == 'identity' and var.requires_grad:
                    graph.node(
                        str(id(var)), f'{var.data:.4g}', style='filled', fillcolor='lightblue')
                else:
                    graph.node(str(id(var)), f'{var.data:.4g}')
                var.is_graph = True
            label = None
            if var.requires_grad:
                label = f'{var.grad:.4g}'
            graph.edge(str(id(var)), str(id(self)), label=label)
            var.draw_graph(graph)
    
    def clear_graph(self):
        for var in self.vars:
            var.clear_graph()

    def zero_grad(self):
        for var in self.vars:
            var.zero_grad()


class Identity(Gate):
    def __init__(self):
        super().__init__("identity")

    def forward(self, x):
        return x

    def backward(self, grad, make_graph=False):
        return grad


class Add(Gate):
    def __init__(self):
        super().__init__("add")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data + y.data

    def backward(self, grad, make_graph=False):
        self.vars[0].backward(grad, make_graph=make_graph)
        self.vars[1].backward(grad, make_graph=make_graph)


class Mul(Gate):
    def __init__(self):
        super().__init__("mul")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data * y.data

    def backward(self, grad, make_graph=False):
        var0, var1 = self.vars[0], self.vars[1]
        if not make_graph:
            var0, var1 = var0.data, var1.data
        self.vars[0].backward(grad * var1, make_graph=make_graph)
        self.vars[1].backward(grad * var0, make_graph=make_graph)


class Neg(Gate):
    def __init__(self):
        super().__init__("neg")

    def forward(self, x):
        self.vars = [x]
        return -x.data

    def backward(self, grad, make_graph=False):
        self.vars[0].backward(-grad, make_graph=make_graph)


class Abs(Gate):
    def __init__(self):
        super().__init__("abs")

    def forward(self, x):
        self.vars = [x]
        return abs(x.data)

    def backward(self, grad, make_graph=False):
        self.vars[0].backward(
            grad * (-1 if self.vars[0].data < 0 else 1), make_graph=make_graph)


class Div(Gate):
    def __init__(self):
        super().__init__("div")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data / y.data

    def backward(self, grad, make_graph=False):
        var0, var1 = self.vars[0], self.vars[1]
        if not make_graph:
            var0, var1 = var0.data, var1.data
        if self.vars[0].requires_grad:
            self.vars[0].backward(grad / var1, make_graph=make_graph)
        if self.vars[1].requires_grad:
            self.vars[1].backward(-grad * var0 / var1 **
                                  2, make_graph=make_graph)


class Pow(Gate):
    def __init__(self):
        super().__init__("pow")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data ** y.data

    def backward(self, grad, make_graph=False):
        var0, var1 = self.vars[0], self.vars[1]
        if not make_graph:
            var0, var1 = var0.data, var1.data
        if self.vars[0].requires_grad:
            self.vars[0].backward(grad * var1 * var0 **
                                  (var1 - 1), make_graph=make_graph)
        if self.vars[1].requires_grad:
            self.vars[1].backward(grad * var0 ** var1 *
                                  np.log(var0), make_graph=make_graph)


class Sin(Gate):
    def __init__(self):
        super().__init__("sin")

    def forward(self, x):
        self.vars = [x]
        return np.sin(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad * np.cos(var0), make_graph=make_graph)


class Asin(Gate):
    def __init__(self):
        super().__init__("asin")

    def forward(self, x):
        self.vars = [x]
        return np.arcsin(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(
            grad / np.sqrt(1 - var0 ** 2), make_graph=make_graph)


class Sinh(Gate):
    def __init__(self):
        super().__init__("sinh")

    def forward(self, x):
        self.vars = [x]
        return np.sinh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad * np.cosh(var0), make_graph=make_graph)


class Asinh(Gate):
    def __init__(self):
        super().__init__("asinh")

    def forward(self, x):
        self.vars = [x]
        return np.arcsinh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(
            grad / np.sqrt(1 + var0 ** 2), make_graph=make_graph)


class Cos(Gate):
    def __init__(self):
        super().__init__("cos")

    def forward(self, x):
        self.vars = [x]
        return np.cos(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(-grad * np.sin(var0), make_graph=make_graph)


class Acos(Gate):
    def __init__(self):
        super().__init__("acos")

    def forward(self, x):
        self.vars = [x]
        return np.arccos(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(-grad / np.sqrt(1 - var0 **
                              2), make_graph=make_graph)


class Cosh(Gate):
    def __init__(self):
        super().__init__("cosh")

    def forward(self, x):
        self.vars = [x]
        return np.cosh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad * np.sinh(var0), make_graph=make_graph)


class Acosh(Gate):
    def __init__(self):
        super().__init__("acosh")

    def forward(self, x):
        self.vars = [x]
        return np.arccosh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(
            grad / np.sqrt(var0 ** 2 - 1), make_graph=make_graph)


class Tan(Gate):
    def __init__(self):
        super().__init__("tan")

    def forward(self, x):
        self.vars = [x]
        return np.tan(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad / np.cos(var0) ** 2, make_graph=make_graph)


class Atan(Gate):
    def __init__(self):
        super().__init__("atan")

    def forward(self, x):
        self.vars = [x]
        return np.arctan(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad / (1 + var0 ** 2), make_graph=make_graph)


class Tanh(Gate):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        self.vars = [x]
        return np.tanh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(
            grad * (1 - np.tanh(var0) ** 2), make_graph=make_graph)


class Atanh(Gate):
    def __init__(self):
        super().__init__("atanh")

    def forward(self, x):
        self.vars = [x]
        return np.arctanh(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad / (1 - var0 ** 2), make_graph=make_graph)


class Exp(Gate):
    def __init__(self):
        super().__init__("exp")

    def forward(self, x):
        self.vars = [x]
        return np.exp(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad * np.exp(var0), make_graph=make_graph)


class Log(Gate):
    def __init__(self):
        super().__init__("log")

    def forward(self, x):
        self.vars = [x]
        return np.log(x.data)

    def backward(self, grad, make_graph=False):
        var0 = self.vars[0]
        if not make_graph:
            var0 = var0.data
        self.vars[0].backward(grad / var0, make_graph=make_graph)
