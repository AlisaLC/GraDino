import numpy as np


class Gate:
    def __init__(self, name):
        self.name = name
        self.vars = []

    def __repr__(self):
        return f"Gate({self.name})"

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class Identity(Gate):
    def __init__(self):
        super().__init__("identity")

    def forward(self, x):
        return x

    def backward(self, grad=1):
        return grad


class Add(Gate):
    def __init__(self):
        super().__init__("add")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data + y.data

    def backward(self, grad):
        self.vars[0].backward(grad)
        self.vars[1].backward(grad)


class Mul(Gate):
    def __init__(self):
        super().__init__("mul")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data * y.data

    def backward(self, grad):
        self.vars[0].backward(grad * self.vars[1].data)
        self.vars[1].backward(grad * self.vars[0].data)


class Neg(Gate):
    def __init__(self):
        super().__init__("neg")

    def forward(self, x):
        self.vars = [x]
        return -x.data

    def backward(self, grad):
        self.vars[0].backward(-grad)


class Abs(Gate):
    def __init__(self):
        super().__init__("abs")

    def forward(self, x):
        self.vars = [x]
        return abs(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * (-1 if self.vars[0].data < 0 else 1))


class Div(Gate):
    def __init__(self):
        super().__init__("div")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data / y.data

    def backward(self, grad):
        if self.vars[0].requires_grad:
            self.vars[0].backward(grad / self.vars[1].data)
        if self.vars[1].requires_grad:
            self.vars[1].backward(-grad * self.vars[0].data /
                                  self.vars[1].data ** 2)


class Pow(Gate):
    def __init__(self):
        super().__init__("pow")

    def forward(self, x, y):
        self.vars = [x, y]
        return x.data ** y.data

    def backward(self, grad):
        if self.vars[0].requires_grad:
            self.vars[0].backward(
                grad * self.vars[1].data * self.vars[0].data ** (self.vars[1].data - 1))
        if self.vars[1].requires_grad:
            self.vars[1].backward(grad * self.vars[0].data **
                                  self.vars[1].data * np.log(self.vars[0].data))


class Sin(Gate):
    def __init__(self):
        super().__init__("sin")

    def forward(self, x):
        self.vars = [x]
        return np.sin(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * np.cos(self.vars[0].data))


class Asin(Gate):
    def __init__(self):
        super().__init__("asin")

    def forward(self, x):
        self.vars = [x]
        return np.arcsin(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / np.sqrt(1 - self.vars[0].data ** 2))


class Sinh(Gate):
    def __init__(self):
        super().__init__("sinh")

    def forward(self, x):
        self.vars = [x]
        return np.sinh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * np.cosh(self.vars[0].data))


class Asinh(Gate):
    def __init__(self):
        super().__init__("asinh")

    def forward(self, x):
        self.vars = [x]
        return np.arcsinh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / np.sqrt(1 + self.vars[0].data ** 2))


class Cos(Gate):
    def __init__(self):
        super().__init__("cos")

    def forward(self, x):
        self.vars = [x]
        return np.cos(x.data)

    def backward(self, grad):
        self.vars[0].backward(-grad * np.sin(self.vars[0].data))


class Acos(Gate):
    def __init__(self):
        super().__init__("acos")

    def forward(self, x):
        self.vars = [x]
        return np.arccos(x.data)

    def backward(self, grad):
        self.vars[0].backward(-grad / np.sqrt(1 - self.vars[0].data ** 2))


class Cosh(Gate):
    def __init__(self):
        super().__init__("cosh")

    def forward(self, x):
        self.vars = [x]
        return np.cosh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * np.sinh(self.vars[0].data))


class Acosh(Gate):
    def __init__(self):
        super().__init__("acosh")

    def forward(self, x):
        self.vars = [x]
        return np.arccosh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / np.sqrt(self.vars[0].data ** 2 - 1))


class Tan(Gate):
    def __init__(self):
        super().__init__("tan")

    def forward(self, x):
        self.vars = [x]
        return np.tan(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / np.cos(self.vars[0].data) ** 2)


class Atan(Gate):
    def __init__(self):
        super().__init__("atan")

    def forward(self, x):
        self.vars = [x]
        return np.arctan(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / (1 + self.vars[0].data ** 2))


class Tanh(Gate):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        self.vars = [x]
        return np.tanh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * (1 - np.tanh(self.vars[0].data) ** 2))


class Atanh(Gate):
    def __init__(self):
        super().__init__("atanh")

    def forward(self, x):
        self.vars = [x]
        return np.arctanh(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / (1 - self.vars[0].data ** 2))


class Exp(Gate):
    def __init__(self):
        super().__init__("exp")

    def forward(self, x):
        self.vars = [x]
        return np.exp(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad * np.exp(self.vars[0].data))


class Log(Gate):
    def __init__(self):
        super().__init__("log")

    def forward(self, x):
        self.vars = [x]
        return np.log(x.data)

    def backward(self, grad):
        self.vars[0].backward(grad / self.vars[0].data)
