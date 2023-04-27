from . import variable as vp
from .variable import Variable as vn

class no_grad:
    def __init__(self):
        pass

    def __enter__(self):
        vp.is_grad_enabled = False

    def __exit__(self, exc_type, exc_value, traceback):
        vp.is_grad_enabled = True