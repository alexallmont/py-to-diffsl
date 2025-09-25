import inspect

class PyToDiffslExpr:
    """
    Diffsl generator class, uses python operator precedence to expand into
    sub expressions. For now this is done simply with strings but a more
    comprehensive builder method is preferable.
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    # TODO add remaining methods (div, sub, etc.), use setattr to auto generate

    def __add__(self, rhs):
        return PyToDiffslExpr(f"({self} + {rhs})")

    def __radd__(self, rhs):
        return PyToDiffslExpr(f"({rhs} + {self})")

    def __mul__(self, rhs):
        return PyToDiffslExpr(f"({self} * {rhs})")

    def __rmul__(self, rhs):
        return PyToDiffslExpr(f"({rhs} * {self})")


# TODO add other methods supported by diffsl
def sin(v):
    return PyToDiffslExpr(f"sin{v}")


def gen_diffsl(fn):
    """
    Generate diffsl from python expression

    The function arguments are extracted as the diffsl params and to create the
    expressions used to evaluate the diffsl code from the provided function.
    Any equation in the passed function is not itself evaluated mathematically,
    it's purely using python as one domain-specific language (DSL) as a means to
    generate another DSL, in this case diffsl.
    """
    signature = inspect.signature(fn)
    params = []
    for name in signature.parameters.keys():
        params.append(PyToDiffslExpr(name))
    return fn(*params)
