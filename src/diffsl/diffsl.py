from dataclasses import dataclass
import inspect
from typing import Callable, Union, Sequence

# FIXME: TODO
# * Binary arithmetic: +, -, *, /, **, %, //
# * Unary arithmetic: -x, +x, abs(x)
# * Comparisons: <, <=, ==, !=, >=, >
# * Boolean/bitwise: &, |, ^, ~ (Python’s and/or cannot be overloaded)
# * Elementary functions: sin, cos, tan, asin, acos, atan, exp, log, sqrt, pow
# * Hyperbolic functions: sinh, cosh, tanh
# * Rounding: floor, ceil, round, trunc
# * Min/max: minimum, maximum, clip
# * Selection: where
# * Constants: pi, e

# For arrays, JAX also traces:

# * Indexing/slicing: x[i], x[:, 0]
# * Reshaping: reshape, transpose, broadcast_to, squeeze
# * Reductions: sum, prod, mean, max, min
# * Linear algebra: dot, matmul, solve, inv, etc.

Number = Union[int, float]


class Expr:
    def __add__(self, other): return Binary("+", self, ensure_expr(other))
    def __radd__(self, other): return Binary("+", ensure_expr(other), self)
    def __sub__(self, other): return Binary("-", self, ensure_expr(other))
    def __rsub__(self, other): return Binary("-", ensure_expr(other), self)
    def __mul__(self, other): return Binary("*", self, ensure_expr(other))
    def __rmul__(self, other): return Binary("*", ensure_expr(other), self)
    def __truediv__(self, other): return Binary("/", self, ensure_expr(other))
    def __rtruediv__(self, other): return Binary("/", ensure_expr(other), self)
    def __pow__(self, other): return Binary("**", self, ensure_expr(other))
    def __rpow__(self, other): return Binary("**", ensure_expr(other), self)
    def __neg__(self): return Unary("-", self)


@dataclass(frozen=True, slots=True)
class StateValue(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class ParamValue(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Unary(Expr):
    op: str
    arg: Expr


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    op: str
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Call(Expr):
    func: str
    args: tuple[Expr, ...]


def ensure_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return ParamValue(x)
    raise TypeError(f"Cannot convert {type(x)!r} to Expr")


def ensure_exprs(value):
    if isinstance(value, Expr):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [ensure_expr(v) for v in value]
    raise TypeError(f"Expected Expr or sequence of Expr, got {type(value)!r}")


def sin(x): return Call("sin", (ensure_expr(x),))
def cos(x): return Call("cos", (ensure_expr(x),))
def exp(x): return Call("exp", (ensure_expr(x),))
def log(x): return Call("log", (ensure_expr(x),))

PRECEDENCE = {
    "call": 100,
    "unary": 90,
    "**": 80,
    "*": 70,
    "/": 70,
    "+": 60,
    "-": 60,
}

def expr_to_diffsl(expr, parent_prec=0, is_right=False):
    match expr:
        case StateValue(name):
            s = name
            prec = 100
        case ParamValue(name):
            s = str(name)
            prec = 100
        case Call(func, args):
            s = f"{func}({', '.join(expr_to_diffsl(a) for a in args)})"
            prec = PRECEDENCE["call"]
        case Unary(op, arg):
            s = f"{op}{expr_to_diffsl(arg, PRECEDENCE['unary'])}"
            prec = PRECEDENCE["unary"]
        case Binary(op, lhs, rhs):
            prec = PRECEDENCE[op]
            left = expr_to_diffsl(lhs, prec, False)
            right = expr_to_diffsl(rhs, prec - 1 if op == "**" else prec, True)
            s = f"{left} {op} {right}"
        case _:
            raise TypeError(type(expr))

    need_parens = prec < parent_prec or (prec == parent_prec and is_right)
    return f"({s})" if need_parens else s


class DiffslSystemBuilder:
    def __init__(self):
        self.rhs_fn = None
        self.state_dict = {}
        self.param_dict = {}

    def state(self, name, init):
        self.state_dict[name] = init
        return self

    def param(self, name, value):
        self.param_dict[name] = value
        return self

    def rhs(self, fn):
        self.rhs_fn = fn
        return self

    def _trace(self, fn):
        sig = inspect.signature(fn)

        args = []
        for name in sig.parameters:
            if name in self.state_dict:
                args.append(StateValue(name))
            elif name in self.param_dict:
                args.append(ParamValue(name))
            else:
                raise KeyError(f"Unknown symbol: {name}")

        result = fn(*args)
        return ensure_exprs(result)

    def to_diffsl(self, newline: bool=True, indent_depth: int=4) -> str:
        if newline:
            delim = "\n"
            indent = " " * indent_depth
        else:
            delim = " "
            indent = ""

        codelines = []
        for name, init in self.param_dict.items():
            codelines.append(f"{name} {{ {init} }}")

        codelines.append("u_i {")
        for name, value in self.state_dict.items():
            codelines.append(f"{indent}{name} = {value},")
        codelines.append("}")

        codelines.append("F_i {")
        rhs = self._trace(self.rhs_fn)
        for expr in rhs:
            codelines.append(f"{indent}{expr_to_diffsl(expr)},")
        codelines.append("}")

        return delim.join(codelines)


def system_to_diffsl(
    rhs: Callable,
    state: dict[str, Number],
    params: dict[str, Number],
    newline: bool=True,
    indent_depth: int=4,
) -> str:
    builder = DiffslSystemBuilder()
    for name, init in state.items():
        builder.state(name, init)
    for name, value in params.items():
        builder.param(name, value)
    builder.rhs(rhs)
    return builder.to_diffsl(newline=newline, indent_depth=indent_depth)
