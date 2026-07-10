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


def to_diffsl(expr: Expr) -> str:
    match expr:
        case StateValue(name):
            return name
        case ParamValue(name):
            return name
        case Unary("-", arg):
            return f"(-{to_diffsl(arg)})"
        case Unary(op, arg):
            return f"{op}({to_diffsl(arg)})"
        case Binary(op, lhs, rhs):
            return f"({to_diffsl(lhs)} {op} {to_diffsl(rhs)})"
        case Call(func, args):
            return f"{func}({', '.join(to_diffsl(a) for a in args)})"
        case _:
            raise TypeError(type(expr))


class SystemBuilder:
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

    def to_diffsl(self) -> str:
        codelines = []
        for name, init in self.param_dict.items():
            codelines.append(f"{name} {{ {init} }}")

        codelines.append("u_i {")
        for name, value in self.state_dict.items():
            codelines.append(f"  {name} = {value},")
        codelines.append("}")

        codelines.append("F_i {")
        rhs = self._trace(self.rhs_fn)
        for expr in rhs:
            codelines.append(f"  {to_diffsl(expr)},")
        codelines.append("}")

        return "\n".join(codelines)


def to_diffsl(rhs: Callable, states: dict[str, Number], params: dict[str, Number]) -> str:
    builder = SystemBuilder()
    for name, init in states.items():
        builder.state(name, init)
    for name, value in params.items():
        builder.param(name, value)
    builder.rhs(rhs)
    return builder.to_diffsl()
