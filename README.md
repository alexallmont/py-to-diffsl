# Python to diffsl

Prototype to trace python lowering into diffsl code (a la JAX).

This is early experimental code; it's not clear yet if this can work with
the tensor types and conditions that diffsl supports.

Lotka-Volterra example:

```py
    code = diffsl.system_to_diffsl(
        rhs=lambda y1, y2, a, b, c, d: [
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        ],
        state={"y1": 1, "y2": 1},
        params={"a": 2/3, "b": 4/3, "c": 1, "d": 1},
    )
```
