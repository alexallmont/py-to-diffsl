# Python to diffsl

Prototype to trace Python (a la JAX) to lower into diffsl code.

This is experimental code and may not be released, as it it's not yet clear
if this can work with the tensor types and conditions that diffsl supports.

## Lotka-Volterra example:

Run with `PYTHONPATH=src/diffsl python examples/main.py`

```py
    code = diffsl.system_to_diffsl(
        rhs=lambda y1, y2, a, b, c, d: [
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        ],
        state={"y1": 1, "y2": 1},
        params={"a": 2/3, "b": 4/3, "c": 1, "d": 1},
    )
    print(code)
```
