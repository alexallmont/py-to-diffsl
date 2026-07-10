import diffsl as dsl
import numpy as np
import pydiffsol as ds


def minimal_system_builder_example():
    def fn(x):
        return [dsl.sin(1 + x)]

    code = dsl.SystemBuilder() \
        .rhs(fn) \
        .state("x", 0) \
        .param("y", 1) \
        .to_diffsl()

    print(code)


def lotka_volterra_example():
    code = dsl.to_diffsl(
        rhs=lambda y1, y2, a, b, c, d: [
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        ],
        state={"y1": 1, "y2": 1},
        params={"a": 2/3, "b": 4/3, "c": 1, "d": 1},
    )

    ode = ds.Ode(code)
    solution = ode.solve(np.array([]), 40.0)
    print(solution.ys)


if __name__ == "__main__":
    lotka_volterra_example()
