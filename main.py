import diffsl
import matplotlib.pyplot as plt
import numpy as np
import pydiffsol as ds

# Taken from Lotka-Volterra example https://pydiffsol.readthedocs.io/en/latest/examples/population_dynamics.html
if __name__ == "__main__":
    code = diffsl.system_to_diffsl(
        rhs=lambda y1, y2, a, b, c, d: [
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        ],
        state={"y1": 1, "y2": 1},
        params={"a": 2/3, "b": 4/3, "c": 1, "d": 1},
    )

    ode = ds.Ode(code)
    sol = ode.solve(np.array([]), 40.0)

    fig, ax = plt.subplots()
    ax.plot(sol.ts, sol.ys[0], label="prey")
    ax.plot(sol.ts, sol.ys[1], label="predator")
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    fig.savefig("prey_predator.svg")
