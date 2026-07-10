from textwrap import dedent

from diffsl import system_to_diffsl

# Source diffsl for these examples from https://github.com/alexallmont/pydiffsol/blob/main/examples

def test_pydiffsol_example_1_1_population_dynamics_solve():
    code = system_to_diffsl(
        rhs=lambda y1, y2, a, b, c, d: [
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        ],
        state={"y1": 1, "y2": 1},
        params={"a": 2/3, "b": 4/3, "c": 1, "d": 1},
    )

    assert code == dedent("""\
        a { 0.6666666666666666 }
        b { 1.3333333333333333 }
        c { 1 }
        d { 1 }
        u_i {
            y1 = 1,
            y2 = 1,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }""")
