from diffsl import DiffslSystemBuilder, system_to_diffsl, sin

def test_diffsl_system_builder():
    code = DiffslSystemBuilder() \
        .rhs(lambda x, y: sin(1 + y + x * 5) * x) \
        .state("x", 0) \
        .param("y", 1) \
        .to_diffsl(newline=False)

    assert code == "y { 1 } u_i { x = 0, } F_i { sin(1 + y + x * 5) * x, }"

def test_diffsl_to_diffsl():
    def fn(x, y):
        return y * sin(y + 2 + 5 * x)

    code = system_to_diffsl(
        rhs=fn,
        state={"x": 0},
        params={"y": 1},
        newline=False
    )

    assert code == "y { 1 } u_i { x = 0, } F_i { y * sin(y + 2 + 5 * x), }"
