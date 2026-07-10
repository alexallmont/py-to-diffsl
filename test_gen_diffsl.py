# FIXME migrate to new AST based diffsl, rename to test_diffsl.py

# from gen_diffsl import *

# def test_gen_diffsl_lambda():
#     diffsl = gen_diffsl(lambda x, y: sin(1 + y + x * 5) * x)
#     assert str(diffsl) == "(sin((1 + y) + (x * 5)) * x)"

# def test_gen_diffsl_def():
#     def fn(x, y):
#         return y * sin(y + 2 + 5 * x)

#     diffsl = gen_diffsl(fn)
#     assert str(diffsl) == "(y * sin((y + 2) + (5 * x)))"
