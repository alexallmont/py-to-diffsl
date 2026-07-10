from gen_diffsl import *

if __name__ == "__main__":
    def fn(x, y):
        return sin(1 + x) * y

    # FIXME rename gen_diffsl to diffsl.trace?
    print(gen_diffsl(fn))

