# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
Demonstration of the code loading API in PopXL.
'''

import popxl
import popxl.ops as ops


def example():
    # Creating a model with popxl
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(1, popxl.float32)

    def expensive_id(x: popxl.Tensor) -> popxl.Tensor:
        return x.T.T

    g = ir.create_graph(expensive_id, x.spec)

    # BEGIN_BASIC

    with ir.main_graph:
        # (1) insert many ops ...

        with popxl.in_sequence():
            # (2) load the code from remote memory into compute tiles on-chip
            ops.remote_code_load(g, destination='executable')
            # call the graph
            ops.call(g, x)

            # insert more ops...

            # call the graph again
            ops.call(g, x)

            # (3) call the graph for the final time
            ops.call(g, x)

    # END_BASIC

    # BEGIN_COMPLEX

    with ir.main_graph:
        with popxl.in_sequence():
            # Dead...

            # Live
            ops.remote_code_load(g, destination='executable')
            ops.call(g, x)

            # Dead again, due to a subsequent load...

            # Live again
            ops.remote_code_load(g, destination='executable')
            ops.call(g, x)

            # Dead again, due to a subsequent load...

            # Live again
            ops.remote_code_load(g, destination='executable')
            ops.call(g, x)

            # Dead again, as graph never called again

    # END_COMPLEX

    # run the model
    with popxl.Session(ir, "ipu_model") as session:
        _ = session.run()


if __name__ == "__main__":
    example()
