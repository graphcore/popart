# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
Demonstrate nested code loading.
'''

import popxl
import popxl.ops as ops


def example():
    # Creating a model with popxl
    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(1, popxl.float32)

    # BEGIN_BASIC

    def expensive_id(x: popxl.Tensor) -> popxl.Tensor:
        return x.T.T

    g1 = ir.create_graph(expensive_id, x.spec)

    def load_g1():
        ops.remote_code_load(g1, destination='executable')

    g2 = ir.create_graph(load_g1)

    with ir.main_graph, popxl.in_sequence():
        # Loads code for g1
        ops.remote_code_load(g1, destination='executable')

        # Loads code for g1
        ops.call(g2)

        # Execute g1
        ops.call(g1, x)

    # END_BASIC

    # BEGIN_NOT_DYN

    def load_graph(g: popxl.Graph):
        ops.remote_code_load(g, destination='executable')

    g3 = ir.create_graph(load_graph, g1)
    g4 = ir.create_graph(load_graph, g2)

    with ir.main_graph, popxl.in_sequence():
        ops.remote_code_load(g3, destination='executable')
        ops.call(g3)

        ops.remote_code_load(g4, destination='executable')
        ops.call(g4)

    # END_NOT_DYN

    # run the model
    with popxl.Session(ir, "ipu_model") as session:
        _ = session.run()


if __name__ == "__main__":
    example()
