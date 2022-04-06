# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl

base_shape = [3, 3]

replication_factor = 8

# Create a tensor with default settings, that is: load same value to all replicas.
tensor_1 = popxl.variable(np.ndarray(base_shape))

# Create a tensor with one variable on each of the replicas:
tensor_2 = popxl.variable(np.ndarray([replication_factor] + base_shape),
                          ir.replica_grouping(group_size=1))

# Create a tensor where two and two replicas are grouped together
group_size = 2
tensor_3 = popxl.variable(
    np.ndarray([replication_factor / group_size] + base_shape),
    ir.replica_grouping(group_size=2))

# Create a tensor where tensors are grouped with an orthogonal replica.
tensor_3 = popxl.variable(
    np.ndarray([replication_factor / group_size] + base_shape),
    ir.replica_grouping(stride=4))
