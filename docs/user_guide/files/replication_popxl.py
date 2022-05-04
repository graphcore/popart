# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import numpy as np

ir = popxl.Ir()

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

# Create a tensor which is grouped across sequential replicas (0 and 1, 2 and 3) and
# return all the group's variables when requested. The returned array will be of shape
# [replication_factor] + base_shape
group_size = 2
tensor_4 = popxl.variable(
    np.ndarray([replication_factor / group_size] + base_shape),
    ir.replica_grouping(group_size=2),
    retrieval_mode="all_replicas")

# Create a tensor which is grouped across orthogonal replicas (0 and 2, 1 and 3)
# and return all the group's variables when requested. The returned array will be of shape
# [replication_factor] + base_shape
tensor_5 = popxl.variable(
    np.ndarray([replication_factor / group_size] + base_shape),
    ir.replica_grouping(stride=4),
    retrieval_mode="all_replicas")
