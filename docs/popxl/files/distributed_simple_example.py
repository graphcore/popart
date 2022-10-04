# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Code Begin
import numpy as np
import popxl

num_groups = 2
nelms_per_group = 4
host_shape = (num_groups, nelms_per_group)
v_data = np.arange(0, num_groups * nelms_per_group).reshape(host_shape)

ir = popxl.Ir(replication="popdist")
with ir.main_graph:
    grouping = ir.replica_grouping(group_size=1)
    v = popxl.variable(v_data, grouping=grouping)

    replica_shape = v.shape
    assert replica_shape == (nelms_per_group,)

    v += 1

with popxl.Session(ir, "ipu_hw") as sess:
    sess.run()
print(sess.get_tensor_data(v))
# Code End
