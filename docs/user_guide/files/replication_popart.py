# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import numpy
from popart import CommGroup, CommGroupType
from popart import VariableRetrievalMode, VariableSettings

builder = popart.Builder()

# replication factor
repl_factor = 4

# Simple base shape of variable on replica
base_shape = [3, 5]

# size of each group
group_size = 2

# The CommGroup we plan to use
communication_group = CommGroup(CommGroupType.Consecutive, group_size)

# VariableSettings to read from groups
settings_grouped    = VariableSettings(\
                            communication_group,\
                            VariableRetrievalMode.OnePerGroup)

# VariableSettings to read from all replicas
settings_individual = VariableSettings(\
                            communication_group,\
                            VariableRetrievalMode.AllReplicas)

# get init buffer:
num_groups = settings_grouped.groupCount(repl_factor)
shape = [int(repl_factor / num_groups)] + base_shape
initializer = numpy.zeros(shape).astype(numpy.float32)  # example

print(initializer.dtype)

# Creating Variables
a = builder.addInitializedInputTensor(initializer, settings_grouped)
b = builder.addInitializedInputTensor(initializer, settings_individual)

# get IO buffer shapes
shape_a = [settings_grouped.numReplicasReturningVariable(repl_factor)] \
            + base_shape
shape_b = [settings_individual.numReplicasReturningVariable(repl_factor)] \
            + base_shape

# get IO buffers
buffer_a = numpy.ndarray(shape_a)
buffer_b = numpy.ndarray(shape_b)

# finalize IO buffers
weightsIo = popart.PyWeightsIO({a: buffer_a, b: buffer_b})
