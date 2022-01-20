// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/popx/preparedtensor.hpp>

namespace popart {
namespace popx {

PreparedTensorInfo::PreparedTensorInfo(
    const TensorId &srcId_,
    const TensorId &dstId_,
    CanAlias canAlias_,
    RequireParallelWritable requireParallelWritable_)
    : srcId(srcId_), dstId(dstId_), canAlias(canAlias_),
      requireParallelWritable(requireParallelWritable_) {}

} // namespace popx
} // namespace popart
