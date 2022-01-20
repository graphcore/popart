// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PREPAREDTENSOR_HPP
#define GUARD_NEURALNET_PREPAREDTENSOR_HPP

#include <popart/names.hpp>

namespace popart {
namespace popx {

enum class CanAlias { No = 0, Yes };

enum class RequireParallelWritable { No = 0, Yes };

/// Struct that describes how to prepare a required (snap/poplar) tensor for an
/// Opx during lowering.
struct PreparedTensorInfo {

  /**
   * Create PreparedTensorInfo instance.
   * \param srcId                    TensorId of a tensor that can be used as a
   *                                 source / reference. Can be empty.
   * \param dstId                    TensorId of the tensor to prepare.
   * \param canAlias                 If aliasing the srcId tensor to the dstId
   *                                 tensor is permissible.
   * \param requireParallelWritable  If the dstId tensor needs to be (parallel)
   *                                 writable.
   */
  PreparedTensorInfo(const TensorId &srcId,
                     const TensorId &dstId,
                     CanAlias canAlias,
                     RequireParallelWritable requireParallelWritable);

  TensorId srcId;
  TensorId dstId;
  CanAlias canAlias;
  RequireParallelWritable requireParallelWritable;
};

using PreparedTensorInfos = std::vector<PreparedTensorInfo>;

} // namespace popx
} // namespace popart

#endif
