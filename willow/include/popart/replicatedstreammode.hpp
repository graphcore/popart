// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_REPLICATEDSTREAMMODE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_REPLICATEDSTREAMMODE_HPP_

#include <ostream>

namespace popart {

/* ReplicatedStreamMode:
 *
 * Describes how variables are streamed to Replicas
 *
 * ReplicateStreamMode::Broadcast:
 *    The tensor is broadcast to all replicas
 *    Tensor is identical across all replicas
 *
 * ReplicateStreamMode::Replicate:
 *    The tensor has no constraint to be identical
 *    across replicas.
 *
 * The aforementioned stream is host-to-device.
 */
enum class ReplicatedStreamMode {
  Broadcast, //
  Replicate  //
};

std::ostream &operator<<(std::ostream &os, const ReplicatedStreamMode &tt);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_REPLICATEDSTREAMMODE_HPP_
