// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_STREAM_MODE_HPP
#define GUARD_STREAM_MODE_HPP

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

#endif // GUARD_STREAM_MODE_HPP
